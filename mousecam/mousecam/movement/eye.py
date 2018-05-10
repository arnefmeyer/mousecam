#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    functions/classes related to modeling of eye movements using accel. signals
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn import neural_network
from sklearn import grid_search
from six import string_types

from scipy import interpolate


def create_regression_data(eye_ts, eye_xy, imu_ts, imu_data,
                           maxlag=0.2, binwidth=0.02, causal=True,
                           xy_shift=None, max_shift=25, **kwargs):

    v = np.logical_and(~np.isnan(eye_xy[:, 0]),
                       eye_ts > maxlag)
    if not causal:
        v = np.logical_and(v, eye_ts < imu_ts[-1] - maxlag)

    if xy_shift is not None:
        v = np.logical_and(v, np.max(np.abs(xy_shift), axis=1) < max_shift)

    D = max(int(round(maxlag / binwidth+.5)), 1)
    lags = -1*np.arange(D)[::-1] * binwidth
    if not causal:
        lags = np.append(lags, np.arange(1, D) * binwidth)
        D = len(lags)

    N = np.sum(v)
    n_channels = imu_data.shape[1]

    X = np.zeros((N, D*n_channels))
    Y = eye_xy[v, :]

    if xy_shift is not None:
        Y -= xy_shift[v, :]

    f = interpolate.interp1d(imu_ts, imu_data, axis=0,
                             bounds_error=False,
                             fill_value=0)

    for i, ts in enumerate(eye_ts[v]):

        for j in range(n_channels):

            A = f(ts + lags)
            X[i, :] = A.flatten('F')

    return X, Y, np.where(v)[0], lags


def get_estimator(model, linear_estimator='ARD', n_channels=3,
                  optimize_hyperparameters=False,
                  **kwargs):

    if model == 'Linear':
        from lnpy import linear as linear_models
        if linear_estimator == 'ARD':
            estimator = linear_models.ARD(verbose=0)

        elif linear_estimator == 'Ridge':
            estimator = linear_models.Ridge(verbose=False)

    elif model == 'GAM':
        estimator = GAM(linear_model=linear_estimator,
                        n_channels=n_channels)

    elif model == 'MLP':
        estimator = MLP(optimize_hyperparameters=optimize_hyperparameters,
                        alpha=0.0001,
                        hidden_layer_sizes=(100, ),
                        activation='relu',
                        solver='adam',
                        **kwargs)

    return estimator


def get_linear_weights(model, n_channels=3, order='F',
                       cov_includes_bias=False):
    """extract linear weights for different accelerometer channels"""

    w = np.copy(model.get_weights())
    m = w.shape[0]/n_channels
    W = np.reshape(w, (m, n_channels), order=order)
#    chan_ind = np.reshape(np.arange(w.shape[0]), (m, n_channels),
#                          order='F')

    C = None
    if hasattr(model, 'cov_posterior'):

        if cov_includes_bias:
            cov = model.cov_posterior[:-1]
        else:
            cov = model.cov_posterior

        C = np.reshape(np.sqrt(np.diag(cov)),
                       (m, n_channels), order=order)

    return W, C


def plot_linear_model(model, ax=None, dimensions=['x', 'y', 'z'], **kwargs):

    W, C = get_linear_weights(model, **kwargs)

    if ax is None:
        ax = plt.gca()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    for i in range(W.shape[1]):

        w = W[:, i]

        xx = np.arange(w.shape[0]) - w.shape[0]
        if C is not None:
            ax.fill_between(xx, w - C[:, i], w + C[:, i], color=colors[i],
                            alpha=.5, lw=0)

        ax.plot(xx, w, '-', color=colors[i], lw=1, label=dimensions[i])

    ax.legend(loc='best', fontsize=6)

    return ax


class ChannelCorrelator():
    """pearson's correlation coefficient between traces"""

    def __init__(self, n_channels=3, **kwargs):

        self.n_channels = n_channels

    def xcorr(self, X, y):

        N, D = X.shape
        n_steps = D / self.n_channels
        ind = np.arange(self.n_channels) * n_steps

        cc = np.zeros((self.n_channels,))
        for i in range(self.n_channels):
            cc[i] = np.diag(np.corrcoef(X[:, ind[i]], y), 1)

        return cc


class GAM(BaseEstimator):
    """generalized additive model (GAM) using linearily filtered input

        Estimation a GAM using all dimensions is not working well.
        Thus we try to estimate the linear weights using a linear model
        (ridge regression or ARD) and then fit a GAM to the linear predictions
        for each channel.

        This implementation is using the mgcv R package via rpy2 the rpy2
        interface. Unfortunately, there are different versions of rpy2
        (and the integrated pandas support) so some parts might seem a bit
        hacky but allow running the code with most versions.

        The linear models requre the lnpy package (srf-review branch):
        https://github.com/arnefmeyer/lnpy/tree/srf-review
    """
    def __init__(self, linear_model='Ridge', n_channels=3, **kwargs):

        self.linear_model = linear_model
        self.n_channels = n_channels
        self._model = None

    @property
    def columns(self):

        columns = ['y']
        columns.extend(['x%d' % (d+1) for d in range(self.n_channels)])

        return columns

    @property
    def model_string(self):

        mod = 'y ~'
        for d in range(self.n_channels):
            mod += ' s(x%d) +' % (d+1)

        return mod[:-2]

    def fit(self, X, y):

        from lnpy import linear as linear_models

        n_channels = self.n_channels

        # estimation a GAM using all dimensions is not working well;
        # thus we try to estimate the linear weights using a linear model
        # and then fit a GAM to the linear predictions for each channel

        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri
        import rpy2.robjects as ro
        pandas2ri.activate()
        import pandas as pd
        try:
            import pandas.rpy.common as com
            com_available = True
        except BaseException:
            com_available = False

        mgcv = importr('mgcv')

        if self.linear_model is None:

            lin_model = linear_models.ARD(verbose=False)
            lin_model.fit(X, y)

        elif isinstance(self.linear_model, string_types):

            if self.linear_model.upper() == 'ARD':
                lin_model = linear_models.ARD(verbose=False)
            elif self.linear_model.upper() == 'RIDGE':
                lin_model = linear_models.Ridge(verbose=False)

            lin_model.fit(X, y)

        N = X.shape[0]
        w = np.copy(lin_model.get_weights())
        m = w.shape[0]/n_channels
        chan_ind = np.reshape(np.arange(w.shape[0]), (m, n_channels),
                              order='F')

        Yw_pred = np.zeros((N, n_channels))
        for j in range(n_channels):

            # predictions for channel j
            Yw_pred[:, j] = np.dot(X[:, chan_ind[:, j]], w[chan_ind[:, j]])

        # fit GAM
        YX = np.hstack((np.atleast_2d(y).T, Yw_pred))
        df = pd.DataFrame(YX, columns=self.columns)
        if com_available:
            df_r = com.convert_to_r_dataframe(df)
        else:
            try:
                df_r = pandas2ri.py2ri(df)
            except BaseException:
                df_r = pandas2ri.pandas2ri(df)

        mod = self.model_string

        m = mgcv.gam(ro.r(mod),
                     data=df_r,
                     family='gaussian()',
                     optimizer='perf')

        self._model = m
        self._linear_model = lin_model

    def predict(self, X):

        if self._model is not None:

            from rpy2.robjects.packages import importr
            from rpy2.robjects import pandas2ri
            mgcv = importr('mgcv')

            import pandas as pd

            n_channels = self.n_channels
            lin_model = self._linear_model

            N = X.shape[0]
            w = np.copy(lin_model.get_weights())
            m = w.shape[0]/n_channels
            chan_ind = np.reshape(np.arange(w.shape[0]), (m, n_channels),
                                  order='F')

            Yw_pred = np.zeros((N, n_channels))
            for j in range(n_channels):

                # predictions for channel j
                Yw_pred[:, j] = np.dot(X[:, chan_ind[:, j]], w[chan_ind[:, j]])

            y = np.zeros((N,))
            YX = np.hstack((np.atleast_2d(y).T, Yw_pred))
            df = pd.DataFrame(YX, columns=self.columns)

            try:
                df_r = pandas2ri.py2ri(df)
            except BaseException:
                df_r = pandas2ri.pandas2ri(df)

            y_pred = mgcv.predict_gam(self._model, df_r, type="response",
                                      se="False")

            return np.asarray(y_pred)

        else:
            raise ValueError('Fit model first!')


class MLP(neural_network.MLPRegressor):
    """multilayer perceptron-based estimator

        This is actually a wrapper around sklearns MLPRegressor class that
        includes automatic parameter grid search and a z-score normalizer.
        However, the default alpha parameter values worked perfectly fine
        when using ADAM so it might not be needed in most cases.
    """
    def __init__(self, alpha_values=None, n_jobs=-1, n_folds=5,
                 alpha=0.0001,
                 optimize_hyperparameters=False,
                 normalize=True, **kwargs):

        if 'loss' in kwargs:
            kwargs.pop('loss')

        super(MLP, self).__init__(**kwargs)

        if alpha_values is None:
            alpha_values = 2. ** np.linspace(-10, 10, 10)

        self.alpha_values = alpha_values
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.optimize_hyperparameters = optimize_hyperparameters

    def get_params(self, deep=False):

        return self.__dict__

    def fit(self, X, y):

        if self.optimize_hyperparameters:

            cv = self.n_folds
            param_grid = {'alpha': self.alpha_values}

            self.optimize_hyperparameters = False
            grid = grid_search.GridSearchCV(self, param_grid,
                                            n_jobs=self.n_jobs,
                                            iid=True,
                                            refit=False,
                                            cv=cv,
                                            verbose=1)
            grid.fit(X, y)

            self.alpha = grid.best_params_['alpha']
            print(self.alpha, self.alpha_values[0], self.alpha_values[-1])
            self.fit(X, y)
            self.optimize_hyperparameters = True

        else:
            super(MLP, self).fit(X, y)

    def predict(self, X, *args, **kwargs):

        return super(MLP, self).predict(X, *args, **kwargs)
