#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    blubb
"""

from scipy import signal
import numpy as np


def filter_data_(data, fs, f_lower=300., f_upper=6000., order=2,
                 method='filtfilt', column_wise=False, filt_type='bandpass'):
    """simple bandbass filtering of electrode signals"""

#    if f_lower > 0:
    if filt_type == 'bandpass':
        Wn = (f_lower / fs * 2, f_upper / fs * 2)
        b, a = signal.butter(order, Wn, btype='bandpass', analog=False,
                             output='ba')
#    else:
    elif filt_type == 'lowpass':
        Wn = f_upper / fs * 2
        b, a = signal.butter(order, Wn, btype='lowpass', analog=False,
                             output='ba')
    elif filt_type == 'highpass':
        Wn = f_lower / fs * 2
        b, a = signal.butter(order, Wn, btype='highpass', analog=False,
                             output='ba')
    else:
        raise ValueError('Unknown filter type: %s' % filt_type)

    if method == 'filtfilt':
        filt_fun = signal.filtfilt
    elif method == 'lfilter':
        filt_fun = signal.lfilter
    else:
        raise ValueError('Invalid filtering method: ' + method)

    if column_wise:
        data_ = np.zeros_like(data)
        for i in range(data.shape[1]):
            data_[:, i] = filt_fun(b, a, data[:, i])
    else:
        data = filt_fun(b, a, data, axis=0)

    return data


def filter_data(data, fs, method='lfilter', adjust_delay=False, **kwargs):
    """filtering method that allows adjusting of FIR filter delay"""

    F = filter_data_(data, fs, method=method, **kwargs)

    if method == 'lfilter' and adjust_delay:

        # estimate delay by filtering a pulse
        N = int(2*fs)
        x = np.zeros((N,))
        pulse_index = int(N/2,)
        x[pulse_index] = 1
        y = filter_data_(x, fs, method=method, **kwargs)

        max_index = np.argmax(np.abs(y))
        dn = max_index - pulse_index
        if dn > 0:
            if F.ndim == 1:
                F = np.append(F[dn:], np.zeros((dn,), dtype=F.dtype))
            else:
                F = np.append(F[dn:, :], np.zeros((dn, F.shape[1]),
                              dtype=F.dtype), axis=0)

    return F


def corrlag(x, y, maxlag=1000, normalize=True, center=True):
    """correlation function with max. time lag"""

    assert x.shape[0] == y.shape[0]

    N = x.shape[0]
    i1 = N-1 - maxlag
    i2 = N-1 + maxlag+1

    if center:
        cxy = signal.correlate(x - x.mean(), y - y.mean(), 'full')
    else:
        cxy = signal.correlate(x, y, 'full')

    if normalize:
        cc = np.diag(np.corrcoef(x, y), 1)
        cxy = cxy / cxy[N-1] * cc

    return cxy[i1:i2], np.arange(-maxlag, maxlag+1)
