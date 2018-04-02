#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    accelerometer-related stuff
"""


import numpy as np
import matplotlib.pyplot as plt
import quantities as pq

# -----------------------------------------------------------------------------
# accelerometer signal wrapper classes
# -----------------------------------------------------------------------------

GRAVITY = 9.80665


class AccelerometerSignal(object):
    """Wrapper class for 3 axis accelerometer signals

        Parameters
        ==========
        xyz : array
            Array of shape (# samples, channels). If no unit is given it is
            assumed that values are given in volts. Python's quantity package
            is used to keep track of units.
        fs : float
            Samplerate in Hz
        channel_order : list
            Order of accelerometer channels
    """

    def __init__(self, xyz, fs, vmin=None, vmax=None,
                 units='g', g=GRAVITY,
                 channel_order=['x', 'y', 'z'],
                 reorder_channels=True,
                 invert_axes=False):

        if not hasattr(xyz, 'units'):
            xyz = pq.Quantity(xyz, units=pq.volt)

        self.xyz = xyz
        self.samplerate = fs
        self.units = units
        self.g = g

        if vmin is not None and vmax is not None:
            self._center_and_scale(vmin, vmax, units=units, g=g)

        if invert_axes:
            self.xyz *= -1

        if reorder_channels:
            order = [['x', 'y', 'z'].index(k) for k in channel_order]
            self.xyz = self.xyz[:, order]
            channel_order = ['x', 'y', 'z']

        self.channel_order = channel_order

        self._vmin = vmin
        self._vmax = vmax

    def filter(self, f_lower=0, f_upper=50):

        raise NotImplementedError()

    @property
    def magnitude(self):
        """agnitude of accelerometer signal"""
        return np.sqrt(np.sum(self.xyz ** 2, axis=1))

    def get_channel(self, dim):

        return self.xyz[:, self.channel_order.index(dim)]

    @property
    def num_samples(self):
        return self.xyz.shape[0]

    @property
    def timestamps(self):
        return np.arange(self.num_samples) / self.samplerate

    def invert_axis(self, which):

        index = self.channel_order.index(which)
        self.xyz[:, index] *= -1

    def _center_and_scale(self, vmin, vmax, units='g', g=GRAVITY):
        """scale voltage (or raw) signals to m/s^2

        For details on accelerometer calibration see
        http://intantech.com/files/Intan_RHD2000_accelerometer_calibration.pdf

        Parameters
        ==========
        vmin : array-like
            minimum voltages for each dimension (corresponding to
            -g = -9.81 m/s^2)
        vmax : array-like
            maximum voltages for each dimension (corresponding to
            g = 9.81 m/s^2)
        units : str
            Either 'm/s^2' or 'g' for gratitaional acceleration (9.81 m/s^2)
        g : float
            The value for g -> m/s^2 conversion; don't use quantities'
            predefined g_0 as it is a bit tricky to then convert it to m/s^2.
        """

        vmin = pq.Quantity(np.asarray(vmin), pq.volt)
        vmax = pq.Quantity(np.asarray(vmax), pq.volt)

        if not hasattr(g, 'units'):
            g = pq.Quantity(g, pq.CompoundUnit('m/s^2'))

        # remove static acceleration
        bias = vmin + .5*(vmax - vmin)
        self.xyz -= bias

        # scale to g (= static acceleration range (-1, 1))
        self.xyz /= (vmax - vmin) / (2*g.units)

        if units == pq.CompoundUnit('m/s^2') or units == 'm/s^2':

            self.xyz *= g.item()
            self.xyz *= pq.CompoundUnit('m/s^2')

    def show(self, show_now=True):
        """plot accelerometer channels"""

        n_samples, n_channels = self.xyz.shape
        fig, axarr = plt.subplots(nrows=n_channels, ncols=1,
                                  sharex=True, sharey=True)
        axarr = np.atleast_1d(axarr)

        t = np.arange(n_samples) / float(self.samplerate)
        label = 'Acceleration ({})'.format(self.xyz.units)

        dims = sorted(self.channel_order)
        for i, ax in enumerate(axarr.flat):

            ind = self.channel_order.index(dims[i])
            ax.plot(t, self.xyz[:, ind])
            ax.set_title(dims[i])
            ax.set_ylabel(label)

            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        ax.set_xlabel('Time (s)')

        fig.tight_layout()

        if show_now:
            plt.show()


class RHD2132AccelerometerSignal(AccelerometerSignal):
    """Accelerometers on Intan's RHD2132 board have a different channel order

    For details see
    http://www.intantech.com/files/Intan_RHD2000_accelerometer_calibration.pdf

    The channels are reordered to match the eye's coordinate system.

    """
    def __init__(self, xyz, fs, channel_order=['z', 'x', 'y'], **kwargs):

        super(RHD2132AccelerometerSignal, self).__init__(
            xyz, fs, channel_order=channel_order, **kwargs)
