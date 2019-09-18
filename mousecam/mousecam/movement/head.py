#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    function to visualize head orientations based on accelerometer signals
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
import math


ISO_AZIMUTH = 45.
ISO_ELEVATION = np.arcsin(1/np.sqrt(3)) * 180 / np.pi

DEFAULT_AZIMUTH = 30.
DEFAULT_ELEVATION = 30.


# -----------------------------------------------------------------------------
# estimation of head orientation from accelerometer data
# -----------------------------------------------------------------------------

def cart2sph(xyz, degrees=False):
    """
    r: radius
    theta: angle between OP line and z-axis (0 <= theta <= pi)
    phi: azimuth angle in x-y-plane (0 <= phi <= 360)

    This function uses the ISO convention often used in physics:
    https://en.wikipedia.org/wiki/Spherical_coordinate_system
    http://mathworld.wolfram.com/SphericalCoordinates.html
    """

    if xyz.ndim == 1:
        xyz = np.atleast_2d(xyz)

    r = np.sqrt(np.sum(xyz**2, axis=1))
    theta = np.arccos(xyz[:, -1] / r)
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])

    if degrees:
        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)

    return r, theta, phi


def sph2cart(r, theta, phi, degrees=False):

    if degrees:
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)

    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)

    return np.vstack((x, y, z)).T


def rotate_spherical(xyz, theta, phi, **kwargs):

    r, T, P = cart2sph(xyz, **kwargs)
    xyz_ = sph2cart(r, T+theta, P+phi, **kwargs)

    return xyz_


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def compute_pitch_and_roll(xyz, fs,
                           f_cutoff=2.,
                           filt_type='lowpass',
                           method='filtfilt',
                           gravity=False):

    if f_cutoff is not None:
        # extract gravity component
        from ..util import signal as mcs
        xyz_lowpass = mcs.filter_data(xyz, fs,
                                      f_upper=f_cutoff,
                                      filt_type=filt_type,
                                      method=method)
    else:
        xyz_lowpass = xyz

    norm = np.sqrt(np.sum(xyz_lowpass**2, axis=1))

    # pitch (y/z plane with normal vector (1, 0, 0))
    pitch = np.dot((xyz_lowpass.T / norm).T, [1, 0, 0])

    # roll (x/z plane with normal vector (0, -1, 0))
    roll = np.dot((xyz_lowpass.T / norm).T, [0, 1, 0])

    if not gravity:
        pitch = 90 - np.degrees(np.arccos(pitch))
        roll = 90 - np.degrees(np.arccos(roll))

    return pitch, roll


def estimate_density_sphere(xyz,
                            step_theta=5,
                            step_phi=5,
                            method='hist',
                            oversample=1,
                            smoothing=0):

    theta_grid = np.arange(0, 180+.5, step_theta)
    phi_grid = np.arange(0, 360+.5, step_phi) - 180
    T, P = np.meshgrid(theta_grid, phi_grid)

    if method == 'vecquant':
        # approach: use a fine grid on a sphere with radius 1 and
        # approximate a histogram by some kind of vector quantization
        # of normalized acceleration vectors

        # transform to cartesian coordinates
        N = T.size
        xyz_grid = sph2cart(np.ones((N,)),
                            T.ravel(), P.ravel(),
                            degrees=True)
        H = np.zeros((N,))
        for i, vec in enumerate(xyz):
            index = np.argmin(np.sum((xyz_grid - vec)**2, axis=1))
            H[index] += 1

        # normalize by area of surface element
        areas = np.sin(np.deg2rad(T)) * np.deg2rad(step_theta) ** 2
        areas[areas < 1e-5] = 1e-5
        H = H / areas.ravel()

    elif method == 'hist':

        # histogram of surface patches
        radius, theta, phi = cart2sph(xyz, degrees=True)

        H = np.histogram2d(theta, phi, bins=(theta_grid, phi_grid))[0]

        dA = step_theta
        A = np.sin(np.deg2rad(theta_grid[:-1]+.5*dA)) * np.deg2rad(dA) ** 2

        H = (H.T / A).T

        # this is a bit hacky but for visualization purposes it is certainly
        # sufficient to not use the exact bin centers but to "close" the
        # sphere using the whole 360 degrees range
        n_theta = theta_grid.shape[0] - 1
        n_phi = phi_grid.shape[0] - 1

        phi_grid2 = np.linspace(phi_grid[0], phi_grid[-1], n_phi)
        theta_grid2 = np.linspace(theta_grid[0], theta_grid[-1], n_theta)

        if oversample > 1:
            # interpolate to get nice plot
            from scipy import interpolate

            f = interpolate.interp2d(phi_grid2, theta_grid2, H,
                                     kind='linear',
                                     copy=True,
                                     bounds_error=False,
                                     fill_value=np.NaN)

            # create finer grid
            n_phi = int(round(n_phi*oversample))
            n_theta = int(round(n_theta * oversample))
            phi_grid2 = np.linspace(phi_grid[0], phi_grid[-1],
                                    n_phi)
            theta_grid2 = np.linspace(theta_grid[0], theta_grid[-1],
                                      n_theta)
            H = f(phi_grid2, theta_grid2)

        if smoothing is not None and smoothing > 1:
            x = np.linspace(-3., 3., smoothing)
            xx, yy = np.meshgrid(x, x)
            G = np.exp((-xx**2 - yy**2))
            H = signal.convolve2d(H, G/G.sum(), mode='same', boundary='wrap')

        # make sure circular parts are consistent
        H[:, 0], H[:, -1] = (H[:, 0] + H[:, -1])/2., (H[:, 0] + H[:, -1])/2.

        # get bin centers
        P, T = np.meshgrid(phi_grid2, theta_grid2)
        theta_grid = theta_grid2
        phi_grid = phi_grid2
        xyz_centers = sph2cart(np.ones((n_theta * n_phi,)),
                               T.ravel(),
                               P.ravel(),
                               degrees=True)

    return xyz_centers, H.reshape(T.shape), theta_grid, phi_grid


def plot_density_sphere(xyz_centers, H, ax=None, method='hist',
                        azimuth=None, elevation=None, backend='visvis',
                        oversample=1, smoothing=None, zoom=1.7,
                        cmap='jet',
                        scaling='log',
                        log_offset=-.1,
                        clim=None,
                        isometric=False,
                        light_ambient=.6,
                        light_specular=.5,
                        avg_direction=None,
                        pitch_label=True,
                        roll_label=True,
                        pitch_arrow=True,
                        roll_arrow=True,
                        arrow_color=(.25, .95, .8),
                        close_figure=False):

    if azimuth is None:
        azimuth = DEFAULT_AZIMUTH
    if elevation is None:
        elevation = DEFAULT_ELEVATION

    scaling = scaling.lower()
    if scaling.startswith('log'):
        H = H / float(H.sum())
        v = H > 0
        if scaling == 'log':
            H[v] = np.log(H[v])
        elif scaling == 'log2':
            H[v] = np.log2(H[v])
        H[~v] = H[v].min() + log_offset

    if smoothing is not None and smoothing > 1:
        x = np.linspace(-3., 3., smoothing)
        xx, yy = np.meshgrid(x, x)
        G = np.exp((-xx**2 - yy**2))
        H = signal.convolve2d(H, G/G.sum(), mode='same', boundary='wrap')

    if backend in ['mpl', 'matplotlib']:

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=315, altdeg=45)
        cm = getattr(plt.cm, cmap)
        colors = ls.shade(H, cmap=cm, blend_mode='soft', vert_exag=1)
        ax.plot_surface(xyz_centers[:, 0].reshape(H),
                        xyz_centers[:, 1].reshape(H),
                        xyz_centers[:, 2].reshape(H),
                        cstride=1, rstride=1,
                        facecolors=colors,
                        linewidth=0,
                        antialiased=False,
                        shade=False)

        # add arrows in front of sphere
        import mousecam_helpers as helpers
        arrow_props = dict(arrowstyle='simple',
                           mutation_scale=15,
                           mutation_aspect=None,
                           linewidth=.5,
                           facecolor=3*[.75],
                           edgecolor=3*[.1])
        length = .6

        arr = helpers.Arrow3D((1, 1+length), (0, 0), (0, 0),
                              **arrow_props)
        ax.add_artist(arr)

        arr = helpers.Arrow3D((0, 0), (1, 1+length), (0, 0),
                              **arrow_props)
        ax.add_artist(arr)

        arr = helpers.Arrow3D((0, 0), (0, 0), (1, 1+length),
                              **arrow_props)
        ax.add_artist(arr)

        extents = 3*[[-.6, .6]]
        ax.auto_scale_xyz(*extents)
        ax.set_aspect('equal')
        ax.axis('off')

        ax.view_init(elev=elevation, azim=360-azimuth)

    elif backend in ['mayavi', 'mlab']:

        from mayavi import mlab

        fig = mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
        fig.scene.parallel_projection = True

        mlab.mesh(xyz_centers[:, 0].reshape(H.shape),
                  xyz_centers[:, 1].reshape(H.shape),
                  xyz_centers[:, 2].reshape(H.shape),
                  scalars=H, colormap='jet')

        from tvtk.tools import visual
        visual.set_viewer(fig)
        arrow_kw = dict(color=(.75, .75, .75),
                        length_cone=.1,
                        radius_cone=.05,
                        radius_shaft=.025)
        Arrow3D(0, 0, 0, 1.4, 0, 0, **arrow_kw)
        Arrow3D(0, 0, 0, 0, 1.4, 0, **arrow_kw)
        Arrow3D(0, 0, 0, 0, 0, 1.4, **arrow_kw)

        if isometric:
            fig.scene.isometric_view()
        else:
            mlab.view(azimuth=-azimuth, elevation=elevation, figure=fig)
        fig.scene.camera.zoom(zoom)

        ax = mlab.screenshot(figure=fig, mode='rgb', antialiased=False)

    elif backend in ['visvis']:

        import visvis as vv

        app = vv.use()

        fig = vv.figure()
        ax = vv.subplot(111)

        ax.axis.visible = 0
        ax.axis.xLabel = 'x'
        ax.axis.yLabel = 'y'
        ax.axis.zLabel = 'z'

        length = 1.4

        for i in range(3):
            direction = np.zeros((3,))
            direction[i] = 1
            cyl = vv.solidCylinder(translation=(0, 0, 0),
                                   scaling=(.05, .05, length),
                                   direction=tuple(direction),
                                   axesAdjust=False, axes=ax)
            cyl.faceColor = (.75, .75, .75)

            translation = np.zeros((3,))
            translation[i] = length
            cone = vv.solidCone(translation=tuple(translation),
                                scaling=(.1, .1, .2),
                                direction=tuple(direction),
                                axesAdjust=False, axes=ax)
            cone.faceColor = (.75, .75, .75)

        surf = vv.surf(xyz_centers[:, 0].reshape(H.shape),
                       xyz_centers[:, 1].reshape(H.shape),
                       xyz_centers[:, 2].reshape(H.shape),
                       H, axes=ax)

        # set colormap
        cmap = cmap.lower()
        if cmap.endswith('_r'):
            cm = vv.colormaps[cmap[:-2]]
            cm = cm[::-1]
        else:
            cm = vv.colormaps[cmap]
        surf.colormap = cm

        if clim is not None:
            # apply colormap limits
            surf.clim = clim
#            ax.Draw()

        # add labels to indicate pitch and/or roll axes
        aa = np.deg2rad(np.linspace(45, 315, 100) - 90)
        r = .25 + .05
        d = 1.25 + .25
        color = (.25, .25, .25)

        if pitch_arrow:
            # pitch rotation (in x/z plane, i.e. around y-axis)
            xx = r * np.cos(aa)
            zz = r * np.sin(aa)
            yy = d * np.ones_like(xx)
            vv.plot(xx, yy, zz, lw=5, lc=color, ls="-",
                    axesAdjust=False, axes=ax)

            translation = (xx[0], yy[0], zz[0])
            direction = (xx[0] - xx[1],
                         yy[0] - yy[1],
                         zz[0] - zz[1])
            cone = vv.solidCone(translation=translation,
                                scaling=(.05, .05, .1),
                                direction=direction,
                                axesAdjust=False, axes=ax)
            cone.faceColor = color

        if pitch_label:
            vv.Text(ax, 'Pitch', x=0, y=.9*d, z=.5,
                    fontSize=28, color=color)

        if roll_arrow:
            # roll rotation (in y/z plane, i.e. around x-axis)
            yy = r * np.cos(aa[::-1])
            zz = r * np.sin(aa[::-1])
            xx = d * np.ones_like(xx)
            vv.plot(xx, yy, zz, lw=5, lc=color, ls="-",
                    axesAdjust=False, axes=ax)

            translation = (xx[0], yy[0], zz[0])
            direction = (xx[0] - xx[1],
                         yy[0] - yy[1],
                         zz[0] - zz[1])
            cone = vv.solidCone(translation=translation,
                                scaling=(.05, .05, .1),
                                direction=direction,
                                axesAdjust=False, axes=ax)
            cone.faceColor = color

        if roll_label:
            vv.Text(ax, 'Roll', x=1.25*d, y=-.8, z=0,
                    fontSize=28, color=color)

        if avg_direction is not None:
            # indicate direction of avg head orientation
            avg_direction = avg_direction / np.sqrt(np.sum(avg_direction**2))
            avg_direction *= length
            cyl = vv.solidCylinder(translation=(0, 0, 0),
                                   scaling=(.05, .05, length),
                                   direction=tuple(avg_direction),
                                   axesAdjust=False, axes=ax)
            cyl.faceColor = arrow_color

            cone = vv.solidCone(translation=tuple(avg_direction),
                                scaling=(.1, .1, .2),
                                direction=tuple(avg_direction),
                                axesAdjust=False, axes=ax)
            cone.faceColor = arrow_color

        zoom_ = vv.view()['zoom']
        if isometric:
            vv.view(dict(azimuth=90+ISO_AZIMUTH, elevation=ISO_ELEVATION),
                    zoom=zoom_*zoom, axes=ax)
        else:
            vv.view(dict(azimuth=90+azimuth, elevation=elevation),
                    zoom=zoom*zoom_, axes=ax)

        ax.light0.ambient = light_ambient
        ax.light0.specular = light_specular

        fig.DrawNow()
        app.ProcessEvents()

        img = vv.screenshot(None, ob=ax, sf=2, bg=None, format=None)
        ax = img

        if close_figure:
            vv.close(fig)
            fig = None

    return fig, ax


def cylinder(r, n=20):

    m = len(r)
    dum = np.arange(0, n+1, dtype=float)
    theta = dum/n*2*np.pi
    sintheta = np.sin(theta)
    sintheta[n] = 0
    sintheta = np.transpose(sintheta)

    x = np.outer(r, np.cos(theta))  # dot product 1D vectors
    y = np.outer(r, sintheta)
    dum = np.ones([1, n+1], dtype=float)
    z = np.outer(np.arange(0, m, dtype=float)/(m-1), dum)

    return x, y, z


def sphere(r, n=20):

    theta = np.arange(-n, n+1, 2, dtype=float) / n * np.pi
    phi = np.arange(-n, n+1, 2, dtype=float) / n * np.pi/2
    cosphi = np.cos(phi)
    cosphi[0] = 0
    cosphi[n] = 0
    sintheta = np.sin(theta)
    sintheta[0] = 0
    sintheta[n] = 0

    x = np.outer(cosphi, np.cos(theta))
    y = np.outer(cosphi, sintheta)
    z = np.outer(np.sin(phi), np.ones([1, n+1]))

    return x, y, z


def Arrow3D(x1, y1, z1, x2, y2, z2, color=(.5, .5, .5),
            length_cone=.1, radius_cone=.02, radius_shaft=.01):

    from tvtk.tools import visual
    arrow = visual.arrow(x=x1, y=y1, z=z1, color=color)
    arrow.length_cone = length_cone
    arrow.radius_cone = radius_cone
    arrow.radius_shaft = radius_shaft

    arrow_length = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    arrow.actor.scale = [arrow_length, arrow_length, arrow_length]
    arrow.pos = arrow.pos / arrow_length
    arrow.axis = [x2-x1, y2-y1, z2-z1]

    return arrow


# -----------------------------------------------------------------------------
# cartoon mouse head
# -----------------------------------------------------------------------------

def plot_mouse_head(show_now=False, size=(500, 500), ax=None,
                    theta=0, phi=0, pitch=None, roll=None,
                    isometric=True,
                    azimuth=None, elevation=None, zoom=1.5,
                    gravity_axes=True, head_axes=True,
                    close_figure=False,
                    color_mouse=(.5, .5, .5),
                    color_head_axes=(.25, .95, .8),
                    color_gravity_axes=(.75, .75, .75),
                    backend='visvis',
                    light_ambient=.6,
                    light_specular=.5,
                    arrow_kw=dict(length_cone=.1,
                                  radius_cone=.05,
                                  radius_shaft=.025)):

    if azimuth is None:
        azimuth = DEFAULT_AZIMUTH
    if elevation is None:
        elevation = DEFAULT_ELEVATION

    t = np.arange(0, 1*np.pi-2*np.pi/10, np.pi/10)
    r = 2 + np.cos(t)
    n = 20

    if backend in ['mayavi', 'mlab']:

        from mayavi import mlab
        from tvtk.tools import visual

        fig = mlab.figure(bgcolor=(1, 1, 1), size=size)
        fig.scene.parallel_projection = True

        # head
        c = -3
        [X, Y, Z] = cylinder(r, n=n)
        head = mlab.mesh(X, Y, c+Z*6, color=(.5, .5, .5))

        # nose
        [x, y, z] = sphere(20)
        nose = mlab.mesh(x*1.5, y*1.5, c+z*1.5+6, color=(.5, .5, .5))

        # EARS
        color = (.5, .5, .5)
        hsE1 = mlab.mesh((x*1.0)-2.427, (y*1.8)-1.763, c+(z*0.4)+0,
                         color=color)
        hsE2 = mlab.mesh((x*1.0)+2.427, (y*1.8)-1.763, c+(z*0.4)+0,
                         color=color)

        # EYES
        [x, y, z] = sphere(10)
        color = (.9, .9, .9)
        hsEYE1 = mlab.mesh((x*.8)-1.2, (y*.8)-1.6, c+(z*.8)+3.5, color=color)
        hsEYE2 = mlab.mesh((x*.8)+1.2, (y*.8)-1.6, c+(z*.8)+3.5, color=color)

        # pupils
        hsPUP1 = mlab.mesh((x*.3)-1.476, (y*.3)-1.98, c+(z*.3)+4.147,
                           color=(0.1, 0.1, 0.1))
        hsPUP2 = mlab.mesh((x*.3)+1.476, (y*.3)-1.98, c+(z*.3)+4.147,
                           color=(0.1, 0.1, 0.1))

        # whiskers
        Px = np.array([-1.154, -1.214, -1.154, +1.154, +1.214, +1.154])
        Py = np.array([-0.375, 0, 0.375, -0.375, 0, 0.375])
        Pz = np.ones(np.size(Px))*3.882
        WL = np.array([[0.5, 2], [0.05, 0.4]])  # whisker length

        hsWHISK = []
        for W in range(0, len(Px)):  # W=0

            if Px[W] < 0:
                XSIGN = -1
            else:
                XSIGN = +1

            if Py[W] < 0:
                YSIGN = -1
            elif Py[W] == 0:
                YSIGN = 0
            else:
                YSIGN = +1

            x = np.append(Px[W], Px[W]+XSIGN*WL[0, :])
            y = np.append(Py[W], Py[W]+YSIGN*WL[1, :])
            z = np.append(Pz[W], Pz[W])

            tck, u = interpolate.splprep([x, y], k=2)
            xi, yi = interpolate.splev(np.linspace(0, 1, 100),
                                       tck, der=0)

            zi = np.ones(np.size(yi))*Pz[W]
            hsWHISK.append(mlab.plot3d(xi, yi, zi, color=(0, 0, 0),
                                       line_width=2))

        # rotate all objects
        angle_x = -90
        angle_y = 0
        angle_z = -90

        for actor in [head, nose, hsE1, hsE2, hsEYE1, hsEYE2, hsPUP1, hsPUP2]:
            actor.actor.actor.rotate_z(angle_z)
            actor.actor.actor.rotate_x(angle_x)
            actor.actor.actor.rotate_y(angle_y)

    #        actor.actor.actor.rotate_y(phi)
    #        actor.actor.actor.rotate_z(theta)

            actor.actor.actor.rotate_x(theta)
            actor.actor.actor.rotate_y(phi)

        for i in range(0, len(hsWHISK)):
            hsWHISK[i].actor.actor.rotate_z(angle_z)
            hsWHISK[i].actor.actor.rotate_x(angle_x)
            hsWHISK[i].actor.actor.rotate_y(angle_y)

            hsWHISK[i].actor.actor.rotate_x(theta)
            hsWHISK[i].actor.actor.rotate_y(phi)

        if head_axes:
            # axes of head-centered coordinate system
            visual.set_viewer(fig)

            arr1 = Arrow3D(0, 0, 0, 0, -6, 0, color=color_head_axes,
                           **arrow_kw)

            for arr in [arr1]:  # , arr2, arr3]:

                arr.actor.rotate_z(angle_z)
                arr.actor.rotate_x(angle_x)
                arr.actor.rotate_y(angle_y)

                arr.actor.rotate_x(theta)
                arr.actor.rotate_y(phi)

        if gravity_axes:

            # gravitational coordinate system
            visual.set_viewer(fig)

            arr1 = Arrow3D(0, 0, 0, -6, 0, 0, color=(.8, .8, .8), **arrow_kw)
            arr2 = Arrow3D(0, 0, 0, 0, -6, 0, color=(.5, .5, .5), **arrow_kw)
            arr3 = Arrow3D(0, 0, 0, 0, 0, 6, color=(.25, .25, .25), **arrow_kw)

            for arr in [arr1, arr2, arr3]:

                arr.actor.rotate_z(angle_z)
                arr.actor.rotate_x(angle_x)
                arr.actor.rotate_y(angle_y)

        if isometric:
            fig.scene.isometric_view()
        else:
            mlab.view(azimuth=azimuth, elevation=elevation, figure=fig)
        fig.scene.camera.zoom(zoom)

        if show_now:
            mlab.show()

        img = mlab.screenshot(figure=fig, mode='rgb', antialiased=False)

    elif backend in ['visvis']:

        import visvis as vv

        if ax is None:
            app = vv.use()
            fig = vv.figure()
            ax = vv.subplot(111)

        else:
            app = None
            fig = ax.GetFigure()

        ax.bgcolor = (1, 1, 1)

        ax.axis.visible = 0
        ax.axis.xLabel = 'x'
        ax.axis.yLabel = 'y'
        ax.axis.zLabel = 'z'

        # head
        c = -3
        [X, Y, Z] = cylinder(r, n=n)
        head = vv.surf(c+6*Z, X, Y)
        head.faceColor = color_mouse

        # nose
        [x, y, z] = sphere(20)
        nose = vv.surf(c+z*1.5+6, x*1.5, y*1.5)
        nose.faceColor = color_mouse

        # ears
        color = (.5, .5, .5)
        ear1 = vv.surf(c+(z*0.4)+0, (x*1.0)-2.427, -1*((y*1.8)-1.763))
        ear1.faceColor = color
        ear2 = vv.surf(c+(z*0.4)+0, (x*1.0)+2.427, -1*((y*1.8)-1.763))
        ear2.faceColor = color_mouse

        # eyes
        [x, y, z] = sphere(10)
        color = (.9, .9, .9)
        eye1 = vv.surf(c+(z*.8)+3.5, (x*.8)-1.2, -1*((y*.8)-1.6))
        eye2 = vv.surf(c+(z*.8)+3.5, (x*.8)+1.2, -1*((y*.8)-1.6))
        [setattr(eye, 'faceColor', color) for eye in [eye1, eye2]]

        # pupils
        color = (.1, .1, .1)
        pupil1 = vv.surf(c+(z*.3)+4.147-.5, (x*.3)-1.476-.2, -1*((y*.3)-1.98))
        pupil2 = vv.surf(c+(z*.3)+4.147-.5, (x*.3)+1.476+.2, -1*((y*.3)-1.98))
        [setattr(pupil, 'faceColor', color) for pupil in [pupil1, pupil2]]

        # whiskers
        Px = np.array([-1.154, -1.214, -1.154, +1.154, +1.214, +1.154])
        Py = np.array([-0.375, 0, 0.375, -0.375, 0, 0.375])
        Pz = np.ones(np.size(Px))*3.882
        WL = np.array([[0.5, 2], [0.05, 0.4]])  # whisker length

        whiskers = []
        for W in range(0, len(Px)):  # W=0

            if Px[W] < 0:
                XSIGN = -1
            else:
                XSIGN = +1

            if Py[W] < 0:
                YSIGN = -1
            elif Py[W] == 0:
                YSIGN = 0
            else:
                YSIGN = +1

            x = np.append(Px[W], Px[W]+XSIGN*WL[0, :])
            y = np.append(Py[W], Py[W]+YSIGN*WL[1, :])
            z = np.append(Pz[W], Pz[W])

            tck, u = interpolate.splprep([x, y], k=2)
            xi, yi = interpolate.splev(np.linspace(0, 1, 100),
                                       tck, der=0)

            zi = np.ones(np.size(yi))*Pz[W]
            whisker = vv.plot(zi, xi, -1*yi, lw=2, lc=(0, 0, 0))
            whiskers.append(whisker)

        if head_axes:
            # show vector indicating orientation of head
            length = 1
            shrink = .825

            if pitch is not None and roll is not None:
                # convert pitch/roll to spherical coordinates by rotating
                # a reference point around cartesian axes; note that x and
                # y are swapped in visvis
                p0 = np.array([0, 0, 1])

                a = np.deg2rad(roll)
                Rx = np.array([[1, 0, 0],
                               [0, np.cos(a), -np.sin(a)],
                               [0, np.sin(a), np.cos(a)]])

                b = np.deg2rad(pitch)
                Ry = np.array([[np.cos(b), 0, np.sin(b)],
                               [0, 1, 0],
                               [-np.sin(b), 0, np.cos(b)]])

                direction = np.dot(Ry, np.dot(Rx, p0)) * length
            else:
                direction = sph2cart(length, theta, phi,
                                     degrees=True).ravel()
            cyl = vv.solidCylinder(translation=(0, 0, .25),
                                   scaling=(.05, .05, shrink*length),
                                   direction=tuple(direction),
                                   axesAdjust=False, axes=ax)
            cyl.faceColor = (.25, .95, .8)
            cyl.do_not_rotate = True
            cyl.do_not_scale = True

            translation = direction*shrink + np.array([0, 0, .25])
            cone = vv.solidCone(translation=tuple(translation),
                                scaling=(.1, .1, .2),
                                direction=tuple(direction),
                                axesAdjust=False, axes=ax)
            cone.faceColor = (.25, .95, .8)
            cone.do_not_rotate = True
            cone.do_not_scale = True

        if gravity_axes:
            # show vector indicating (negative) orientation of gravity
            length = 1
            shrink = .825
            color = (.75, .75, .75)

            direction = np.array([0, 0, 1])
            cyl = vv.solidCylinder(translation=(0, 0, .25),
                                   scaling=(.05, .05, shrink*length),
                                   direction=tuple(direction),
                                   axesAdjust=False, axes=ax)
            cyl.faceColor = color
            cyl.do_not_rotate = True
            cyl.do_not_scale = True

            translation = direction*shrink + np.array([0, 0, .25])
            cone = vv.solidCone(translation=tuple(translation),
                                scaling=(.1, .1, .2),
                                direction=tuple(direction),
                                axesAdjust=False, axes=ax)
            cone.faceColor = color
            cone.do_not_rotate = True
            cone.do_not_scale = True

        # compute pitch and/or roll if not given
        if pitch is None or roll is None:

            # we have to find rotations (=angles) between the unit vector
            # in spherical coordinates (1, theta, phi) and the planes
            # for pitch (y/z) and roll (x/z); note that xyz is already
            # normaizeed (norm = 1)
            xyz = sph2cart(1, theta, phi, degrees=True).ravel()

            if pitch is None:
                # pitch (y/z plane with normal vector (1, 0, 0))
                pitch = 90 - np.degrees(np.arccos(np.dot(xyz, [1, 0, 0])))

            if roll is None:
                # roll (x/z plane with normal vector (0, -1, 0))
                roll = 90 - np.degrees(np.arccos(np.dot(xyz, [0, -1, 0])))

        for obj in ax.wobjects:

            if not hasattr(obj, 'do_not_scale'):
                rot = vv.Transform_Scale(sx=.25, sy=.25, sz=.25)
                obj.transformations.append(rot)

            if obj.__class__ != vv.core.axises.CartesianAxis and\
                    not hasattr(obj, 'do_not_rotate'):

                # note that x and y are swapped
                rot = vv.Transform_Rotate(pitch, ax=0, ay=1, az=0)
                obj.transformations.append(rot)

                rot = vv.Transform_Rotate(roll, ax=1, ay=0, az=0)
                obj.transformations.append(rot)

        zoom = vv.view()['zoom']
        if isometric:
            vv.view(dict(azimuth=90+ISO_AZIMUTH, elevation=ISO_ELEVATION),
                    zoom=4*zoom, axes=ax)
        else:
            vv.view(dict(azimuth=90+azimuth, elevation=elevation),
                    zoom=4*zoom, axes=ax)

        ax.light0.ambient = light_ambient
        ax.light0.specular = light_specular

        fig.DrawNow()

        if app is not None:
            app.ProcessEvents()

        img = vv.screenshot(None, ob=ax, sf=2, bg=None, format=None)

        if close_figure:
            vv.close(fig)
            fig = None

    return fig, img


def plot_reference_sphere(theta=0, phi=0, isometric=True, ax=None,
                          azimuth=None, elevation=None, degrees=True,
                          labels=True,
                          light_ambient=.6, zoom=1.4,
                          arrow_color=(.25, .95, .8),
                          close_figure=False):

    if azimuth is None:
        azimuth = DEFAULT_AZIMUTH
    if elevation is None:
        elevation = DEFAULT_ELEVATION

    import visvis as vv

    if ax is None:
        app = vv.use()
        fig = vv.figure()
        ax = vv.subplot(111)

    else:
        app = None
        fig = ax.GetFigure()

    ax.axis.visible = 0
    ax.axis.xLabel = 'x'
    ax.axis.yLabel = 'y'
    ax.axis.zLabel = 'z'

    # coordinate system
    length = 1.4
    cyl_diameter = .05
    for i in range(3):
        direction = np.zeros((3,))
        direction[i] = 1
        cyl = vv.solidCylinder(translation=(0, 0, 0),
                               scaling=(cyl_diameter, cyl_diameter, length),
                               direction=tuple(direction),
                               axesAdjust=False, axes=ax)
        cyl.faceColor = (.75, .75, .75)

        translation = np.zeros((3,))
        translation[i] = length
        cone = vv.solidCone(translation=tuple(translation),
                            scaling=(.1, .1, .2),
                            direction=tuple(direction),
                            axesAdjust=False, axes=ax)
        cone.faceColor = (.75, .75, .75)

    # example direction
    length = 1
    shrink = .825
    direction = sph2cart(1, theta, phi, degrees=degrees).ravel()
    cyl = vv.solidCylinder(translation=(0, 0, 0),
                           scaling=(.05, .05, shrink*length),
                           direction=tuple(direction),
                           axesAdjust=False, axes=ax)
    cyl.faceColor = arrow_color

    translation = direction*shrink
    cone = vv.solidCone(translation=tuple(translation),
                        scaling=(.1, .1, .2),
                        direction=tuple(direction),
                        axesAdjust=False, axes=ax)
    cone.faceColor = arrow_color

    # indicate unit sphere
    sphere = vv.solidSphere((0, 0, 0), N=100, M=100)
    sphere.faceColor = (.5, .5, .5, .25)

    # some lines on sphere indicating main axes
    phi = np.linspace(0, 2*np.pi, 100)
    theta = np.ones_like(phi) * np.pi/2.
    r = np.ones_like(phi)
    xyz = sph2cart(r, theta, phi, degrees=False)
    vv.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], lw=1, lc=(.5, .5, .5), ls="-",
            mc='b', axesAdjust=False, axes=ax)

    theta = np.linspace(-np.pi, np.pi, 100)
    phi = np.ones_like(theta) * 0
    r = np.ones_like(phi)
    xyz = sph2cart(r, theta, phi, degrees=False)
    vv.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], lw=1, lc=(.5, .5, .5), ls="-",
            mc='b', axesAdjust=False, axes=ax)

    theta = np.linspace(-np.pi, np.pi, 100)
    phi = np.ones_like(theta) * np.pi/2.
    r = np.ones_like(phi)
    xyz = sph2cart(r, theta, phi, degrees=False)
    vv.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], lw=1, lc=(.5, .5, .5), ls="-",
            mc='b', axesAdjust=False, axes=ax)

    # add pitch and roll axes
    aa = np.deg2rad(np.linspace(45, 315, 100) - 90)
    r = .25 + .025
    d = 1.25 + .05
    color = (.25, .25, .25)

    # pitch rotation (in x/z plane, i.e. around y-axis)
    xx = r * np.cos(aa)
    zz = r * np.sin(aa)
    yy = d * np.ones_like(xx)
    vv.plot(xx, yy, zz, lw=5, lc=color, ls="-",
            axesAdjust=False, axes=ax)

    translation = (xx[0], yy[0], zz[0])
    direction = (xx[0] - xx[1],
                 yy[0] - yy[1],
                 zz[0] - zz[1])
    cone = vv.solidCone(translation=translation,
                        scaling=(.05, .05, .1),
                        direction=direction,
                        axesAdjust=False, axes=ax)
    cone.faceColor = color

    if labels:
        vv.Text(ax, 'Pitch', x=0, y=1.25*d, z=.25, fontSize=28, color=color)

    # roll rotation (in y/z plane, i.e. around x-axis)
    yy = r * np.cos(aa)
    zz = r * np.sin(aa)
    xx = d * np.ones_like(xx)
    vv.plot(xx, yy, zz, lw=5, lc=color, ls="-",
            axesAdjust=False, axes=ax)

    translation = (xx[-1], yy[-1], zz[-1])
    direction = (xx[-1] - xx[-2],
                 yy[-1] - yy[-2],
                 zz[-1] - zz[-2])
    cone = vv.solidCone(translation=translation,
                        scaling=(.05, .05, .1),
                        direction=direction,
                        axesAdjust=False, axes=ax)
    cone.faceColor = color

    if labels:
        vv.Text(ax, 'Roll', x=1.25*d, y=-.8, z=0, fontSize=28, color=color)

    # set camera view
    zoom_ = vv.view()['zoom']
    if isometric:
        vv.view(dict(azimuth=90+ISO_AZIMUTH, elevation=ISO_ELEVATION),
                zoom=zoom*zoom_, axes=ax)
    else:
        vv.view(dict(azimuth=90+azimuth, elevation=elevation),
                zoom=zoom*zoom_, _axes=ax)

    ax.light0.ambient = light_ambient
    ax.light0.specular = .5

    fig.DrawNow()

    if app is not None:
        app.ProcessEvents()

    img = vv.screenshot(None, ob=ax, sf=2, bg=None, format=None)

    if close_figure:
        vv.close(fig)
        fig = None

    return fig, img
