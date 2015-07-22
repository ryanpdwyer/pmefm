# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function
import numpy as np
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py


def plot_simple(x, y, magic=None, scale='linear', xlim=None, ylim=None,
                xlabel=None, ylabel=None, figax=None,
                rcParams={'backend': 'Qt4Agg'}, **plot_kwargs):
    if figax is not None:
        fig, ax = figax
    else:
        with mpl.rc_context(rc=rcParams):
            fig = plt.figure()
            ax = fig.add_subplot(111)

    plotting_functions = {
        'linear': ax.plot,
        'semilogy': ax.semilogy,
        'semilogx': ax.semilogx,
        'loglog': ax.loglog}

    if magic is None:
        plotting_functions[scale](x, y, **plot_kwargs)
    else:
        plotting_functions[scale](x, y, magic, **plot_kwargs)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return fig, ax


class PMEFMEx(object):
    """
    General workup procedure:

    p = PMEFMEx(fs, fx, v_tip, x_m, T, E_x)
    p.set_filters()
    p.plot_phase()
    p.filt_out()
    """
    def __call__(self, *args):
        if len(args) == 1:
            return getattr(self, args[0])[self.m]
        else:
            return tuple(getattr(self, arg)[self.m] for arg in args)

    def __init__(self, fs, fx, v_tip, x_m, T=None,
                 phi_x=None, E_x=None, phi_t=None):
        u"""
        fs: sampling frequency [Hz]
        fx: x modulation frequency [Hz]
        v_tip: tip velocity [µm/s]
        x_m: x modulation amplitude [µm]
        T: total data collection time [s]
        x_total: total distance in x [µm]
        phi_x: function describing surface potential as a function of x
        E_x: function describing electric field as a function of x_m
        """
        self.fs = fs
        self.dt = 1./fs

        if phi_t is not None:
            self.t = np.arange(phi_t.size) * self.dt
            self.T = phi_t.size * self.dt
        else:
            self.T = T
            self.t = np.arange(0, self.T, self.dt)

        self.fx = fx
        self.v_tip = v_tip
        self.dx = self.v_tip / self.fs
        self.ks = 1 / self.dx
        self.x_m = x_m
        self.x_dc = self.v_tip * self.t
        self.x_ac = x_m * np.sin(2*np.pi*self.fx*self.t)
        self.x = self.x_dc + self.x_ac

        if phi_x is not None:
            self.phi_x = phi_x
            self.phi = self.phi_x(self.x)
        elif E_x is not None:
            self.E_x = E_x
            self.E_x_dc = self.E_x(self.x_dc)
            self.phi_x = np.cumsum(self.E_x_dc) * self.dx
            self.phi = np.interp(self.x, self.x_dc, self.phi_x)
        elif phi_t is not None:
            self.phi = phi_t
        else:
            raise ValueError("Must specify phi_x or E_x")

    def set_filters(self, fir_ac=None, fir_dc=None):
        """Apply filters to generate the lock-in and dc components of phi"""

        if fir_ac is None:
            self.fir_ac = signal.firwin(self.fs / (self.fx * 0.5) * 10,
                                        self.fx * 0.5, nyq=0.5 * self.fs,
                                        window='blackman')
        else:
            self.fir_ac = fir_ac

        if fir_dc is None:
            self.fir_dc = signal.firwin(self.fs/(self.fx*0.5) * 10,
                                        self.fx*0.5, nyq=0.5*self.fs,
                                        window='blackman')
        else:
            self.fir_dc = fir_dc

        indices = np.arange(self.phi.size)
        fir_ac_size = self.fir_ac.size
        fir_dc_size = self.fir_dc.size

        fir_max_size = max(fir_ac_size, fir_dc_size)

        self.m = indices[fir_max_size//2: -fir_max_size//2]
        self.tm = self.t[self.m]

        self._lock = np.exp(np.pi * 2j * self.fx * self.t)

        self.phi_lock = signal.fftconvolve(self.phi * self._lock * 2,
                                           self.fir_ac,
                                           mode='same')

        self.phi_lock_a = np.abs(self.phi_lock)
        self.phi_lock_phase = np.unwrap(np.angle(self.phi_lock))

        self.V_dc = signal.fftconvolve(self.phi, self.fir_dc, mode='same')

    def plot_phase(self):
        if not hasattr(self, 'phi_lock_phase'):
            raise ValueError("set_filters before plotting phase")

        plt.plot(self.tm, self.phi_lock_phase[self.m])
        if hasattr(self, 'm_fit'):
            plt.plot(self.tm, self.phase[self.m])
            plt.axvspan(np.min(self.t[self.m_fit]),
                        np.max(self.t[self.m_fit]),
                        alpha=0.5,
                        color='0.5')

    def fit_phase(self, t_min, t_max):
        if not hasattr(self, 'phi_lock_phase'):
            raise ValueError("set_filters before fitting mask")

        self.m_fit = np.logical_and(self.t >= t_min, self.t < t_max)
        self.mb = np.polyfit(self.t[self.m_fit],
                             self.phi_lock_phase[self.m_fit], 1)
        self.phase = np.polyval(self.mb, self.t)

    def filt_out(self, iir_ac=None, iir_dc=None, alpha=1, n=1):
        """Final stage of output filtering, applying an infinite impulse
        response filter to the lockin and dc signals."""
        # Warp factor accounts for using iir filter with filtfilt
        # (running the filter twice)
        warp_factor = (np.sqrt(2) - 1)**(-1/(2*n))
        f_dig = self.v_tip / self.x_m / (self.fs/2)

        f_c = warp_factor * f_dig * alpha

        if iir_ac is None:

            self.iir_ac = signal.butter(n, f_c)

        if iir_dc is None:

            self.iir_dc = signal.butter(n, f_c)

        self.phi_lock_lp = signal.filtfilt(*self.iir_ac, x=self.phi_lock)

        self.V_lock_lp = np.real(np.exp(-1j * self.phase) * self.phi_lock_lp)

        self.V_dc_lp = signal.filtfilt(*self.iir_dc, x=self.V_dc)

        self.E_dc = np.diff(self.V_dc_lp) / self.dx
        self.E_mod = self.V_lock_lp / self.x_m

    def plot(self, x, y, scale='linear', comp='abs', xlim=None, ylim=None,
             xlabel=None, ylabel=None, figax=None, rcParams={}, **plot_kwargs):

        xp = getattr(self, x)[self.m]
        yp = getattr(self, y)[self.m]

        if np.any(np.iscomplex(yp)):
            if comp == 'abs':
                yp = np.abs(yp)
            elif comp == 'real':
                yp = np.real(yp)
            elif comp == 'imag':
                yp = np.imag(yp)
            elif comp == 'both':
                figax = plot_simple(xp, np.real(yp), scale=scale, xlim=xlim, ylim=ylim,
                        xlabel=xlabel, ylabel=ylabel, figax=figax,
                        rcParams=rcParams, **plot_kwargs)
                yp = np.imag(yp)

        figax = plot_simple(xp, yp, scale=scale, xlim=xlim, ylim=ylim,
                        xlabel=xlabel, ylabel=ylabel, figax=figax,
                        rcParams=rcParams, **plot_kwargs)

        return figax
