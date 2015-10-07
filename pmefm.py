# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function
import numpy as np
from scipy import signal
from scipy.special import j0, j1, jn, jn_zeros
from scipy import optimize
from matplotlib import gridspec
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import sigutils
from scipy.signal.signaltools import _centered


def _fit_phase(t, phase, amp, phase_reversals=True):
    if phase_reversals:
        dphi_max = np.pi/2
    else:
        dphi_max = np.pi
    f = lambda x: np.sum(amp**2*abs((abs(abs(phase - (x[0]*t + x[1])) - dphi_max) - dphi_max))**2)
    return f


def auto_phase(t, z, x0=np.array([0., 0.]), phase_reversals=True):
    """"""
    phase = np.angle(z)
    amp = abs(z) / np.std(z)
    return optimize.fmin_slsqp(_fit_phase(t, phase, amp, phase_reversals), x0,)


def _j1filt(x):
    return np.where(x == 0, np.ones_like(x), np.where(x < jn_zeros(0, 1)[0],
                    j0(x) / (2 * j1(x)/x), np.zeros_like(x)))

def _j0filt(x):
    return np.where(x < jn_zeros(0, 1)[0], 1, 0)


def _matched_filters(ks, x_m, N_pts, dec=16, window='hann', n_pts_eval_fir=48000):
    ks = ks / dec
    N = N_pts // dec
    k = np.linspace(0, ks/2, n_pts_eval_fir)

    resp_ac = _j1filt(k * 2 * np.pi * x_m)

    fir_ac_dec = signal.firwin2(N, k, resp_ac, nyq=k[-1], window=window)
    fir_dc_dec = signal.firwin(N, jn_zeros(0, 1)[0] / (2*np.pi*x_m),
                               nyq=k[-1], window=window)

    # Manually force gain to 1 at DC; firwin2 rounding errors probable cause of
    # minor losses (< 1 percent)
    fir_ac_dec = fir_ac_dec / np.sum(fir_ac_dec)
    fir_dc_dec = fir_dc_dec / np.sum(fir_dc_dec)

    fir_ac = np.fft.irfft(np.fft.rfft(fir_ac_dec), fir_ac_dec.size * dec)
    fir_dc = np.fft.irfft(np.fft.rfft(fir_dc_dec), fir_dc_dec.size * dec)

    return fir_ac, fir_dc


def phase_err(x):
    return np.pi/2*(signal.sawtooth((x-np.pi/2)*2, width=1))


def ex2h5(filename, p):

    with h5py.File(filename, 'w') as f:
        f['LockinE'] = p.E_mod
        f['CPD'] = p.phi
        f['Czz'] = 0
        f.attrs['Inputs.Scan rate [Hz]'] = p.fs
        f.attrs['Inputs.Start scan [V]'] = 0
        f.attrs['Inputs.End scan [V]'] = (p.dx*p.t.size) * 10
        f.attrs['Inputs.Pos Mod Freq (Hz)'] = p.fx
        f.attrs['Inputs.Pos Mod rms (V)'] = p.x_m / (
            np.sqrt(2) * 0.95 * 15 * 100)
        f.attrs['Inputs.Gate voltage'] = 0
        f.attrs['Inputs.Drain voltage'] = 0


def _h5toPMEFM(filename):
    with h5py.File(filename, 'r') as f:
        phi_t = f['CPD'][:]
        fs = f.attrs['Inputs.Scan rate [Hz]']
        dt = 1/fs
        T = phi_t.size * dt
        fx = f.attrs['Inputs.Pos Mod Freq (Hz)']
        x_m = f.attrs['Inputs.Pos Mod rms (V)'] * np.sqrt(2) * 0.95 * 15 * 0.1
        x_tot = (f.attrs['Inputs.End scan [V]'] -
                 f.attrs['Inputs.Start scan [V]']) * 0.1
        v_tip = x_tot / T

    return {'fs': fs, 'fx': fx, 'v_tip': v_tip, 'x_m': x_m, 'phi_t': phi_t}


def fft(x, t=1, real=True, window='rect'):
    if isinstance(t, int) or isinstance(t, float):
        dt = t
    else:
        dt = t[1] - t[0]

    if real:
        fft_func = np.fft.rfft
        fft_freq_func = np.fft.rfftfreq
    else:
        fft_func = np.fft.fft
        fft_freq_func = np.fft.fftfreq

    window_factor = signal.get_window(window, x.size)

    ft = fft_func(x * window_factor)
    ft_freq = fft_freq_func(x.size, d=dt)

    return ft, ft_freq


def freqdemod_filter_used(f, fc, bw):
    f_norm = np.abs((f - fc) / bw)
    return np.where(f_norm <= 1, np.cos(f_norm*np.pi/2), 1e-16)


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

    labels = {'x_dc':   u"Position [µm]",
              'x_ac':   u"Position [µm]",
              'x':     u"Position [µm]",
              't':      u"Time [s]",
              'V_dc':  u"Potential [V]",
              'E_mod': u"Electric Field [V/µm]",
              'E_dc':  u"Electric Field [V/µm]",
              'phi_t': u"Potential [V]",}

    rcParams = {'axes.labelsize': 14, 'figure.figsize': (6,4.5)}

    @staticmethod
    def load_hdf5(filename):
        return PMEFMEx(**_h5toPMEFM(filename))

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
        self.kx = self.fx / self.v_tip
        self.dx = self.v_tip / self.fs
        self.ks = 1 / self.dx
        self.x_m = x_m

        self.k0_dc = jn_zeros(0, 1)[0] / (2 * np.pi * x_m)
        self.k0_ac = jn_zeros(1, 1)[0] / (2 * np.pi * x_m)
        self.f0_dc = self.k0_dc * self.v_tip
        self.f0_ac = self.k0_ac * self.v_tip

        self.x_dc = self.v_tip * self.t
        self.x_ac = x_m * np.sin(2*np.pi*self.fx*self.t)
        self.x = self.x_dc + self.x_ac

        if phi_x is not None:
            self.phi_x = phi_x
            self.phi = self.phi_x(self.x)
        elif E_x is not None:
            self.E_x = E_x
            self.E_x_dc = self.E_x(self.x_dc)
            self.phi_x = -1 * np.cumsum(self.E_x_dc) * self.dx
            self.phi = np.interp(self.x, self.x_dc, self.phi_x)
        elif phi_t is not None:
            self.phi = phi_t
        else:
            raise ValueError("Must specify phi_x or E_x")

    def fir_filter(self, fir_ac=None, fir_dc=None, f_ac=None, f_dc=None,
                   a_ac=10, a_dc=10, alpha=None, filter_name=None, **kwargs):
        """Apply filters to generate the lock-in and dc components of phi"""

        if filter_name == 'bessel_matched':
            N_pts = kwargs.get('N_pts', int(self.ks / self.k0_dc * 6))
            dec = kwargs.get('dec', 32)
            n_pts_eval_fir = kwargs.get('n_pts_eval_fir', 2**16)
            window = kwargs.get('window', 'hann')

            fir_ac, fir_dc = _matched_filters(self.ks, self.x_m, N_pts, dec, window,
                                              n_pts_eval_fir)

            self.fir_ac = fir_ac
            self.fir_dc = fir_dc
        else:
            if fir_ac is None:
                if f_ac is None and alpha is None:
                    f_ac = self.fx * 0.5
                elif alpha is not None:
                    f_ac = self.v_tip/self.x_m * alpha
                self.fir_ac = signal.firwin(self.fs / (f_ac) * a_ac,
                                            f_ac, nyq=0.5 * self.fs,
                                            window='blackman')
            else:
                self.fir_ac = fir_ac

            if fir_dc is None:
                if f_dc is None and alpha is None:
                    f_dc = self.fx * 0.5
                elif alpha is not None:
                    f_dc = self.v_tip/self.x_m * alpha
                self.fir_dc = signal.firwin(self.fs/(f_dc) * a_dc,
                                            f_dc, nyq=0.5*self.fs,
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

        self.V_lock = self.phi_lock

        self.phi_lock_a = np.abs(self.phi_lock)
        self.phi_lock_phase = np.angle(self.phi_lock)

        self.phi_dc = signal.fftconvolve(self.phi, self.fir_dc, mode='same')
        self.V_dc = self.phi_dc

    def plot_phase(self, rcParams={}):
        if not hasattr(self, 'phi_lock_phase'):
            raise ValueError("set_filters before plotting phase")

        rcParams_ = self.rcParams

        rcParams_.update(rcParams)

        with mpl.rc_context(rcParams_):
            fig, ax = plt.subplots()
            phase = np.angle(self.phi_lock)
            ax.plot(self.tm, phase[self.m])
            if hasattr(self, 'phase'):
                adjusted_phase = phase_err(self.phase - phase)
                ax.plot(self.tm, (phase + adjusted_phase)[self.m])
            if hasattr(self, 'm_fit'):
                ax.axvspan(np.min(self.t[self.m_fit]),
                           np.max(self.t[self.m_fit]),
                           alpha=0.5,
                           color='0.5')

            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Phase [rad.]')

        return fig, ax

    def fit_phase(self, t_min, t_max):
        DeprecationWarning("'fit_phase' is deprecated. Use 'linfit_phase' instead")
        if not hasattr(self, 'phi_lock'):
            raise ValueError("set_filters before fitting mask")

        self.m_fit = np.logical_and(self.t >= t_min, self.t < t_max)
        phi_lock_phase = np.unwrap(np.angle(self.phi_lock[self.m_fit]))

  
        self.mb = np.polyfit(self.t[self.m_fit],
                             phi_lock_phase, 1)
        self.phase = np.polyval(self.mb, self.t)

    linfit_phase = fit_phase

    def auto_phase(self, x0=np.array([0., 0.])):
        """"""
        m = self.m
        self.mb = optimize.fmin_slsqp(_fit_phase(self.t[m],
                                                 self.phi_lock_phase[m],
                                                 self.phi_lock_a[m]),
                                      x0,)


        self.phase = np.polyval(self.mb, self.t)


    def manual_phase(self, m, b):
        self.mb = (m, b)
        self.phase = np.polyval(self.mb, self.t)

    def iir_filt(self, iir_ac=None, iir_dc=None, alpha=1, n=1):
        """Final stage of output filtering, applying an infinite impulse
        response filter to the lockin and dc signals."""
        # Warp factor accounts for using iir filter with filtfilt
        # (running the filter twice)
        self.alpha = alpha
        self.n = n
        warp_factor = (np.sqrt(2) - 1)**(-1/(2*n))
        f_dig = self.v_tip / self.x_m / (self.fs/2)

        f_c = warp_factor * f_dig * alpha

        if iir_ac is None:

            self.iir_ac = signal.butter(n, f_c)

        if iir_dc is None:

            self.iir_dc = signal.butter(n, f_c)

        self.V_lock = signal.filtfilt(*self.iir_ac, x=self.phi_lock)

        self.V_dc = signal.filtfilt(*self.iir_dc, x=self.phi_dc)

    def output(self):

        self.V_ac = np.real(np.exp(-1j * self.phase) * self.V_lock)

        self.E_dc = -1 * np.diff(self.V_dc) / self.dx
        self.E_mod = self.V_ac / self.x_m

    def plot(self, x, y, scale='linear', comp='abs', xlim=None, ylim=None,
             xlabel=None, ylabel=None, figax=None, rcParams={}, **plot_kwargs):

        xp = getattr(self, x)[self.m]
        yp = getattr(self, y)[self.m]

        if figax is None:
            if xlabel is None:
                xlabel = self.labels.get(x, None)
            if ylabel is None:
                ylabel = self.labels.get(y, None)

        if np.any(np.iscomplex(yp)):
            if comp == 'abs':
                yp = np.abs(yp)
            elif comp == 'real':
                yp = np.real(yp)
            elif comp == 'imag':
                yp = np.imag(yp)
            elif comp == 'both':
                figax = plot_simple(xp, np.real(yp), scale=scale,
                                    xlim=xlim, ylim=ylim,
                                    xlabel=xlabel,
                                    ylabel=ylabel,
                                    figax=figax,
                                    rcParams=rcParams, **plot_kwargs)
                yp = np.imag(yp)

        figax = plot_simple(xp, yp, scale=scale, xlim=xlim, ylim=ylim,
                            xlabel=xlabel, ylabel=ylabel, figax=figax,
                            rcParams=rcParams, **plot_kwargs)

        return figax

    def plot_filters(self, filters=['fir_ac', 'fir_dc'], k_space=False,
                     xlim=None, xlog=False, mag_lim=None, phase_lim=None,
                     gain_point=-3, figax=None, rcParams=None):

        if xlim is None:
            if k_space:
                x_min = self.k0_dc / 20
                x_max = self.kx
            else:
                x_min = self.f0_dc / 20
                x_max = self.fx
            if xlog:
                xlim = np.array([x_min, x_max])
            else:
                xlim = np.array([0, x_max])
        else:
            xlim = np.atleast_1d(xlim)

        if k_space:
            fs = self.ks
            xlabel = u'Wavenumber [1/µm]'
        else:
            fs = self.fs
            xlabel = u'Frequency [Hz]'

        freq = []
        resp = []
        for filter_name in filters:
            if 'fir' in filter_name:
                f_, resp_ = sigutils.freqz(getattr(self, filter_name),
                                           fs=fs, xlim=xlim)
            elif 'iir' in filter_name:
                f_, resp_ = sigutils.freqz(*getattr(self, filter_name),
                                           fs=fs, xlim=xlim)
                # iir filter applied twice with filtfilt
                resp_ = resp_ * resp_.conj()

            freq.append(f_)
            resp.append(resp_)

        figax = sigutils.bodes(freq, resp, xlim=xlim, xlog=xlog,
                               mag_lim=mag_lim, phase_lim=phase_lim,
                               gain_point=gain_point, figax=figax,
                               rcParams=rcParams)

        figax[1][1].set_xlabel(xlabel)

    def plot_output(self, xlim=None, E_lim=None, V_lim=None, grid=False,
                    minor=False,
                    figax=None, rcParams={'figure.figsize': (6.5, 6.5),
                                          'axes.labelsize': 14}):
        x = self('x_dc')

        if figax is None:
            with mpl.rc_context(rcParams):
                fig = plt.figure()
                gs = gridspec.GridSpec(2, 1, height_ratios=(3, 1))
                gs.update(hspace=0.001)
                ax1 = plt.subplot(gs[0])
                ax2 = plt.subplot(gs[1], sharex=ax1)
                plt.setp(ax1.get_xticklabels(), visible=False)

        else:
            fig, (ax1, ax2) = figax

        ax1.plot(x, self('E_mod'), label='mod')
        ax1.plot(x, self('E_dc'), label='dc')
        ax2.plot(x, self('V_dc'))

        ax1.set_yticks(ax1.get_yticks()[1:])
        ax2.set_yticks(ax2.get_yticks()[:-1])

        if xlim is not None:
            ax2.set_xlim(*xlim)
        if E_lim is not None:
            ax1.set_ylim(*E_lim)
        if V_lim is not None:
            ax2.set_ylim(*V_lim)

        if minor:
            x_ticks = ax2.get_xticks()
            x_major_delta = np.diff(x_ticks)
            ax2.set_xticks(x_ticks[:-1] + x_major_delta/2,
                           minor=True)

        ax1.set_ylabel(u"Electric field [V/µm]")
        ax2.set_xlabel(u"Position [µm]")
        ax2.set_ylabel(u"Potential [V]")

        if grid:
            ax1.grid(which='both', color='0.8', linestyle='-')
            ax2.grid(which='both', color='0.8', linestyle='-')

        return fig, (ax1, ax2)
