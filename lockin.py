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
import click
import h5py
import sys
import pathlib
from scipy.optimize import curve_fit
from scipy.signal.signaltools import _centered

# Inputs: t, x
# Cantilever amplitude, phase, frequency
# FIR filter (check response at f_c)
# Infer cantilever frequency if necessary
# Modulate, run through filter
# Amp, phase, fit_phase (assume zero)
# Outputs:
# A, dphi, f0


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


def freq_from_fft(sig, fs):
    """Estimate frequency from peak of FFT

    """
    # Compute Fourier transform of windowed signal
    windowed = sig * signal.blackmanharris(len(sig))
    f = np.fft.rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    i = np.argmax(abs(f)) # Just use this for less-accurate, naive version
    true_i = parabolic(np.log(abs(f)), i)[0]

    # Convert to equivalent frequency
    return fs * true_i / len(windowed)


def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    f is a vector and x is an index for that vector.

    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.

    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.

    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)

    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)


def _print_magnitude_data(w, rep, fs):
    df = pd.DataFrame()
    df['f'] = w /(2*np.pi) * fs
    df['mag'] = abs(rep)
    df['dB'] = 20 * np.log10(df['mag'].values)
    df.sort(columns="f", inplace=True)
    print(df.to_string(index=False, float_format="{:.3f}".format))
    return df


# x data
# (guess f0)
# filter (b, a)
# phasing
# Don't actually need time data

def lock2(f0, fp, fc, fs, coeff_ratio=8, coeffs=None,
          window='blackman', print_response=True):
    """Create a gentle fir filter. Pass frequencies below fp, cutoff frequencies
    above fc, and gradually taper to 0 in between.
    """
    nyq = fs/2
    fp = fp / nyq
    fc = fc / nyq

    if coeffs is None:
        coeffs = int(round(coeff_ratio / fc, 0))

    # Force number of coefficients odd
    alpha = (1-fp*1.0/fc)
    n = int(round(1000. / alpha) // 2)

    N = n * 2 + 1
    f = np.linspace(0, fc, n+1)

    fm = np.zeros(n + 2)
    mm = np.zeros(n + 2)
    fm[:-1] = f
    # Append fm = nyquist frequency by hand; needed by firwin2
    fm[-1] = 1.
    m = signal.tukey(N, alpha=alpha)
    mm[:-1] = m[n:]

    b = signal.firwin2(coeffs, fm, mm,
                       nfreqs=2**(int(round(np.log2(coeffs)))+3)+1,
                       window=window)

    b = b / np.sum(b)

    w, rep = signal.freqz(b, worN=np.pi*np.array([0., fp/2, fp, fc, 2*fc,
                                                  0.5*f0/nyq, f0/nyq, 1.]))
    if print_response:
        print("Response:")
        _print_magnitude_data(w, rep, fs)

    return b


class LockIn(object):
    def __init__(self, t, x, fs):
        self.t = t
        self.x = x
        self.fs = fs

        self.f0_est = freq_from_fft(self.x, self.fs)

    def lock(self, f0=None, bw_ratio=0.5, coeff_ratio=9., bw=None, coeffs=None,
             window='blackman'):

        t = self.t
        fs = self.fs
        if f0 is None:
            self.f0 = f0 = self.f0_est

        if bw is None:
            bw = bw_ratio * f0 / (self.fs/2)
        else:
            bw = bw / (self.fs/2)

        if coeffs is None:
            coeffs = round(coeff_ratio / bw, 0)
            if coeffs > self.x.size:
                raise ValueError(
"""No valid output when 'coeffs' > t.size (coeffs: {}, t.size: {}).
Reduce coeffs by increasing bw, bw_ratio, or decreasing coeff_ratio,
or provide more data.""".format(coeffs, t.size))

        self.b = b = signal.firwin(coeffs, bw, window=window)

        w, rep = signal.freqz(b, worN=np.pi*np.array([0., bw/2, bw, f0/self.fs, f0/(self.fs/2.), 1.]))

        print("Response:")
        _print_magnitude_data(w, rep, fs)

        self.z = z = signal.fftconvolve(self.x * np.exp(-2j*np.pi*f0*t), 2*b,
                                       "same")

        n_fir = b.size
        indices = np.arange(t.size)
        # Valid region mask
        # This is borrowed explicitly from scipy.signal.sigtools.fftconvolve
        self.m = m = _centered(indices, t.size - n_fir + 1)

        self.A = abs(self.z)
        self.phi = np.angle(self.z)

    def lock2(self, f0=None, fp_ratio=0.1, fc_ratio=0.4, coeff_ratio=8,
              fp=None, fc=None, coeffs=None, window='blackman',
              print_response=True):
        t = self.t
        fs = self.fs

        if f0 is None:
            self.f0 = f0 = self.f0_est

        if fp is None:
            fp = fp_ratio * f0

        if fc is None:
            fc = fc_ratio * f0

        self.fir = b = lock2(f0, fp, fc, fs, coeff_ratio, coeffs, window,
                             print_response=print_response)

        if coeffs > self.x.size:
            raise ValueError(
    """No valid output when 'coeffs' > t.size (coeffs: {}, t.size: {}).
    Reduce coeffs by increasing bw, bw_ratio, or decreasing coeff_ratio,
    or provide more data.""".format(coeffs, t.size))

        self.z = z = signal.fftconvolve(self.x * np.exp(-2j*np.pi*f0*t), 2*b,
                                       "same")  # Use other valid criteria?

        n_fir = b.size
        indices = np.arange(t.size)
        # Valid region mask
        # This is borrowed explicitly from scipy.signal.sigtools.fftconvolve
        self.m = m = np.zeros_like(t, dtype=bool)
        self.m[_centered(indices, t.size - n_fir + 1)] = True

        self.A = abs(self.z)
        self.phi = np.angle(self.z)

    def lock_butter(self, N, f3dB, t_exclude=0, f0=None, print_response=True):
        t = self.t
        fs = self.fs
        nyq = fs / 2.
        f3dB = f3dB / nyq 

        self.iir = ba = signal.iirfilter(N, f3dB, btype='low')

        if f0 is None:
            self.f0 = f0 = self.f0_est


        self.z = z = signal.lfilter(self.iir[0], self.iir[1], self.z)
        # TODO: Fix accounting on final / initial point
        m = self.m
        self.m = self.m & (t >= (t[m][0] + t_exclude)) & (t < (t[m][-1] - t_exclude))

        self.A = abs(self.z)
        self.phi = np.angle(self.z)

        if print_response:
            w, rep = signal.freqz(self.iir[0], self.iir[1],
                        worN=np.pi*np.array([0., f3dB/2, f3dB,
                                             0.5*f0/nyq, f0/nyq, 1.]))
            print("Response:")
            _print_magnitude_data(w, rep, fs)



    def autophase(self, ti=None, tf=None, unwrap=False, x0=[0., 0.]):
        t = self.t
        m = self.m
        z = self.z

        if unwrap:
            phi = np.unwrap(self.phi)
        else:
            phi = self.phi

        if ti is None and tf is None:
            mask = m
        elif ti is not None and tf is None:
            mask = m & (t >= ti)
        elif ti is None and tf is not None:
            mask = m & (t < tf)
        else:
            mask = m & (t >= ti) & (t < tf)


        self.mb = mb = auto_phase(t[mask], phi[mask], x0)

        self.phi_fit = np.polyval(mb, t)

        self.dphi = np.unwrap(((self.phi - self.phi_fit + np.pi) % (2*np.pi))
                               - np.pi)

        self.df = np.gradient(self.dphi) * self.fs / (2*np.pi)

        self.f0corr = self.f0 + mb[0]

    def phase(self, ti=None, tf=None, weight=True):
        t = self.t
        m = self.m
        z = self.z

        if ti is None and tf is None:
            mask = m
        elif ti is not None and tf is None:
            mask = m & (t >= ti)
        elif ti is None and tf is not None:
            mask = m & (t < tf)
        else:
            mask = m & (t >= ti) & (t < tf)

        phi = np.unwrap(self.phi[mask])
        std = np.std(self.phi[mask])
        phi_norm = phi / std

        try:
            if weight:
                A = abs(z[mask]) / np.std(abs(z[mask]))
                self.mb = mb = np.polyfit(t[mask], phi_norm, 1, w=A) * std
            else:
                self.mb = mb = np.polyfit(t[mask], phi_norm, 1) * std
        except TypeError:
            print(t)
            print(ti)
            print(tf)
            raise

        self.phi_fit = np.polyval(mb, t)

        self.dphi = np.unwrap(((self.phi - self.phi_fit + np.pi) % (2*np.pi))
                              - np.pi)

        self.df = np.zeros(t.size)
        self.df = np.gradient(self.dphi) * self.fs / (2*np.pi)

        self.f0corr = self.f0 + mb[0] / (2*np.pi)

    def decimate(self, factor=None):
        if factor is None:
            factor = int(self.fs//self.f0)

        self.dec_t = self.t[self.m][::factor]
        self.dec_phi = self.dphi[self.m][::factor]
        self.dec_A = self.A[self.m][::factor]
        self.dec_df = self.df[self.m][::factor]
        self.dec_f0 = self.f0
        self.dec_fs = self.fs/factor
        self.dec_z = self.z[self.m][::factor]

    def phase_dec(self, ti=None, tf=None, weight=True):
        t = self.dec_t
        m = np.ones_like(self.dec_z, dtype=bool)
        z = self.dec_z

        if ti is None and tf is None:
            mask = m
        elif ti is not None and tf is None:
            mask = m & (t >= ti)
        elif ti is None and tf is not None:
            mask = m & (t < tf)
        else:
            mask = m & (t >= ti) & (t < tf)

        phi = np.unwrap(np.angle(z))
        std = np.std(phi[mask])
        phi_norm = phi / std
        try:
            if weight:
                A = abs(z[mask]) / np.std(abs(z[mask]))
                self.mb = mb = np.polyfit(t[mask], phi_norm[mask], 1, w=A) * std
            else:
                self.mb = mb = np.polyfit(t[mask], phi_norm[mask], 1) * std
        except TypeError:
            print(t)
            print(ti)
            print(tf)
            raise

        phi_fit = np.polyval(mb, t)

        dphi = np.unwrap(((phi - phi_fit + np.pi) % (2*np.pi)) - np.pi)

        df = np.gradient(dphi) * self.dec_fs / (2*np.pi)

        self.f0_dec_direct = self.f0 + mb[0] / (2*np.pi)


class FIRState(object):
    def __init__(self, fir, dec, t0=0., fs=1.):
        self.fir = fir
        self.nfir_mid = (len(fir) - 1)//2
        self.dec = dec
        self.t0 = t0
        self.fs = fs
        self.t0_dec = t0 + self.nfir_mid / self.fs
        self.data = np.array([])
        self.output = np.array([])

    def filt(self, data):
        n = self.fir.size
        x = np.r_[self.data, data]
        y = signal.fftconvolve(x, self.fir, mode="full")
        indices = np.arange(y.size)

        m = indices[n-1:-n+1]
        if len(m) == 0:
            self.data = x
        else:
            m_dec = m[::self.dec]
            self.output = np.r_[self.output, y[m_dec]]
            self.data = x[m_dec[-1] - (n-1) + self.dec:]


class FIRStateLock(object):
    def __init__(self, fir, dec, f0, phi0, t0=0, fs=1.):
        self.fir = fir
        self.nfir_mid = (len(fir) - 1)//2
        self.dec = dec
        self.f0 = f0
        self.w0 = f0/fs
        self.phi0 = self.phi_i = phi0 + 2*np.pi*self.w0
        self.t0 = t0
        self.fs = fs
        self.t0_dec = t0 + self.nfir_mid / self.fs
        self.z = np.array([], dtype=np.complex128)
        self.z_out = np.array([], dtype=np.complex128)

    def filt(self, data):
        n = self.fir.size
        phi = (-2*np.pi*self.w0*np.arange(1, data.size+1) + self.phi_i
               ) % (2*np.pi)
        self.phi_i = phi[-1]

        z = np.r_[self.z, data * np.exp(1j*phi)]
        y = signal.fftconvolve(z, 2*self.fir, mode="full")

        indices = np.arange(y.size)
        m = indices[n-1:-n+1]
        if len(m) == 0:
            self.z = z
        else:
            m_dec = m[::self.dec]
            self.z_out = np.r_[self.z_out, y[m_dec]]
            self.z = z[m_dec[-1] - (n-1) + self.dec:]

    def get_t(self):
        return self.t0_dec + np.arange(self.z_out.size)/self.fs * self.dec


class FIRStateLockVarF(object):
    def __init__(self, fir, dec, f0, phi0, t0=0, fs=1.):
        self.fir = fir
        self.nfir_mid = (len(fir) -1)//2
        self.dec = dec
        self.f0 = f0
        self.w0 = lambda t: f0(t) / fs
        self.phi0 = self.phi_i = phi0 + 2*np.pi*self.w0(t0)
        self.t0 = t0
        self.t_now = t0
        self.fs = fs
        self.t0_dec = t0 + self.nfir_mid / self.fs
        self.z = np.array([], dtype=np.complex128)
        self.z_out = np.array([], dtype=np.complex128)

    def filt(self, data):
        n = self.fir.size
        m = data.size
        t = self.t_now + np.arange(m, dtype=np.float64) / self.fs
        w = self.w0(t)
        phi = (-2*np.pi*np.cumsum(w) + self.phi_i) % (2*np.pi)
        self.phi_i = phi[-1]
        self.t_now = t[-1]

        z = np.r_[self.z, data * np.exp(1j*phi)]
        y = signal.fftconvolve(z, 2*self.fir, mode="full")

        indices = np.arange(y.size)
        m = indices[n-1:-n+1]
        if len(m) == 0:
            self.z = z
        else:
            m_dec = m[::self.dec]
            self.z_out = np.r_[self.z_out, y[m_dec]]
            self.z = z[m_dec[-1] - (n-1) + self.dec:]

    def get_t(self):
        return self.t0_dec + np.arange(self.z_out.size)/self.fs * self.dec


def adiabatic_phasekick(y, dt, tp, t0, T_before, T_after, T_bf, T_af,
                        fp, fc, fs_dec, T_before_offset=0., print_response=True):
    """Workup an individual adiabatic phasekick dataset, starting from the raw
    cantilever vs. time data.

    Parameters
    ----------

    y: array
        Cantilever displacement vs time.
    dt: float
        Spacing between time points.
    tp: float
        Pulse time.
    t0: float
        Initial time, using the convention that the pulse starts at t = 0.
    T_before: float
        Time to average phase before the pulse time.
    T_after: float
        Time to average phase after the pulse time.
    T_bf: float
        Time to average frequency before pulse time.
    T_af: float
        Time to average frequency after the pulse time.
    fp: float
        Lock-in filter setting: pass frequencies below ``fp``.
    fc: float
        Lock-in filter setting: cutoff frequencies above ``fc``.
    fs_dec: float
        Lock-in setting: Decimate to a sampling frequency ``fs_dec``.

    Returns
    -------
    FIRStateLockVarF
        An FIRStateLockVarF instance.

    """
    fs = 1. / dt
    t = np.arange(y.size) * dt + t0
    li = LockIn(t, y, fs)
    li.lock2(fp=fp, fc=fc, print_response=print_response)
    tedge = li.fir.size * dt / 2
    li.phase(ti=-T_bf+T_before_offset, tf=T_before_offset)
    f1 = li.f0corr
    phi0 = -li.phi[0]
    li.phase(ti=tp, tf=(tp+T_af))
    f2 = li.f0corr
    # Decimate by a conservative factor
    dec = int(np.floor(fs / fs_dec))
    
    def f_var(t):
        return np.where(t > tp, f2, f1)

    lockstate = FIRStateLockVarF(li.fir, dec, f_var, phi0, t0=t0, fs=fs)
    lockstate.filt(y)
    lockstate.dphi = np.unwrap(np.angle(lockstate.z_out))
    lockstate.df = np.gradient(lockstate.dphi) * (
            fs / (dec * 2*np.pi))
    
    lockstate.tp = tp
    lockstate.t = t = lockstate.get_t()
    lockstate.delta_phi = (np.mean(lockstate.dphi[(t >= tp) & (t < (tp + T_after))]) - 
                            np.mean(lockstate.dphi[(t >= -T_before) & (t < 0)])
                           )

    return lockstate


def workup_gr(ds, T_before, T_after, T_bf=0.001, T_af=0.002, fp=1000, fc=4000,
              fs_dec=16000, t0=0.05):
    """Lockin workup of the data in h5py dataset ds (Sarah / John h5 file).
    This assumes that ds contains attributes dt, pulse time, which enable us to
    perform the workup."""
    dt = ds.attrs['dt']
    tp = ds.attrs['pulse time']
    y = ds[:]
    return adiabatic_phasekick(y, dt, tp, -t0, T_before, T_after, T_bf, T_af,
                        fp, fc, fs_dec, T_before_offset=0., print_response=True)


def plot_phasekick_control(df):
    fig, ax = plt.subplots()
    ax.plot(df['tp [s]']*1e3, df['control dphi [cyc]'], 'bo')
    ax.plot(df['tp [s]']*1e3, df['data dphi [cyc]'], 'go')
    ax.set_xlabel('tp [ms]')
    ax.set_ylabel('phase shift [cyc.]')
    return fig, ax

def delta_phi_group(subgr, tp, T_before, T_after, T_bf=0.025, T_af=0.04,
                    fp=1000, fc=4000, fs_dec=16000, T_before_offset=0., print_response=True):
    y = subgr['cantilever-nm'][:]
    dt = subgr['dt [s]'].value
    t0 = subgr['t0 [s]'].value
    lockstate = adiabatic_phasekick(y, dt, tp, t0, T_before, T_after, T_bf, T_af,
                fp, fc, fs_dec, T_before_offset, print_response)
    return lockstate.delta_phi, lockstate

def workup_adiabatic_w_control(fh, T_before, T_after, T_bf=0.025, T_af=0.04,
                        fp=1000, fc=4000, fs_dec=16000):
    """Return a DataFrame containing phase shift vs. pulse time for
       experiment and control."""
    tps = fh['tp'][:] * 0.001 # ms to s
    tp_groups = fh['ds'][:]
    df = pd.DataFrame()
    df['tp [s]'] = tps
    i = 0
    for control_or_data in ('control', 'data'):
        delta_phi = []
        for (tp_group, tp) in zip(tp_groups, tps):
            print_response = i == 0
            dphi, _ = delta_phi_group(
                fh[control_or_data][tp_group], tp, T_before, T_after,
                T_bf, T_af, fp, fc, fs_dec, print_response=print_response)
            i += 1
            sys.stdout.write('.')
            delta_phi.append(dphi/(2*np.pi))
        df[control_or_data+' dphi [cyc]'] = delta_phi

    return df


def workup_file(gr, out_file, T_before, T_after,
                T_bf=0.002, T_af=0.002, fp=1000, fc=4000, fs_dec=16000, t0=0.05,
                overwrite=False, show_progress=True):
    out = []
    tp = []
    N = len(gr.items())
    m = int(N//10)
    i = 1
    for ds_name, ds in gr.items():
        out.append(workup_gr(ds, T_before, T_after, T_bf, T_af, fp, fc, fs_dec, t0))
        tp.append(ds.attrs['pulse time'])
        if show_progress and i % m == 0:
            print("{i}/{N} complete")
        i += 1

    return tp, out


def workup_adiabatic_realtime(fh, fp, fc, ti, tf, tiphase, p0=None,
                              show_progress=True):
    popts = []
    index = []
    lis = []
    for i, gr in enumerate([gr for gr in fh.values() if isinstance(gr, h5py.Group)]):
        index.append(int(gr.name.split('/')[-1]))
        li = adiabatic2lockin(gr)
        print_response = i == 0
        li.lock2(fp=fp, fc=fc, print_response=print_response)
        li.phase(ti=tiphase, tf=0.)
        lis.append(li)
        if p0 is None:
            popt, pcov = fitexpfall(li.t, li.df, ti, tf)
        else:
            popt, pcov = fitexpfall(li.t, li.df, ti, tf, p0=p0)
        popts.append(popt)
        sys.stdout.write('.')

    popts = np.array(popts)
    df = pd.DataFrame(data=popts*np.array([1., 1000., 1]), index=index,
                      columns=['df', 'tau', 'f0'])
    f0 = np.array([li.f0corr for li in lis])
    df['f0'] = df['f0'] + f0
    return df, lis


def expfall(x, df, tau, f0): 
    return np.where( x >= 0, df*(1-np.exp(-(x)/tau)), 0) + f0

def expfallt(x, df, tau, f0, t0): 
    return np.where( x >= t0, df*(1-np.exp(-(x-t0)/tau)), 0) + f0

def fitexpfall(t, f, ti, tf, p0=None, fit_t0=False):
    m = (t > ti) & (t <= tf)

    if p0 is None:
        popt, pcov = curve_fit(expfall, t[m], f[m])
    else:
        popt, pcov = curve_fit(expfall, t[m], f[m])

    if fit_t0:
        popt2 = list(popt)
        popt2.append(0)
        return curve_fit(expfallt, t[m], f[m], p0=popt2)
    else:
        return popt, pcov


def adiabatic2lockin(gr,):
    """Return a LockIn instance from an adiabatic phasekick formatted h5 file.
    
    Cantilever oscillator data is stored in 'cantilever-nm'."""
    x = gr['cantilever-nm'][:]
    dt = gr['dt [s]'].value
    N = x.size
    t0 = gr['t0 [s]'].value
    t = np.arange(N)*dt + t0
    return LockIn(t, x, 1./dt)

@click.command()
@click.argument('filename', type=click.Path())
@click.argument('fp', type=float)
@click.argument('fc', type=float)
@click.argument('ti', type=float)
@click.argument('tf', type=float)
@click.argument('tiphase', type=float)
def workup_adiabatic_avg(filename, fp, fc, ti, tf, tiphase):
    csv = filename.replace('.h5', '.csv')
    popts = []
    index = []
    with h5py.File(filename, 'r') as fh:
        for gr in [gr for gr in fh.values() if isinstance(gr, h5py.Group)]:
            index.append(int(gr.name.split('/')[-1]))
            li = adiabatic2lockin(gr)
            li.lock2(fp=fp, fc=fc)
            li.phase(ti=tiphase, tf=0.)
            popt, pcov = fitexpfall(li.t, li.df, ti, tf, 0)
            popts.append(popt)

    popts = np.array(popts)
    df = pd.DataFrame(data=popts*np.array([1., 1000., 1]), index=index,
                      columns=['df', 'tau', 'f0'])
    df.to_csv(csv, index=True)  

    print(popts.mean(0))
    print(popts.std(0, ddof=1))
    df.to_csv()



@click.command()
@click.argument('filename', type=click.Path())
@click.argument('fp', type=float)
@click.argument('fc', type=float)
@click.argument('t_before', type=float)
@click.argument('t_after', type=float)
@click.option('--tbf', type=float, default=0.001)
@click.option('--taf', type=float, default=0.001)
@click.option('--output', '-o', type=str, default=None)
def adiabatic_phasekick_cli(filename, fp, fc, t_before, t_after, tbf, taf, output=None):
    if output is None:
        pdf = filename.replace('.h5', '.pdf')
        csv = filename.replace('.h5', '.csv')
    else:
        pdf = output+'.pdf'
        csv = output+'.csv'

    with h5py.File(filename, 'r') as fh:
        df = workup_adiabatic_w_control(fh, t_before, t_after, tbf, taf,
                        fp, fc, fs_dec=4*fc)


    df.to_csv(csv, index=False)
    fig, ax = plot_phasekick_control(df)
    fig.savefig(pdf)



@click.command()
@click.argument('fp', type=float)
@click.argument('fc', type=float)
@click.argument('f0', type=float)
@click.argument('fs', type=float)
@click.argument('output', type=click.Path())
@click.option('--coeff-ratio', default=8, help="Est. quality factor [unitless]")
@click.option('--coeffs', default=None, help="Est. spring const. [N/m]")
@click.option('--window', '-w', default="blackman", help='Window function')
def lockcli(fp, fc, f0, fs, output, coeff_ratio=8, coeffs=None,
            window='blackman'):
    fir = lock2(f0, fp, fc, fs, coeff_ratio, coeffs, window)
    with h5py.File(output, 'w') as fh:
        fh["fir"] = fir


@click.command()
@click.argument('fp', type=float)
@click.argument('fc', type=float)
@click.argument('f0', type=float)
@click.argument('fs', type=float)
@click.argument('iofile', type=click.Path())
@click.option('--coeff-ratio', default=8,
              help="ratio of coefficients to use (default: 8)")
@click.option('--coeffs', default=None, help="number of coefficients")
@click.option('--window', '-w', default="blackman", help='Window function')
@click.option('--phi0', default=0, help="Initial phase")
@click.option('--t0', default=0, help="Initial time")
def firlockstate(fp, fc, f0, fs, iofile, coeff_ratio=8, coeffs=None,
                 window='blackman', phi0=0, t0=0):
    fir = lock2(f0, fp, fc, fs, coeff_ratio, coeffs, window)
    dec = int(np.floor(fs / (4*fc)))
    new_dt = dec / fs
    firlock = FIRStateLock(fir, dec, f0, phi0, t0, fs)
    with h5py.File(iofile, 'r+') as fh:
        fh["fir"] = fir
        firlock.filt(fh["x"][:])
        fh["z"] = firlock.z_out
        fh["dphi"] = np.unwrap(np.angle(firlock.z_out))
        fh["dphi"].attrs["units"] = "radians"
        fh["df"] = np.gradient(np.unwrap(np.angle(firlock.z_out))) * (
            fs / (dec * 2*np.pi)
        )
        fh["f0"] = f0
        fh["new_dt"] = new_dt
        fh["new_fs"] = fs / dec
        fh["new_t0"] = firlock.t0_dec
