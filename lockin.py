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
              fp=None, fc=None, coeffs=None, window='blackman'):
        t = self.t
        fs = self.fs

        if f0 is None:
            self.f0 = f0 = self.f0_est

        if fp is None:
            fp = fp_ratio * f0

        if fc is None:
            fc = fc_ratio * f0

        self.fir = b = lock2(f0, fp, fc, fs, coeff_ratio, coeffs, window)

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




def workup_gr(ds, T_before, T_after, T_bf=0.03, T_af=0.06, fp=1000, fc=4000,
              fs_dec=16000, t0=0.05):
    """Lockin workup of the data in h5py dataset ds.
    This assumes that ds contains attributes dt, pulse time, which enable us to
    perform the workup."""
    dt = ds.attrs['dt']
    tp = ds.attrs['pulse time']
    fs = 1. / dt
    y = ds[:]
    x = np.arange(y.size, dtype=np.float64)*dt
    li = LockIn(x, y, fs)
    li.lock2(fp=fp, fc=fc)
    tedge = li.fir.size * dt / 2
    li.phase(ti=t0-T_bf-tedge, tf=t0-tedge)
    f1 = li.f0corr
    phi0 = (- li.phi[0])
    li.phase(ti=(t0+tp+tedge), tf=(t0+tp+T_af+tedge))
    f2 = li.f0corr
    dec = int(np.floor(fs / fs_dec))
    def f_var(t):
        return np.where(t > t0 + tp, f2, f1)
    lockstate = FIRStateLockVarF(li.fir, dec, f_var, phi0, fs=fs)
    lockstate.filt(y)
    lockstate.dphi = np.unwrap(np.angle(lockstate.z_out))
    lockstate.df = np.gradient(lockstate.dphi) * (
            fs / (dec * 2*np.pi))
    
    lockstate.tp = tp
    lockstate.t = t = lockstate.get_t()
    lockstate.delta_phi = (np.mean(lockstate.dphi[(t >= (t0 + tp)) & (t < t0 + tp + T_after)]) - 
                            np.mean(lockstate.dphi[(t >= (t0 - T_before)) & (t < t0)])
                           )
    return lockstate

def workup_file(gr, out_file, T_before, T_after,
                T_bf=0.03, T_af=0.06, fp=1000, fc=4000, fs_dec=16000, t0=0.05,
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
