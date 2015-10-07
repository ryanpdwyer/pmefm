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
          window='blackman'):
    nyq = fs/2
    fp = fp / nyq
    fc = fc / nyq

    if coeffs is None:
        coeffs = int(round(coeff_ratio / fc, 0))

    # Force number of coefficients even
    alpha = (1-fp*1.0/fc)
    n = int(round(1000. / alpha) // 2)
    N = n * 2 + 1
    f = np.linspace(0, fc, n+1)

    fm = np.zeros(n + 2)
    mm = np.zeros(n + 2)
    fm[:-1] = f
    fm[-1] = 1.
    m = signal.tukey(N, alpha=alpha)
    mm[:-1] = m[n:]

    b = signal.firwin2(coeffs, fm, mm,
                       nfreqs=2**(int(round(np.log2(coeffs)))+3)+1,
                       window=window)

    b = b / np.sum(b)

    w, rep = signal.freqz(b, worN=np.pi*np.array([0., fp/2, fp, fc,
                                                  0.5*f0/nyq, f0/nyq, 1.]))

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
        nyq = fs/2
        if f0 is None:
            self.f0 = f0 = self.f0_est

        if fp is None:
            fp = fp_ratio * f0 / nyq
        else:
            fp = fp / nyq

        if fc is None:
            fc = fc_ratio * f0 / nyq
        else:
            fc = fc / nyq

        if coeffs is None:
            coeffs = round(coeff_ratio / fc, 0)

        alpha = (1-fp*1.0/fc)
        n = int(round(1000. / alpha) // 2)
        N = n * 2 +1
        f = np.linspace(0, fc, n+1)

        fm = np.zeros(n + 2)
        mm = np.zeros(n + 2)
        fm[:-1] = f
        fm[-1] = 1.
        m = signal.tukey(N, alpha=alpha)
        mm[:-1] = m[n:]

        b = signal.firwin2(coeffs, fm, mm,
                                    nfreqs=2**(int(round(np.log2(coeffs)))+3)+1,
                                    window=window)

        self.b = b = b / np.sum(b)

        w, rep = signal.freqz(b, worN=np.pi*np.array([0., fp/2, fp, fc, 0.5*f0/nyq, f0/nyq, 1.]))

        print("Response:")
        _print_magnitude_data(w, rep, fs)

        if coeffs > self.x.size:
            raise ValueError(
    """No valid output when 'coeffs' > t.size (coeffs: {}, t.size: {}).
    Reduce coeffs by increasing bw, bw_ratio, or decreasing coeff_ratio,
    or provide more data.""".format(coeffs, t.size))

        self.z = z = signal.fftconvolve(self.x * np.exp(-2j*np.pi*f0*t), 2*b,
                                       "same")

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


        self.mb = mb = auto_phase(t[mask], z[mask], x0)

        self.phi_fit = np.polyval(mb, t)

        self.dphi = np.unwrap(((self.phi - self.phi_fit + np.pi) % (2*np.pi)) - np.pi)

        self.df = np.zeros(t.size)
        self.df[1:] = np.diff(self.dphi) * self.fs / (2*np.pi)

        self.f0 = self.f0 + mb[0]

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

        if weight:
            A = abs(z[mask]) / np.std(abs(z[mask]))
            self.mb = mb = np.polyfit(t[mask], phi_norm, 1, w=A) * std
        else:
            self.mb = mb = np.polyfit(t[mask], phi_norm, 1) * std

        self.phi_fit = np.polyval(mb, t)

        self.dphi = np.unwrap(((self.phi - self.phi_fit + np.pi) % (2*np.pi)) - np.pi)

        self.df = np.zeros(t.size)
        self.df[1:] = np.diff(self.dphi) * self.fs / (2*np.pi)

        self.f0 = self.f0 + mb[0] / (2*np.pi)

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

        if weight:
            A = abs(z[mask]) / np.std(abs(z[mask]))
            self.mb = mb = np.polyfit(t[mask], phi_norm[mask], 1, w=A) * std
        else:
            self.mb = mb = np.polyfit(t[mask], phi_norm[mask], 1) * std

        phi_fit = np.polyval(mb, t)

        dphi = np.unwrap(((phi - phi_fit + np.pi) % (2*np.pi)) - np.pi)

        df = np.zeros(t.size)
        df[1:] = np.diff(dphi) * self.dec_fs / (2*np.pi)

        self.f0_dec_direct = self.f0 + mb[0] / (2*np.pi)




class FIRState(object):
    def __init__(self, fir, dec, t0=0., fs=1.):
        self.fir = fir
        self.nfir_mid = (len(fir) -1)//2
        self.dec = dec
        self.t0 = t0
        self.fs = fs
        self.t0_dec = t0 + self.nfir_mid / self.fs
        self.data = np.array([])
        self.output = np.array([])
        

    def filt(self, data):
        fir = self.fir
        n = self.fir.size
        x = np.r_[self.data, data]
        y = signal.fftconvolve(x, self.fir, mode="full")
        indices = np.arange(y.size)
        # Valid region mask
        # This is borrowed explicitly from scipy.signal.sigtools.fftconvolve
        m = indices[n-1:-n]
        if len(m) == 0:
            self.data = x
        else:
            m_dec = m[::self.dec]
            self.output = np.r_[self.output, y[m_dec]]
            self.data = x[m_dec[-1] - (n-1) + self.dec:]
            print(m_dec)


class FIRStateLock(object):
    def __init__(self, fir, dec, f0, phi0, t0=0, fs=1.):
        self.fir = fir
        self.nfir_mid = (len(fir) - 1)//2
        self.dec = dec
        self.f0 = f0
        self.w0 = f0/fs
        self.phi0 = self.phi_i = phi0
        self.t0 = t0
        self.fs = fs
        self.t0_dec = t0 + self.nfir_mid / self.fs
        self.z = np.array([], dtype=np.complex128)
        self.z_out = np.array([], dtype=np.complex128)

    def filt(self, data):
        n = self.fir.size
        phi = -2*np.pi*self.w0*np.arange(1, data.size+1) + self.phi_i
        self.phi_i = phi[-1] % (2*np.pi)
        z_new = data * np.exp(1j*phi)

        z = np.r_[self.z, z_new]
        y = signal.fftconvolve(z, 2*self.fir, mode="full")
        indices = np.arange(y.size)
        # Valid region mask
        # This is borrowed explicitly from scipy.signal.sigtools.fftconvolve
        m = indices[n-1:-n]
        if len(m) == 0:
            self.z = z
        else:
            m_dec = m[::self.dec]
            self.z_out = np.r_[self.z_out, y[m_dec]]
            self.z = z[m_dec[-1] - (n-1) + self.dec:]

    def get_t(self):
        return self.t0_dec + np.arange(self.z_out.size)/self.fs * self.dec


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
