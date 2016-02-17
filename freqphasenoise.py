from __future__ import division, absolute_import, print_function
import numpy as np
from scipy import signal
import sigutils
import phasekick
import lockin
from bunch import Bunch

def expfall(t, tau, toff=np.inf):
    """Expontential decay, starting at t = 0, and abruptly going to 0
    at toff, with time constant tau."""
    return np.where((t >= 0) & (t < toff), np.exp(-t/tau), 0)


def psd2sigma(psd, fs):
    """Return gaussian standard deviation sigma necessary to produce
    white noise power spectral density ``psd`` at sampling frequency ``fs``."""
    return np.sqrt(psd*fs/2)


def x2phidet(x, p_x=2e-7):
    return 2 * p_x / x**2


def lp2diff(lp, fs, tp):
    Np = int(round(fs * tp, 0))
    diff = np.zeros(lp.size*2+Np)
    diff[:lp.size] = lp[::-1]
    diff[-lp.size:] = -lp
    
    return diff


def average_df(b, Navg, fp_ratio=1, fc_ratio=4):
    b.dt = dt = 1/b.fs
    b.N = N = int(b.T*b.fs)+1
    b.t = t = (np.arange(N) - (N-1)/2)*dt
    b.f = f = expfall(t, b.tau)*b.df_inf
    
    noise = np.random.randn(Navg, N) * psd2sigma(b.P_intrins, b.fs)
    m_phi_noise = np.random.randn(Navg, N) * psd2sigma(x2phidet(b.xA, p_x=b.P_det_x), b.fs)
    m_f_noise = np.c_[np.diff(m_phi_noise, axis=1), np.zeros(m_phi_noise.shape[0])] / dt
    f_noised = f + noise + m_f_noise

    fir = lockin.lock2(5e3, fp_ratio/b.tau, fc_ratio/b.tau, b.fs,
                       print_response=False)
    f_filt = np.empty_like(f_noised)
    for i, out in enumerate(f_noised):
        f_filt[i, :] = signal.fftconvolve(out, fir, mode='same')
    return b, f_filt


def several_dphi(b, fp, fc, tps = np.arange(0, 5, 0.5)):
    dt = 1/b.fs
    N = int(b.T*b.fs)+1
    t = (np.arange(N) - (N-1)/2)*dt
    
    P_det = x2phidet(b.xA, p_x=b.P_det_x)
    
    b2 = lockin.lock2(5e3, fp, fc, b.fs, coeff_ratio=4)
    lp = b2[b2.size//2:]
    lp = lp/sum(lp)

    dphis = np.empty_like(tps)
    tps = tps * b.tau
    for i, tp in enumerate(tps):
        f = expfall(t, b.tau, toff=tp)*b.df_inf
        noise = np.random.randn(N) * psd2sigma(b.P_intrins, b.fs)
        m_phi_noise = np.random.randn(N) * psd2sigma(x2phidet(b.xA, p_x=b.P_det_x), b.fs)
        phi = np.cumsum(f + noise)*dt + m_phi_noise
        diff = lp2diff(lp, b.fs, tp)
        phi_out = signal.fftconvolve(phi, diff, mode='same')
        ind = np.argmin(abs(t - tp/2.))
        dphis[i] = phi_out[ind]
    
    return tps, dphis