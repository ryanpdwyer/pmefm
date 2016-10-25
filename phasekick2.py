"""Phasekick2.py



"""
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
idx = pd.IndexSlice
import sigutils
import click
import h5py
import sys
import os
from tqdm import tqdm
import io
import pathlib
from scipy import interpolate
from scipy import signal
from six import string_types
from scipy.optimize import curve_fit
from scipy.signal.signaltools import _centered
import lockin
from lockin import LockIn, FIRStateLockVarF
from bunch import Bunch

# Inputs should be a dict of params


def individual_phasekick(y, dt, t0, t1, t2, tp, N_dec, lockin_fir, before_fir,
                         after_fir):
    """
        x
        fs
        t1
        t2
        tp
        lockin_fir (chosen by fp, fc)
        N_dec (chosen by int(fs/fs_dec))
        smooth_fir (chosen by opt. filt)
    """

    fs = 1. / dt
    t = np.arange(y.size) * dt + t0
    lock = LockIn(t, y, fs)
    lock.run(fir=lockin_fir)
    lock.phase(tf=-t2)
    fc0 = lock.f0corr
    N_b = before_fir.size
    N_a = after_fir.size
    # Find N_smooth points before begining of pulse
    # Flip filter coefficients for convolution
    df_before = np.dot(before_fir[::-1], lock.df[lock.t < 0][-N_b:])
    
    # Don't flip filter coefficients, this is 'anti-casual',
    # inferring f at tp from f at times t > tp
    df_after = np.dot(after_fir, lock.df[lock.t > tp][:N_a])

    f1 = fc0 + df_before
    f2 = fc0 + df_after

    lock.phase(ti=-dt*N_b/5, tf=0)
    phi0 = -lock.phi[0]
    def f_var(t):
        return np.where(t > tp, f2, f1)

    lockstate = FIRStateLockVarF(lockin_fir, N_dec, f_var, phi0, t0=t0, fs=fs)
    lockstate.filt(y)
    lockstate.dphi = np.unwrap(np.angle(lockstate.z_out))
    lockstate.df = np.gradient(lockstate.dphi) * (
            fs / (N_dec * 2*np.pi))
    lockstate.t = td = lockstate.get_t()
    
    lockstate.phi0 = np.dot(before_fir[::-1], lockstate.dphi[td < 0][-N_b:])
    lockstate.phi1 = np.dot(after_fir, lockstate.dphi[td > tp][:N_a])
    lockstate.delta_phi = lockstate.phi1 - lockstate.phi0
    # Save useful parameters for later use
    lockstate.tp = tp
    lockstate.fc0 = fc0
    lockstate.f1 = f1
    lockstate.f2 = f2
    return lock, lockstate


def individual_phasekick2(y, dt, t0, t1, t2, tp, N_dec, lockin_fir,
                          weight_before, weight_after):
    """
        x
        fs
        t1
        t2
        tp
        lockin_fir (chosen by fp, fc)
        N_dec (chosen by int(fs/fs_dec))
        weight_before (chosen by opt. filter)
        weight_after (chosen by opt. filter)
    """

    fs = 1. / dt
    t = np.arange(y.size) * dt + t0
    lock = LockIn(t, y, fs)
    lock.run(fir=lockin_fir)
    lock.phase(tf=-t2)
    fc0 = lock.f0corr
    N_b = weight_before.size
    N_a = weight_after.size
    # Find N_smooth points before begining of pulse
    # Flip filter coefficients for convolution
    # Could also fit phase to a line here
    df_before = np.polyfit(lock.t[lock.t < 0][-N_b:],
                           lock.df[lock.t < 0][-N_b:],
                           0,
                           w=weight_before[::-1])
    
    # Don't flip filter coefficients, this is 'anti-casual',
    # inferring f at tp from f at times t > tp
    df_after = np.polyfit(lock.t[lock.t > tp][:N_a],
                           lock.df[lock.t > tp][:N_a],
                           0,
                           w=weight_after)

    f1 = fc0 + df_before
    f2 = fc0 + df_after

    lock.phase(ti=-dt*N_b/5, tf=0)
    phi0 = -lock.phi[0]
    def f_var(t):
        return np.where(t > tp, f2, f1)

    lockstate = FIRStateLockVarF(lockin_fir, N_dec, f_var, phi0, t0=t0, fs=fs)
    lockstate.filt(y)
    lockstate.dphi = np.unwrap(np.angle(lockstate.z_out))
    lockstate.df = np.gradient(lockstate.dphi) * (
            fs / (N_dec * 2*np.pi))
    lockstate.t = td = lockstate.get_t()
    
    mb_before = np.polyfit(td[td < 0][-N_b:],
                           lockstate.dphi[td < 0][-N_b:],
                           1,
                           w=weight_before[::-1])

    mb_after = np.polyfit(td[td > tp][:N_a] - tp,
                         lockstate.dphi[td > tp][:N_a],
                         1,
                         w=weight_after)



    lockstate.mb_before = mb_before
    lockstate.mb_after = mb_after
    lockstate.phi0 = mb_before[1] + tp * mb_before[0]
    lockstate.phi1 = mb_after[1]
    lockstate.delta_phi = lockstate.phi1 - lockstate.phi0
    # Save useful parameters for later use
    lockstate.tp = tp
    lockstate.fc0 = fc0
    lockstate.f1 = f1
    lockstate.f2 = f2
    return lock, lockstate
