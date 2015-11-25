# -*- coding: utf-8 -*-
"""
The phasekick workup is reproduced below. In particular, this file contains
a version of the 
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
import pathlib
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.signal.signaltools import _centered
import lockin

def offset_cos(phi, A, phi0, b):
    return A*np.cos(phi + phi0) + b

def measure_dA_dphi(lock, li, tp):
    """Correct for impulsive phase shift at end of pulse time.

    Note: This will currently work poorly for long pulse times.
    I should add an actual calculation of the phase shift directly at the end
    of the pulse time."""
    i_tp = np.arange(lock.t.size)[lock.t < tp][-1]
    # Use 20 data points for interpolating; this is slightly over one
    # cycle of our oscillation
    m = np.arange(-10, 11) + i_tp
    # This interpolator worked reasonably for similar, low-frequency sine waves
    interp = interpolate.KroghInterpolator(lock.t[m], lock.x[m])
    x0 = interp(tp)[()]
    # We only need t0 approximately; the precise value of f0 doesn't matter very much.
    t0 = li.t[(li.t < tp)][-1]
    f0 = li.df[(li.t < tp)][-1] + li.f0(t0)
    v0 = interp.derivative(tp)[()]
    x2 = v0 / (2*np.pi*f0)
    
    phi0 = np.arctan2(-x2, x0)
    
    ml = (li.t >= (tp - 10e-3)) & (li.t < (tp - 1e-3))
    mr = (li.t >= (tp + 1e-3)) & (li.t < (tp + 10e-3))
    A = abs(li.z_out)
    mbl = np.polyfit(li.t[ml], A[ml], 1)
    mbr = np.polyfit(li.t[mr], A[mr], 1)
    dA = np.polyval(mbr, tp) - np.polyval(mbl, tp)
    
    return phi0, dA

# Interface layers:
# -Raw hdf5 file to object / structure.

# What plots should I generate?
# dphi vs. dx

# Logical way to represent this information is a pandas MultiIndex.

def workup_adiabatic_w_control_correct_phase(fh, T_before, T_after, T_bf, T_af,
                        fp, fc, fs_dec):
    tps = fh['tp'][:] * 0.001 # ms to s
    tp_groups = fh['ds'][:]
    df = pd.DataFrame(index=pd.MultiIndex.from_product(
        (['data', 'control'], tp_groups), names=['expt', 'ds']))
    lis = {}
    locks = {}
    i = 0
    for control_or_data in ('control', 'data'):
        lis[control_or_data] = []
        locks[control_or_data] = []
        for (tp_group, tp) in zip(tp_groups, tps):
            gr = fh[control_or_data][tp_group]
            print_response = i == 0
            t1 = gr.attrs['Adiabatic Parameters.t1 [ms]'] * 0.001
            t2 = gr.attrs['Adiabatic Parameters.t2 [ms]'] * 0.001
            lock = lockin.adiabatic2lockin(gr)
            lock.lock2(fp=fp, fc=fc, print_response=False)
            lock.phase(tf=-t2)
            f0_V0 = lock.f0corr
            lock.phase(ti=-t2, tf=0)
            f0_V1 = lock.f0corr
            dphi, li = lockin.delta_phi_group(
                gr, tp, T_before, T_after,
                T_bf, T_af, fp, fc, fs_dec, print_response=print_response)

            phi_at_tp, dA = measure_dA_dphi(lock, li, tp)
            lis[control_or_data].append(li)
            locks[control_or_data].append(lock)
            i += 1
            sys.stdout.write('.')

            curr_index = (control_or_data, tp_group)
            df.loc[curr_index, 'tp'] = tp
            df.loc[curr_index, 'dphi [cyc]'] = dphi/(2*np.pi)
            df.loc[curr_index, 'f0 [Hz]'] = f0_V0
            df.loc[curr_index, 'df_dV [Hz]'] = f0_V1 - f0_V0
            df.loc[curr_index, 'dA [nm]'] = dA
            df.loc[curr_index, 'phi_at_tp [rad]'] = phi_at_tp
            df.loc[curr_index, 'relative time [s]'] = gr['relative time [s]'].value

    df.sort_index(inplace=True)

    control = df.xs('control')
    data = df.xs('data')
    popt_phi, pcov_phi = optimize.curve_fit(offset_cos,
        control['phi_at_tp [rad]'], control['dphi [cyc]'])
    popt_A, pcov_A = optimize.curve_fit(offset_cos,
        control['phi_at_tp [rad]'], control['dA [nm]'])

    df['dphi_corrected [cyc]'] = (df['dphi [cyc]']
                            - offset_cos(df['phi_at_tp [rad]'], *popt_phi))

    # Extra informationdd
    extras = {'popt_phi': popt_phi,
         'pcov_phi': pcov_phi,
         'popt_A': popt_A,
         'pcov_A': pcov_A,
         'lis': lis,
          'locks': locks}

    # Do a fit to the corrected phase data here.

    return df, extras


# Plot zero time
# Plot df0 vs t
# Plot phasekick, with fit (save fit parameters)

def plot_zero_time(extras, figax=None, rcParams={}):
    if figax is None:
        with mpl.rc_context(rcParams):
            fig, ax = plt.subplots()
    else:
        fig, ax = figax

    data = extras['locks']['data']
    for li in data:
        m = (li.t >= -20e-6) & (li.t < 20e-6)
        ax.plot(li.t[m]*1e6, li.x[m])

    ax.set_xlabel(u"Time [µs]")
    ax.set_ylabel(u"x [nm]")
    ax.set_xlim(-20e-6, 20e-6)
    return fig, ax

def plot_amplitudes(extras, figax=None, rcParams={}):
    if figax is None:
        with mpl.rc_context(rcParams):
            fig, ax = plt.subplots()
    else:
        fig, ax = figax

    for li in extras['lis']['control']:
        ax.plot(li.get_t()[::4]*1e3, abs(li.z_out)[::4], 'green', alpha=0.5)
    for li in extras['lis']['data']:
        t = li.get_t()
        ax.plot(t[::4]*1e3, abs(li.z_out)[::4], 'b', alpha=0.5)

    ax.set_xlabel(u"Time [ms]")
    ax.set_ylabel(u"Amplitude [nm]")
    ax.set_xlim(t[0], t[-1])
    ax.grid()
    return fig, ax


def plot_df0vs_t(df, figax=None, rcParams={}):
    if figax is None:
        with mpl.rc_context(rcParams):
            fig, ax = plt.subplots()
    else:
        fig, ax = figax

    data = df.loc['data']
    control = df.loc['control']

    ax.plot(data['relative time [s]'], data['df_dV [Hz]'], 'bo')
    ax.plot(control['relative time [s]'], control['df_dV [Hz]'], 'go')

    ax.set_xlim('Relative time [s]')
    ax.set_ylim('Frequency shift [Hz]')

    return fig, ax


def plot_phasekick(df, extras, figax=None, rcParams={}):
    if figax is None:
        with mpl.rc_context(rcParams):
            fig, ax = plt.subplots()
    else:
        fig, ax = figax

    data = df.loc['data']
    control = df.loc['control']

    if data['dphi_corrected [cyc]'].max() > 0.15:
        units = 'cyc'
        scale = 1
    else:
        units = 'mcyc'
        scale = 1e3


    ax.plot(control.tp*1e3, control['dphi [cyc]']*scale, 'g.')
    ax.plot(data.tp*1e3, data['dphi [cyc]']*scale, 'b.')

    ax.set_xlabel('Pulse time [ms]')
    ax.set_ylabel('Phase shift [{}.]'.format(units))

    return fig, ax

def plot_phasekick_corrected(df, extras, figax=None, rcParams={}):
    if figax is None:
        with mpl.rc_context(rcParams):
            fig, ax = plt.subplots()
    else:
        fig, ax = figax

    data = df.loc['data']
    control = df.loc['control']

    if data['dphi_corrected [cyc]'].max() > 0.15:
        units = 'cyc'
        scale = 1
    else:
        units = 'mcyc'
        scale = 1e3


    ax.plot(control.tp*1e3, control['dphi_corrected [cyc]']*scale, 'g.')
    ax.plot(data.tp*1e3, data['dphi_corrected [cyc]']*scale, 'b.')

    ax.set_xlabel('Pulse time [ms]')
    ax.set_ylabel('Phase shift [{}.]'.format(units))

    return fig, ax




