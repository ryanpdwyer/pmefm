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
import os
from tqdm import tqdm
import io
import pathlib
from scipy import interpolate
from scipy import signal
from scipy import stats
from six import string_types
from scipy.optimize import curve_fit
from scipy.signal.signaltools import _centered
import lockin
import phasekick2

import docutils.core
import base64
import bs4

def expon_weights(tau, fs, coeff_ratio=5.):
    scale = tau * fs
    i = np.arange(int(round(scale * coeff_ratio)))
    return stats.expon.pdf(i, scale=scale)


def slope_filt(N):
    pass


def percentile_func(x):
    return lambda p: np.percentile(x, p, axis=0)

def masklh(x, l=None, r=None):
    if l is None:
        return (x < r)
    elif r is None:
        return (x >= l)
    else:
        return (x >= l) & (x < r)


def prnDict(aDict, br='\n', html=0,
            keyAlign='l',   sortKey=0,
            keyPrefix='',   keySuffix='',
            valuePrefix='', valueSuffix='',
            leftMargin=4,   indent=1, braces=True):
    '''
return a string representive of aDict in the following format:
    {
     key1: value1,
     key2: value2,
     ...
     }

Spaces will be added to the keys to make them have same width.

sortKey: set to 1 if want keys sorted;
keyAlign: either 'l' or 'r', for left, right align, respectively.
keyPrefix, keySuffix, valuePrefix, valueSuffix: The prefix and
   suffix to wrap the keys or values. Good for formatting them
   for html document(for example, keyPrefix='<b>', keySuffix='</b>'). 
   Note: The keys will be padded with spaces to have them
         equally-wide. The pre- and suffix will be added OUTSIDE
         the entire width.
html: if set to 1, all spaces will be replaced with '&nbsp;', and
      the entire output will be wrapped with '<code>' and '</code>'.
br: determine the carriage return. If html, it is suggested to set
    br to '<br>'. If you want the html source code eazy to read,
    set br to '<br>\n'

version: 04b52
author : Runsun Pan
require: odict() # an ordered dict, if you want the keys sorted.
         Dave Benjamin 
         http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/161403
    '''
   
    if aDict:

        #------------------------------ sort key
        if sortKey:
            dic = aDict.copy()
            keys = dic.keys()
            keys.sort()
            aDict = odict()
            for k in keys:
                aDict[k] = dic[k]

        #------------------- wrap keys with ' ' (quotes) if str
        tmp = ['{']
        ks = [type(x)==str and "'%s'"%x or x for x in aDict.keys()]

        #------------------- wrap values with ' ' (quotes) if str
        vs = [type(x)==str and "'%s'"%x or x for x in aDict.values()] 

        maxKeyLen = max([len(str(x)) for x in ks])

        for i in range(len(ks)):

            #-------------------------- Adjust key width
            k = {1            : str(ks[i]).ljust(maxKeyLen),
                 keyAlign=='r': str(ks[i]).rjust(maxKeyLen) }[1]

            v = vs[i]
            tmp.append(' '* indent+ '%s%s%s:%s%s%s,' %(
                        keyPrefix, k, keySuffix,
                        valuePrefix,v,valueSuffix))

        tmp[-1] = tmp[-1][:-1] # remove the ',' in the last item
        tmp.append('}')

        if leftMargin:
          tmp = [ ' '*leftMargin + x for x in tmp ]

        if not braces:
            tmp = tmp[5:-2]

        if html:
            return '<code>%s</code>' %br.join(tmp).replace(' ','&nbsp;')
        else:
            return br.join(tmp)
    else:
        return '{}'


def phase_step(t, tau, df):
    return df*t + df*tau*(np.exp(-t/tau)-1)

def offset_cos(phi, A, phi0, b):
    return A*np.cos(phi + phi0) + b

def measure_dA_dphi(lock, li, tp, t_fit=2e-3,
                dphi_weight_before=None,
                dphi_weight_after=None):
    """Correct for impulsive phase shift at end of pulse time."""
    fs = li.fs
    if dphi_weight_before is None:
        N_b = int(round(fs*t_fit))
    else:
        N_b = len(dphi_weight_before)

    if dphi_weight_after is None:
        N_a = int(round(fs*t_fit))
    else:
        N_a = len(dphi_weight_after)

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

    ml = masklh(li.t, tp-t_fit, tp)
    mr = masklh(li.t, tp, tp + t_fit)

    ml_phi = np.arange(li.t.size)[li.t <= tp][-N_b:]
    mr_phi = np.arange(li.t.size)[li.t > tp][:N_a]

    A = abs(li.z_out)
    phi = np.unwrap(np.angle(li.z_out))/(2*np.pi)

    mbAl = np.polyfit(li.t[ml], A[ml], 1)
    mbAr = np.polyfit(li.t[mr], A[mr], 1)

    mb_phi_l = np.polyfit(li.t[ml_phi], phi[ml_phi], 1, w=dphi_weight_before)
    mb_phi_r = np.polyfit(li.t[mr_phi], phi[mr_phi], 1, w=dphi_weight_after)

    dA = np.polyval(mbAr, tp) - np.polyval(mbAl, tp)
    dphi = np.polyval(mb_phi_r, tp) - np.polyval(mb_phi_l, tp)

    return phi0, dA, dphi


def measure_dA_dphi_fir(lock, li, tp, dA_dphi_before, dA_dphi_after):
    """Correct for impulsive phase shift at end of pulse time."""

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

    ml = masklh(li.t, tp-t_fit, tp)
    mr = masklh(li.t, tp, tp + t_fit)

    A = abs(li.z_out)
    phi = np.unwrap(np.angle(li.z_out))/(2*np.pi)

    mbAl = np.polyfit(li.t[ml], A[ml], 1)
    mbAr = np.polyfit(li.t[mr], A[mr], 1)

    mb_phi_l = np.polyfit(li.t[ml], phi[ml], 1)
    mb_phi_r = np.polyfit(li.t[mr], phi[mr], 1)

    dA = np.polyval(mbAr, tp) - np.polyval(mbAl, tp)
    dphi = np.polyval(mb_phi_r, tp) - np.polyval(mb_phi_l, tp)

    return phi0, dA, dphi

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
        for (tp_group, tp) in tqdm(zip(tp_groups, tps)):
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

            phi_at_tp, dA, dphi_tp_end = measure_dA_dphi(lock, li, tp)
            lis[control_or_data].append(li)
            locks[control_or_data].append(lock)
            i += 1

            curr_index = (control_or_data, tp_group)
            df.loc[curr_index, 'tp'] = tp
            df.loc[curr_index, 'dphi [cyc]'] = dphi/(2*np.pi)
            df.loc[curr_index, 'f0 [Hz]'] = f0_V0
            df.loc[curr_index, 'df_dV [Hz]'] = f0_V1 - f0_V0
            df.loc[curr_index, 'dA [nm]'] = dA
            df.loc[curr_index, 'dphi_tp_end [cyc]'] = dphi_tp_end
            df.loc[curr_index, 'phi_at_tp [rad]'] = phi_at_tp
            df.loc[curr_index, 'relative time [s]'] = gr['relative time [s]'].value

    sys.stdout.write('\n')

    df.sort_index(inplace=True)

    control = df.xs('control')
    data = df.xs('data')
    popt_phi, pcov_phi = optimize.curve_fit(offset_cos,
        control['phi_at_tp [rad]'], control['dphi_tp_end [cyc]'])
    popt_A, pcov_A = optimize.curve_fit(offset_cos,
        control['phi_at_tp [rad]'], control['dA [nm]'])

    df['dphi_corrected [cyc]'] = (df['dphi [cyc]']
                            - offset_cos(df['phi_at_tp [rad]'], *popt_phi))

    control = df.xs('control')
    data = df.xs('data')


    popt_phase_corr, pcov_phase_corr = optimize.curve_fit(phase_step, data['tp'], data['dphi_corrected [cyc]'])
    popt_phase, pcov_phase = optimize.curve_fit(phase_step, data['tp'], data['dphi [cyc]'])

    # Extra information
    extras = {'popt_phi': popt_phi,
         'pcov_phi': pcov_phi,
         'pdiag_phi': np.diagonal(pcov_phi)**0.5,
         'popt_A': popt_A,
         'pcov_A': pcov_A,
         'pdiag_A': np.diagonal(pcov_A)**0.5,
         'popt_phase_corr': popt_phase_corr,
         'pcov_phase_corr': pcov_phase_corr,
         'pdiag_phase_corr': np.diagonal(pcov_phase_corr)**0.5,
         'popt_phase': popt_phase,
         'pcov_phase': pcov_phase,
         'pdiag_phase': np.diagonal(pcov_phase)**0.5,
         'T_before': T_before,
         'T_after': T_after,
         'T_bf': T_bf,
         'T_af': T_af,
         'fp': fp,
         'fc': fc,
         'basename': basename,
         'filename': filename,
         'fs_dec': fs_dec,
         'file_attrs_str': prnDict(dict(fh.attrs.items()), braces=False),
         'dataset_attrs_str': prnDict(dict(fh['data/0000'].attrs.items()), braces=False),
         'lis': lis,
         'locks': locks}

    # Do a fit to the corrected phase data here.

    return df, extras



def workup_adiabatic_w_control_correct_phase_bnc(fh, T_before, T_after, T_bf, T_af,
                        fp, fc, fs_dec):
    tps = fh['tp tip [s]'][:] # ms to s
    tp_groups = fh['ds'][:]
    df = pd.DataFrame(index=pd.MultiIndex.from_product(
        (['data', 'control'], tp_groups), names=['expt', 'ds']))
    lis = {}
    locks = {}
    i = 0
    for control_or_data in ('control', 'data'):
        lis[control_or_data] = []
        locks[control_or_data] = []
        for (tp_group, tp) in tqdm(zip(tp_groups, tps)):
            gr = fh[control_or_data][tp_group]
            print_response = i == 0
            try:
                N2even = gr.attrs['Calc BNC565 CantClk.N2 (even)']
                t1 = gr.attrs['Abrupt BNC565 CantClk.t1 [s]']
                t2 = np.sum(gr["half periods [s]"][:N2even+1])
                tp = np.sum(gr.attrs["Abrupt BNC565 CantClk.tp tip [s]"])
                t0 = -(t1 + t2)
                x = gr['cantilever-nm'][:]
                dt = fh['data/0000/dt [s]'].value
                fs = 1. / dt
                lock = lockin.adiabatic2lockin(gr, t0=t0)
                lock.lock2(fp=fp, fc=fc, print_response=False)
                lock.phase(tf=-t2)
                f0_V0 = lock.f0corr
                lock.phase(ti=-t2, tf=0)
                f0_V1 = lock.f0corr
                dphi, li = lockin.delta_phi_group(
                    gr, tp, T_before, T_after,
                    T_bf, T_af, fp, fc, fs_dec, print_response=print_response,
                    t0=t0)

                phi_at_tp, dA, dphi_tp_end = measure_dA_dphi(lock, li, tp)
                lis[control_or_data].append(li)
                locks[control_or_data].append(lock)
                i += 1

                curr_index = (control_or_data, tp_group)
                df.loc[curr_index, 'tp'] = tp
                df.loc[curr_index, 'dphi [cyc]'] = dphi/(2*np.pi)
                df.loc[curr_index, 'f0 [Hz]'] = f0_V0
                df.loc[curr_index, 'df_dV [Hz]'] = f0_V1 - f0_V0
                df.loc[curr_index, 'dA [nm]'] = dA
                df.loc[curr_index, 'dphi_tp_end [cyc]'] = dphi_tp_end
                df.loc[curr_index, 'phi_at_tp [rad]'] = phi_at_tp
                df.loc[curr_index, 'relative time [s]'] = gr['relative time [s]'].value
            except Exception as e:
                print(e)
                pass

    try:
        df.sort_index(inplace=True)
        print(df)

        df_clean = df.dropna()

        control = df_clean.xs('control')
        data = df_clean.xs('data')

        popt_phi, pcov_phi = optimize.curve_fit(offset_cos,
            control['phi_at_tp [rad]'], control['dphi_tp_end [cyc]'])
        popt_A, pcov_A = optimize.curve_fit(offset_cos,
            control['phi_at_tp [rad]'], control['dA [nm]'])

        df['dphi_corrected [cyc]'] = (df['dphi [cyc]']
                                - offset_cos(df['phi_at_tp [rad]'], *popt_phi))

        df_clean = df.dropna()
        control = df_clean.xs('control')
        data = df_clean.xs('data')

        # Issues here; not properly sorted, etc.
        popt_phase_corr, pcov_phase_corr = optimize.curve_fit(phase_step, data['tp'], data['dphi_corrected [cyc]'])
        popt_phase, pcov_phase = optimize.curve_fit(phase_step, data['tp'], data['dphi [cyc]'])

        # Extra informationdd
        extras = {'popt_phi': popt_phi,
             'pcov_phi': pcov_phi,
             'pdiag_phi': np.diagonal(pcov_phi)**0.5,
             'popt_A': popt_A,
             'pcov_A': pcov_A,
             'pdiag_A': np.diagonal(pcov_A)**0.5,
             'popt_phase_corr': popt_phase_corr,
             'pcov_phase_corr': pcov_phase_corr,
             'pdiag_phase_corr': np.diagonal(pcov_phase_corr)**0.5,
             'popt_phase': popt_phase,
             'pcov_phase': pcov_phase,
             'pdiag_phase': np.diagonal(pcov_phase)**0.5,
             'T_before': T_before,
             'T_after': T_after,
             'T_bf': T_bf,
             'T_af': T_af,
             'fp': fp,
             'fc': fc,
             'fs_dec': fs_dec,
            'basename': basename,
            'filename': filename,
            'fs_dec': fs_dec,
            'file_attrs_str': prnDict(dict(fh.attrs.items()), braces=False),
            'dataset_attrs_str': prnDict(dict(fh['data/0000'].attrs.items()), braces=False),
             'lis': lis,
             'locks': locks}

        return df, extras

    except Exception as e:
        print(e)
        raise


def fir_tau(tau, fs, ratio=None, N=None, T=None):
    if N is not None:
        pass
    elif T is not None:
        N = int(T*fs)
    elif ratio is not None:
        N = int(fs*tau*ratio)
    else:
        raise ValueError("Must specify N or T")
    
    t = np.arange(N)/fs
    h_raw = np.exp(-t/tau)
    return h_raw / sum(h_raw)


def workup_adiabatic_w_control_correct_phase_bnc2(fh, 
                                                  fp, fc, fs_dec, correct_fir,
                                                  tau_b=None, tau_a=None,
                                                  before_fir=None,
                                                  after_fir=None,
                                                  ):
    tps = fh['tp tip [s]'][:] # ms to s
    tp_groups = fh['ds'][:]
    df = pd.DataFrame(index=pd.MultiIndex.from_product(
        (['data', 'control'], tp_groups), names=['expt', 'ds']))
    lis = {}
    locks = {}
    i = 0
    dt = fh['data/0000/dt [s]'].value
    fs = 1./dt
    lock_fir = lockin.lock2(62e3, fp=fp, fc=fc, fs=fs, coeff_ratio=8,
                                        window='blackman', print_response=False)

    N_dec = int(fs/fs_dec)
    if before_fir is None:
        before_fir = fir_tau(tau_b, fs, ratio=8)
    if after_fir is None:
        after_fir = fir_tau(tau_a, fs, ratio=8)

    for control_or_data in ('control', 'data'):
        lis[control_or_data] = []
        locks[control_or_data] = []
        for (tp_group, tp) in tqdm(zip(tp_groups, tps)):
            gr = fh[control_or_data][tp_group]
            print_response = i == 0
            try:
                N2even = gr.attrs['Calc BNC565 CantClk.N2 (even)']
                t1 = gr.attrs['Abrupt BNC565 CantClk.t1 [s]']
                t2 = np.sum(gr["half periods [s]"][:N2even+1])
                tp = np.sum(gr.attrs["Abrupt BNC565 CantClk.tp tip [s]"])
                t0 = -(t1 + t2)
                x = gr['cantilever-nm'][:]

                lock, li = phasekick2.individual_phasekick(x, dt, t0, t1, t2, tp, 
                    N_dec, lock_fir, before_fir, after_fir)
                dphi = li.delta_phi

                phi_at_tp, dA, dphi_tp_end = measure_dA_dphi(lock, li, tp)
                lis[control_or_data].append(li)
                locks[control_or_data].append(lock)
                i += 1

                curr_index = (control_or_data, tp_group)
                df.loc[curr_index, 'tp'] = tp
                df.loc[curr_index, 'dphi [cyc]'] = dphi/(2*np.pi)
                df.loc[curr_index, 'f0 [Hz]'] = li.fc0
                df.loc[curr_index, 'df_dV [Hz]'] = li.f1 - li.fc0
                df.loc[curr_index, 'df2_dV [Hz]'] = li.f2 - li.fc0
                df.loc[curr_index, 'dA [nm]'] = dA
                df.loc[curr_index, 'dphi_tp_end [cyc]'] = dphi_tp_end
                df.loc[curr_index, 'phi_at_tp [rad]'] = phi_at_tp
                df.loc[curr_index, 'relative time [s]'] = gr['relative time [s]'].value
            except Exception as e:
                print(e)
                pass

    try:
        df.sort_index(inplace=True)
        print(df)

        df_clean = df.dropna()

        control = df_clean.xs('control')
        data = df_clean.xs('data')

        popt_phi, pcov_phi = optimize.curve_fit(offset_cos,
            control['phi_at_tp [rad]'], control['dphi_tp_end [cyc]'])
        popt_A, pcov_A = optimize.curve_fit(offset_cos,
            control['phi_at_tp [rad]'], control['dA [nm]'])

        df['dphi_corrected [cyc]'] = (df['dphi [cyc]']
                                - offset_cos(df['phi_at_tp [rad]'], *popt_phi))

        df_clean = df.dropna()
        df_clean.sort_values('tp', inplace=True)
        control = df_clean.xs('control')
        data = df_clean.xs('data')


        popt_phase_corr, pcov_phase_corr = optimize.curve_fit(phase_step, data['tp'], data['dphi_corrected [cyc]'])
        popt_phase, pcov_phase = optimize.curve_fit(phase_step, data['tp'], data['dphi [cyc]'])

        # Extra informationdd
        extras = {'popt_phi': popt_phi,
             'pcov_phi': pcov_phi,
             'pdiag_phi': np.diagonal(pcov_phi)**0.5,
             'popt_A': popt_A,
             'pcov_A': pcov_A,
             'pdiag_A': np.diagonal(pcov_A)**0.5,
             'popt_phase_corr': popt_phase_corr,
             'pcov_phase_corr': pcov_phase_corr,
             'pdiag_phase_corr': np.diagonal(pcov_phase_corr)**0.5,
             'popt_phase': popt_phase,
             'pcov_phase': pcov_phase,
             'pdiag_phase': np.diagonal(pcov_phase)**0.5,
             'tau_b': tau_b,
             'tau_a': tau_a,
             'fp': fp,
             'fc': fc,
             'fs_dec': fs_dec,
             'lis': lis,
             'locks': locks}

        return df, extras

    except Exception as e:
        print(e)
        raise


def workup_adiabatic_w_control_correct_phase_bnc3(fh, 
                                                  fp, fc, fs_dec,
                                                  w_before=None,
                                                  w_after=None):
    tps = fh['tp tip [s]'][:]
    tp_groups = fh['ds'][:]
    df = pd.DataFrame(index=pd.MultiIndex.from_product(
        (['data', 'control'], tp_groups), names=['expt', 'ds']))
    lis = {}
    locks = {}
    i = 0
    dt = fh['data/0000/dt [s]'].value
    fs = 1./dt
    lock_fir = lockin.lock2(62e3, fp=fp, fc=fc, fs=fs, coeff_ratio=8,
                                        window='blackman', print_response=False)

    N_dec = int(fs/fs_dec)


    for control_or_data in ('control', 'data'):
        lis[control_or_data] = []
        locks[control_or_data] = []
        for (tp_group, tp) in tqdm(zip(tp_groups, tps)):
            gr = fh[control_or_data][tp_group]
            print_response = i == 0

            N2even = gr.attrs['Calc BNC565 CantClk.N2 (even)']
            t1 = gr.attrs['Abrupt BNC565 CantClk.t1 [s]']
            t2 = np.sum(gr["half periods [s]"][:N2even+1])
            tp = np.sum(gr.attrs["Abrupt BNC565 CantClk.tp tip [s]"])
            t0 = -(t1 + t2)
            x = gr['cantilever-nm'][:]

            lock, li = phasekick2.individual_phasekick2(x, dt, t0, t1, t2, tp, 
                N_dec, lock_fir, w_before, w_after)
            dphi = li.delta_phi

            phi_at_tp, dA, dphi_tp_end = measure_dA_dphi(lock, li, tp,
                dphi_weight_before=w_before, dphi_weight_after=w_after)
            lis[control_or_data].append(li)
            locks[control_or_data].append(lock)
            i += 1

            curr_index = (control_or_data, tp_group)
            df.loc[curr_index, 'tp'] = tp
            df.loc[curr_index, 'dphi [cyc]'] = dphi/(2*np.pi)
            df.loc[curr_index, 'f0 [Hz]'] = li.fc0
            df.loc[curr_index, 'df_dV [Hz]'] = li.f1 - li.fc0
            df.loc[curr_index, 'df2_dV [Hz]'] = li.f2 - li.fc0
            df.loc[curr_index, 'dA [nm]'] = dA
            df.loc[curr_index, 'dphi_tp_end [cyc]'] = dphi_tp_end
            df.loc[curr_index, 'phi_at_tp [rad]'] = phi_at_tp
            df.loc[curr_index, 'relative time [s]'] = gr['relative time [s]'].value

    try:
        df.sort_index(inplace=True)
        print(df)

        df_clean = df.dropna()

        control = df_clean.xs('control')
        data = df_clean.xs('data')

        popt_phi, pcov_phi = optimize.curve_fit(offset_cos,
            control['phi_at_tp [rad]'], control['dphi_tp_end [cyc]'])
        popt_A, pcov_A = optimize.curve_fit(offset_cos,
            control['phi_at_tp [rad]'], control['dA [nm]'])

        df['dphi_corrected [cyc]'] = (df['dphi [cyc]']
                                - offset_cos(df['phi_at_tp [rad]'], *popt_phi))

        df_clean = df.dropna()
        df_clean.sort_values('tp', inplace=True)
        control = df_clean.xs('control')
        data = df_clean.xs('data')


        popt_phase_corr, pcov_phase_corr = optimize.curve_fit(phase_step, data['tp'], data['dphi_corrected [cyc]'])
        popt_phase, pcov_phase = optimize.curve_fit(phase_step, data['tp'], data['dphi [cyc]'])

        # Extra information
        filename = fh.filename
        extras = {'popt_phi': popt_phi,
             'pcov_phi': pcov_phi,
             'pdiag_phi': np.diagonal(pcov_phi)**0.5,
             'popt_A': popt_A,
             'pcov_A': pcov_A,
             'pdiag_A': np.diagonal(pcov_A)**0.5,
             'popt_phase_corr': popt_phase_corr,
             'pcov_phase_corr': pcov_phase_corr,
             'pdiag_phase_corr': np.diagonal(pcov_phase_corr)**0.5,
             'popt_phase': popt_phase,
             'pcov_phase': pcov_phase,
             'pdiag_phase': np.diagonal(pcov_phase)**0.5,
              'params':
              {
              'filename': filename,
              'w_before': w_after,
             'w_after': w_before,
             'fp': fp,
             'fc': fc,
             'fs_dec': fs_dec,
             },
             'filename': filename,
             'file_attrs_str': prnDict(dict(fh.attrs.items()), braces=False),
             'dataset_attrs_str': prnDict(dict(fh['data/0000'].attrs.items()), braces=False),
             'lis': lis,
             'locks': locks}

        return df_clean, extras

    except Exception as e:
        print(e)
        raise



# Plot zero time
# Plot df0 vs t
# Plot phasekick, with fit (save fit parameters)

def plot_zero_time(extras, figax=None, rcParams={}, filename=None):
    if figax is None:
        with mpl.rc_context(rcParams):
            fig, ax = plt.subplots()
    else:
        fig, ax = figax

    data = extras['locks']['data']
    for li in data:
        m = (li.t >= -20e-6) & (li.t < 20e-6)
        ax.plot(li.t[m]*1e6, li.x[m], 'k', alpha=0.5)

    ax.set_xlabel(u"Time [µs]")
    ax.set_ylabel(u"x [nm]")
    ax.set_xlim(-20, 20)

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')

    return fig, ax

def plot_amplitudes(extras, figax=None, rcParams={}, filename=None):
    if figax is None:
        with mpl.rc_context(rcParams):
            fig, ax = plt.subplots()
    else:
        fig, ax = figax

    for li in extras['lis']['control']:
        ax.plot(li.get_t()*1e3, abs(li.z_out), 'green', alpha=0.5)
    for li in extras['lis']['data']:
        t = li.get_t()*1e3
        ax.plot(t, abs(li.z_out), 'b', alpha=0.5)

    ax.set_xlabel(u"Time [ms]")
    ax.set_ylabel(u"Amplitude [nm]")
    ax.set_xlim(t[0], t[-1])
    ax.grid()

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
    return fig, ax


def plot_df0vs_t(df, figax=None, rcParams={}, filename=None):
    if figax is None:
        with mpl.rc_context(rcParams):
            fig, ax = plt.subplots()
    else:
        fig, ax = figax

    data = df.loc['data']
    control = df.loc['control']

    ax.plot(data['relative time [s]'], data['df_dV [Hz]'], 'bo')
    ax.plot(control['relative time [s]'], control['df_dV [Hz]'], 'go')

    ax.set_xlabel('Relative time [s]')
    ax.set_ylabel('Frequency shift [Hz]')
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
    return fig, ax

def plot_dA_dphi_vs_t(df, extras, figax=None, rcParams={}, filename=None):
    if figax is None:
        with mpl.rc_context(rcParams):
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    else:
        fig, (ax1, ax2) = figax

    df_sorted = df.sort_values('phi_at_tp [rad]')

    control = df_sorted.loc['control']
    f0 = control['f0 [Hz]'].median()

    td = control['phi_at_tp [rad]']/(2*np.pi*f0) * 1e6
    ax1.plot(td, control['dA [nm]'], 'b.')
    ax1.plot(td, offset_cos(control['phi_at_tp [rad]'], *extras['popt_A']))

    ax2.plot(td, control['dphi_tp_end [cyc]']*1e3, 'b.')
    ax2.plot(td, offset_cos(control['phi_at_tp [rad]'], *extras['popt_phi'])*1e3)

    ax1.grid()
    ax2.grid()
    ax1.set_xlabel(r'$\tau_\mathrm{d} \; [\mu\mathrm{s}]$')
    ax1.set_ylabel(r'$\Delta A \; [\mathrm{nm}]$')
    ax2.set_ylabel(r'$\Delta \phi \; [\mathrm{mcyc.}]$')

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
    return fig, (ax1, ax2)


def plot_phasekick(df, extras, figax=None, rcParams={}, filename=None):
    if figax is None:
        with mpl.rc_context(rcParams):
            fig, ax = plt.subplots()
    else:
        fig, ax = figax

    data = df.loc['data']
    control = df.loc['control']

    if abs(data['dphi [cyc]']).max() > 0.15:
        units = 'cyc'
        scale = 1
    else:
        units = 'mcyc'
        scale = 1e3


    ax.plot(control.tp*1e3, control['dphi [cyc]']*scale, 'g.')
    ax.plot(data.tp*1e3, data['dphi [cyc]']*scale, 'b.')
    ax.plot(data.tp*1e3, phase_step(data.tp, *extras['popt_phase'])*scale)

    ax.set_xlabel('Pulse time [ms]')
    ax.set_ylabel('Phase shift [{}.]'.format(units))

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')

    return fig, ax

def plot_phasekick_corrected(df, extras, figax=None, rcParams={}, filename=None):
    if figax is None:
        with mpl.rc_context(rcParams):
            fig, ax = plt.subplots()
    else:
        fig, ax = figax

    data = df.loc['data']
    control = df.loc['control']

    if abs(data['dphi_corrected [cyc]']).max() > 0.15:
        units = 'cyc'
        scale = 1
    else:
        units = 'mcyc'
        scale = 1e3


    ax.plot(control.tp*1e3, control['dphi_corrected [cyc]']*scale, 'g.')
    ax.plot(data.tp*1e3, data['dphi_corrected [cyc]']*scale, 'b.')
    ax.plot(data.tp*1e3, phase_step(data.tp, *extras['popt_phase_corr'])*scale)

    ax.set_xlabel('Pulse time [ms]')
    ax.set_ylabel('Phase shift [{}.]'.format(units))
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
    return fig, ax


# Create a report
def file_extension(filename):
    """Return the file extension for a given filename. For example,

    Input               Output
    -------------       ---------
    data.csv            csv
    data.tar.gz         gz
    .vimrc              (empty string)"""

    return os.path.splitext(filename)[1][1:]

def img2uri(html_text):
    """Convert any relative reference img tags in html_input to inline data uri.
    Return the transformed html, in utf-8 format."""

    soup = bs4.BeautifulSoup(html_text, "lxml")

    image_tags = soup.find_all('img')

    for image_tag in image_tags:
        image_path = image_tag.attrs['src']
        if 'http' not in image_path:
            base64_text = base64.b64encode(open(image_path, 'rb').read())
            ext = file_extension(image_path)
            
            image_tag.attrs['src'] = (
                "data:image/{ext};base64,{base64_text}".format(
                    ext=ext, base64_text=base64_text)
                )

    return soup.prettify("utf-8")



ReST_temp3 = u"""\
================
Phasekick report
================


**Workup Parameters**

::

    {params_str}


**File Attributes**

::

    {file_attrs_str}



**Dataset Attributes**

::

    {dataset_attrs_str}




Best fit parameters
-------------------

::

     τ = {popt_phase[0]:.3e} ± {pdiag_phase[0]:.2e} s
    Δf = {popt_phase[1]:.3e} ± {pdiag_phase[1]:.2e} Hz


.. image:: {outf_phasekick}

Corrected best fit parameters
-----------------------------

::

     τ = {popt_phase_corr[0]:.3e} ± {pdiag_phase_corr[0]:.2e} s
    Δf = {popt_phase_corr[1]:.3e} ± {pdiag_phase_corr[1]:.2e} Hz



.. image:: {outf_phasekick_corr}

Amplitude phase response 
------------------------

.. image:: {outf_amp_phase_resp}


Frequency shift
---------------

.. image:: {outf_frequency_shift}


Amplitudes
----------

.. image:: {outf_amplitudes}


Zero time
---------

.. image:: {outf_zero_time}

"""

def report_adiabatic_control_phase_corr(filename,
    T_before, T_after, T_bf, T_af, fp, fc, fs_dec, basename=None, outdir=None, format='DAQ'):
    if basename is None:
        basename = os.path.splitext(filename)[0]

    if outdir is not None:
        basename = os.path.join(outdir, os.path.basename(basename))

    with h5py.File(filename, 'r') as fh:
        if format == 'DAQ':
            df, extras = workup_adiabatic_w_control_correct_phase(fh,
                T_before, T_after, T_bf, T_af, fp, fc, fs_dec)
        elif format == 'BNC':
            df, extras = workup_adiabatic_w_control_correct_phase_bnc(fh,
                T_before, T_after, T_bf, T_af, fp, fc, fs_dec)

        d = {'outf_zero_time': basename+'-zerotime.png',
         'outf_phasekick': basename+'-phasekick-uncorrected.png',
         'outf_phasekick_corr': basename+'-phasekick-corrected.png',
         'outf_amplitudes': basename+'-amplitude.png',
         'outf_frequency_shift': basename+'-frequency-shift.png',
         'outf_amp_phase_resp': basename+'-dA-dphi.png',}

        d.update(extras)


        plot_zero_time(extras, filename=d['outf_zero_time'])
        plot_phasekick(df, extras, filename=d['outf_phasekick'])
        plot_phasekick_corrected(df, extras,
            filename=d['outf_phasekick_corr'])
        plot_amplitudes(extras, filename=d['outf_amplitudes'])
        plot_df0vs_t(df, filename=d['outf_frequency_shift'])
        plot_dA_dphi_vs_t(df, extras, filename=d['outf_amp_phase_resp'])


    ReST = ReST_temp3.format(params_str**d)
    image_dependent_html = docutils.core.publish_string(ReST, writer_name='html')
    self_contained_html = unicode(img2uri(image_dependent_html), 'utf8')



    # io module instead of built-in open because it allows specifying
    # encoding. See http://stackoverflow.com/a/22288895/2823213
    with io.open(basename+'.html', 'w', encoding='utf8') as f:
        f.write(self_contained_html)

    df.to_csv(basename+'.csv')

    for fname in [fname for key, fname in d.items() if 'outf' in key]:
        try:
            os.remove(fname)
        except:
            pass


def report_adiabatic_control_phase_corr3(df, extras, basename=None, outdir=None):
    if basename is None:
        basename = os.path.splitext(extras['filename'])[0]

    if outdir is not None:
        basename = os.path.join(outdir, os.path.basename(basename))

    
    d = {
    'basename': basename,
    'outf_zero_time': basename+'-zerotime.png',
     'outf_phasekick': basename+'-phasekick-uncorrected.png',
     'outf_phasekick_corr': basename+'-phasekick-corrected.png',
     'outf_amplitudes': basename+'-amplitude.png',
     'outf_frequency_shift': basename+'-frequency-shift.png',
     'outf_amp_phase_resp': basename+'-dA-dphi.png',}

    d.update(extras)


    plot_zero_time(extras, filename=d['outf_zero_time'])
    plot_phasekick(df, extras, filename=d['outf_phasekick'])
    plot_phasekick_corrected(df, extras,
        filename=d['outf_phasekick_corr'])
    plot_amplitudes(extras, filename=d['outf_amplitudes'])
    plot_df0vs_t(df, filename=d['outf_frequency_shift'])
    plot_dA_dphi_vs_t(df, extras, filename=d['outf_amp_phase_resp'])


    ReST = ReST_temp3.format(params_str=prnDict(extras['params'],braces=False),
                            **d)
    image_dependent_html = docutils.core.publish_string(ReST, writer_name='html')
    self_contained_html = unicode(img2uri(image_dependent_html), 'utf8')



    # io module instead of built-in open because it allows specifying
    # encoding. See http://stackoverflow.com/a/22288895/2823213
    with io.open(basename+'.html', 'w', encoding='utf8') as f:
        f.write(self_contained_html)

    df.to_csv(basename+'.csv')

    for fname in [fname for key, fname in d.items() if 'outf' in key]:
        try:
            os.remove(fname)
        except:
            pass


def gr2t(gr):
    half_periods = gr["half periods [s]"][:]
    N2even = gr.attrs['Calc BNC565 CantClk.N2 (even)']
    t1 = gr.attrs['Abrupt BNC565 CantClk.t1 [s]']
    t2 = np.sum(gr["half periods [s]"][:N2even+1])
    tp = np.sum(gr["half periods [s]"][N2even+1:])
    t0 = -(t1 + t2)
    dt = gr['dt [s]'].value
    x = gr['cantilever-nm'][:]
    return np.arange(x.size)*dt + t0


def gr2lock(gr, fp=2000, fc=8000):
    half_periods = gr["half periods [s]"][:]
    N2even = gr.attrs['Calc BNC565 CantClk.N2 (even)']
    t1 = gr.attrs['Abrupt BNC565 CantClk.t1 [s]']
    t2 = np.sum(gr["half periods [s]"][:N2even+1])
    tp = np.sum(gr["half periods [s]"][N2even+1:])
    t0 = -(t1 + t2)
    x = gr['cantilever-nm'][:]
    dt = gr['dt [s]'].value
    t = np.arange(x.size)*dt + t0
    lock = lockin.LockIn(t, x, 1/dt)
    lock.lock2(fp=fp, fc=fc, print_response=False)
    lock.phase(ti=tp+0.5e-3, tf=tp+3.5e-3)
    lock.half_periods = half_periods
    lock.tp = tp
    lock.t2 = t2
    lock.t1 = t1
    return lock

def gr2lock_daq(gr, fp=2000, fc=8000):
    t1 = gr.attrs['Adiabatic Parameters.t1 [ms]'] * 0.001
    t2 = gr.attrs['Adiabatic Parameters.t2 [ms]'] * 0.001
    li = lockin.adiabatic2lockin(gr)
    li.lock2(fp=fp, fc=fc, print_response=False)
    li.t1 = t1
    li.t2 = t2
    return li


def workup_df(fname_or_fh, fp, fc, tmin, tmax,
              tiphase=None, tfphase=None, butter=None, periodogram=True):
    if isinstance(fname_or_fh, string_types):
        fh = h5py.File(fname_or_fh, 'r')
        fname = fname_or_fh
    else:
        fh = fname_or_fh
        fname = fh.filename

    lis = []
    for ds_name in tqdm(fh['ds']):
        li = gr2lock(fh['data'][ds_name], fp=fp, fc=fc)
        if butter is not None:
            li.lock_butter(**butter)
        if tfphase is None:
            tfphase = -li.t2
        li.phase(ti=tiphase, tf=tfphase)
        lis.append(li)

    ts = np.array([li('t') for li in lis])
    dfs = np.array([li('df') for li in lis])
    dphis = np.array([li('dphi') for li in lis])

    dfs_masked = []
    ts_masked = []
    dphis_masked = []
    for t, df, dphi in zip(ts, dfs, dphis):
        m = (t >= tmin) & (t < tmax)
        ts_masked.append(t[m])
        dfs_masked.append(df[m])
        dphis_masked.append(dphi[m])

    ts_masked = np.array(ts_masked)
    dfs_masked = np.array(dfs_masked)
    dphis_masked = np.array(dphis_masked)

    tfunc = percentile_func(ts_masked)
    dffunc = percentile_func(dfs_masked)

    d = {'lis': lis, 'ts': ts_masked, 'dfs': dfs_masked,
         'dphis': dphis_masked,
            'tf': tfunc, 'dff': dffunc, 'params': {
                'fname_or_fh': fname,
                'fp': fp,
                'fc': fc,
                'tmin': tmin,
                'tmax': tmax,
                'tiphase': tiphase,
                'tfphase': tfphase,
                'butter': butter
            },
            }

    if periodogram:
        psd_f, psd_phi = signal.periodogram(dphis_masked/(2*np.pi), fs=li.fs, detrend='linear')
        _, psd_df = signal.periodogram(dfs_masked, fs=li.fs, detrend='linear')
        d['_psd_phi_all'] = psd_phi
        d['_psd_df_all'] = psd_df
        d['psd_f'] = psd_f
        d['psd_phi'] = np.mean(psd_phi, axis=0)
        d['psd_df'] = np.mean(psd_df, axis=0)

    return d




def workup_df_plot(filename, outfile, fp, fc, tmin, tmax, saveh5=False, format_='BNC', subgr='data'):
    fh = h5py.File(filename, 'r')
    lis = []
    for ds_name in tqdm(fh['ds']):
        if format_ == 'BNC':
            li = gr2lock(fh[subgr][ds_name], fp=fp, fc=fc)
        elif format_ == 'DAQ':
            li = gr2lock_daq(fh[subgr][ds_name], fp=fp, fc=fc)
        else:
            raise ValueError("format_ must be 'BNC' or 'DAQ',\n not '{}'".format(format_))
        li.phase(tf= max(-li.t2, tmin+5e-3))
        lis.append(li)

    ts = np.array([li('t') for li in lis])
    dfs = np.array([li('df') for li in lis])

    dfs_masked = []
    ts_masked = []
    for t, df in zip(ts, dfs):
        m = (t >= tmin) & (t < tmax)
        ts_masked.append(t[m])
        dfs_masked.append(df[m])

    ts_masked = np.array(ts_masked)
    dfs_masked = np.array(dfs_masked)

    t50 = np.median(ts_masked, axis=0)
    df50 = np.percentile(dfs_masked, 50, axis=0)
    df15 = np.percentile(dfs_masked, 15.9, axis=0)
    df85 = np.percentile(dfs_masked, 84.1, axis=0)
    sigma = np.std(dfs_masked, ddof=1, axis=0)
    plt.plot(t50*1000, df50)
    plt.plot(t50*1000, df50, 'k', linewidth=2)
    plt.fill_between(t50*1000, df15, df85, color='0.2', alpha=0.3)
    plt.grid()
    plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if saveh5:
        basefile = os.path.splitext(outfile)[0]
        fh = h5py.File(basefile+'-df-avg'+'.h5', 'w')
        fh['t'] = t50
        fh['df50'] = df50
        fh['df85'] = df85
        fh['df15'] = df15
        fh['df_sigma'] = sigma
        fh['df_err'] = sigma / len(ts)**0.5
        fh.attrs['input_filename'] = filename
        fh.attrs['fp'] = fp
        fh.attrs['fc'] = fc
        fh.attrs['tmin'] = tmin
        fh.attrs['tmax'] = tmax


@click.command()
@click.argument('filename', type=click.Path())
@click.argument('fp', type=float)
@click.argument('fc', type=float)
@click.argument('tmin', type=float)
@click.argument('tmax', type=float)
@click.option('--outdir', type=str, default=None)
@click.option('--basename', type=str, default=None)
@click.option('--saveh5/--no-saveh5', default=False)
@click.option('--format', type=str, default='BNC')
@click.option('--group', type=str, default='data')
def df_vs_t_cli(filename, fp, fc, tmin, tmax, outdir, basename, saveh5, format, group):
    if basename is None:
        basename = os.path.splitext(filename)[0]

    if outdir is not None:
        basename = os.path.join(outdir, os.path.basename(basename))

    workup_df_plot(filename, basename+'.png', fp, fc, tmin, tmax, saveh5, format, group)




@click.command()
@click.argument('filename', type=click.Path())
@click.argument('fp', type=float)
@click.argument('fc', type=float)
@click.argument('t_before', type=float)
@click.argument('t_after', type=float)
@click.option('--tbf', type=float, default=0.001)
@click.option('--taf', type=float, default=0.001)
@click.option('--basename', type=str, default=None)
@click.option('--outdir', type=str, default=None)
@click.option('--format', type=str, default='DAQ')
def report_adiabatic_control_phase_corr_cli(filename,
    t_before, t_after, tbf, taf, fp, fc, basename, outdir, format):
    fs_dec = fc * 4
    
    try:
        report_adiabatic_control_phase_corr(filename,
    t_before, t_after, tbf, taf, fp, fc, fs_dec, basename, outdir, format)
    except Exception as e:
        print(e)
        pass




def align_and_mask(x, y, xi, xf):
    x_aligned = []
    y_aligned = []
    for x_, y_ in zip(x, y):
        m = (x_ >= xi) & (x_ < xf)
        x_aligned.append(x_[m])
        y_aligned.append(y_[m])
    
    return np.array(x_aligned), np.array(y_aligned)

def gr2t_df(gr, fp, fc, tf, pbar=None):
    lis = []
    for ds in gr.values():
        li = gr2lock(ds, fp=fp, fc=fc)
        li.phase(tf=tf)
        lis.append(li)
        if pbar is not None:
            pbar.update()
        
    ts = [li('t') for li in lis]
    dfs = [li('df') for li in lis]
    
    return ts, dfs
    
class AverageTrEFM(object):
    def __init__(self, ts, dfs, t_initial, t_final):
        self.ts = ts
        self.dfs = dfs
        self.t_initial = t_initial
        self.t_final = t_final
        
        self.t, self.df = align_and_mask(ts, dfs,
                                         t_initial, t_final)
        
        self.tm = np.mean(self.t, axis=0)
        self.tm_ms = self.tm*1e3
        self.tm_us = self.tm*1e6
        self.dfm = np.mean(self.df, axis=0)
        self.t50 = self.tp(50)
        self.df50 = self.dfp(50)
    
    def tp(self, p):
        return np.percentile(self.t, p, axis=0)
    
    def tp_ms(self, p):
        return self.tp(p)*1e3
    
    def tp_us(self, p):
        return self.tp(p)*1e6
    
    def dfp(self, p):
        return np.percentile(self.df, p, axis=0)
    
    @classmethod
    def from_group(cls, gr, fp, fc, tf, t_initial, t_final, pbar=None):
        ts, dfs = gr2t_df(gr, fp, fc, tf, pbar=pbar)
        
        return cls(ts, dfs, t_initial, t_final,)





    
