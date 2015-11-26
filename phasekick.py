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
import io
import pathlib
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.signal.signaltools import _centered
import lockin

import docutils.core
import base64
import bs4

def phase_step(t, tau, df):
    return df*t + df*tau*(np.exp(-t/tau)-1)

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

    control = df.xs('control')
    data = df.xs('data')


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
         'lis': lis,
         'locks': locks}

    # Do a fit to the corrected phase data here.

    return df, extras


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

    ax2.plot(td, control['dphi [cyc]']*1e3, 'b.')
    ax2.plot(td, offset_cos(control['phi_at_tp [rad]'], *extras['popt_phi'])*1e3)

    ax1.grid()
    ax2.grid()
    ax1.set_xlabel(r'$\tau_\mathrm{d} \: [\mu\mathrm{s}]$')
    ax1.set_ylabel(r'$\Delta A \: [\mathrm{nm}]$')
    ax2.set_ylabel(r'$\Delta \phi \: [\mathrm{mcyc.}]$')

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
    return fig, (ax1, ax2)


def plot_phasekick(df, figax=None, rcParams={}, filename=None):
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

    ax.set_xlabel('Pulse time [ms]')
    ax.set_ylabel('Phase shift [{}.]'.format(units))

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')

    return fig, ax

def plot_phasekick_corrected(df, figax=None, rcParams={}, filename=None):
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

    soup = bs4.BeautifulSoup(html_text)

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



ReST_temp = u"""
Phasekick report
================

::

        File: {filename}
    T_before: {T_before:.3e}
     T_after: {T_after:.3e}
        T_bf: {T_bf:.3e}
        T_af: {T_af:.3e}
          fp: {fp}
          fc: {fc}
      fs_dec: {fs_dec}


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
    T_before, T_after, T_bf, T_af, fp, fc, fs_dec, basename=None, ext='html'):
    if basename is None:
        basename = os.path.splitext(filename)[0]
    with h5py.File(filename, 'r') as fh:
        df, extras = workup_adiabatic_w_control_correct_phase(fh,
            T_before, T_after, T_bf, T_af, fp, fc, fs_dec)

        d = {'outf_zero_time': basename+'-zerotime.png',
         'outf_phasekick': basename+'-phasekick-uncorrected.png',
         'outf_phasekick_corr': basename+'-phasekick-corrected.png',
         'outf_amplitudes': basename+'-amplitude.png',
         'outf_frequency_shift': basename+'-frequency-shift.png',
         'outf_amp_phase_resp': basename+'-dA-dphi.png',
         'basename': basename,
         'filename': filename,
         'T_before': T_before,
         'T_after': T_after,
         'T_bf': T_bf,
         'T_af': T_af,
         'fp': fp,
         'fc': fc,
         'fs_dec': fs_dec,}

        d.update(extras)

        plot_zero_time(extras, filename=d['outf_zero_time'])
        plot_phasekick(df, filename=d['outf_phasekick'])
        plot_phasekick_corrected(df,
            filename=d['outf_phasekick_corr'])
        plot_amplitudes(extras, filename=d['outf_amplitudes'])
        plot_df0vs_t(df, filename=d['outf_frequency_shift'])
        plot_dA_dphi_vs_t(df, extras, filename=d['outf_amp_phase_resp'])


    ReST = ReST_temp.format(**d)
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



@click.command()
@click.argument('filename', type=click.Path())
@click.argument('fp', type=float)
@click.argument('fc', type=float)
@click.argument('t_before', type=float)
@click.argument('t_after', type=float)
@click.option('--tbf', type=float, default=0.001)
@click.option('--taf', type=float, default=0.001)
@click.option('--basename', type=str, default=None)
def report_adiabatic_control_phase_corr_cil(filename,
    T_before, T_after, T_bf, T_af, fp, fc, fs_dec, basename, ext='html'):
    
    report_adiabatic_control_phase_corr(filename,
    T_before, T_after, T_bf, T_af, fp, fc, fs_dec, basename, ext='html')









    
