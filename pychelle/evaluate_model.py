#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from profiles import gauss
from helper_functions import wl_to_v, v_to_wl
from grism import lines_srs


def evaluate_row(data, wave, model, plot=True, cont_subtract=True):
    base_array = np.ones_like(wave) * float(cont_subtract)
    base_array *= model.Ampl.loc['Contin']
    compdict = {}
    for comp in model.drop('Contin').index.values:
        pars = model.loc[comp]
        pos = pars['Line center'] + pars.Pos
        sig = pars.Sigma
        ampl = pars.Ampl
        comparray = gauss(wave, pos, sig, ampl)
        base_array += comparray
        compdict[comp] = comparray
    resids = data - base_array
    if plot:
        plt.plot(wave, base_array, 'k-')
        plt.plot(wave, resids, 'b-')
        for co in compdict.keys():
            plt.plot(wave, compdict[co], ls='--', label=co)
        plt.legend()
        plt.axvline(0, color='k', ls='--')
        plt.show()
    return base_array, resids


def evaluate_transition(data, wave, model, cont_subtract=False,
                        vmin=None, vmax=None, viz=True):
    if vmin is None:
        vmin = np.median(data) - data.std()
    if vmax is None:
        vmax = data.mean() + data.std() * 3.
    resids = np.zeros_like(data)
    halpha = np.zeros_like(data)
    linecen = model['Line center'][0]
    wavmin, wavmax = linecen - 50, linecen + 50
    plotidx = np.where((wave > wavmin) & (wave < wavmax))[0]
    plotys = np.arange(data.shape[0])
    for r in model.index.levels[0]:
        # print r  # DEBUG
        idxno = float(r.split('-')[0]) - 1.
        halpha[idxno, :], resids[idxno, :] = evaluate_row(
            data[idxno, :],
            wave,
            model.loc[r],
            plot=False,
            cont_subtract=cont_subtract
        )#[1]
    if viz:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        orig = axes[0].pcolormesh(
            wave[plotidx], plotys,
            data[:, plotidx],
            cmap='cubehelix',
            vmax=vmax,
            vmin=vmin,
        )
        axes[1].pcolormesh(
            wave[plotidx], plotys,
            halpha[:, plotidx],
            cmap='cubehelix',
            vmax=vmax,
            vmin=vmin,
        )
        axes[2].pcolormesh(
            wave[plotidx], plotys,
            resids[:, plotidx],
            cmap='cubehelix',
            vmin=vmin,
            vmax=vmax,
        )
        axes[0].axis((linecen-40, linecen+40, 0, 90))
        axes[1].axis((linecen-40, linecen+40, 0, 90))
        axes[2].axis((linecen-40, linecen+40, 0, 90))
    return halpha, resids


def cut_transition(data, wave, transition, z=0, velspace=True, halfwidth=20):
    print transition
    # center = float(transition.split('_')[1])
    center = lines_srs.loc[transition] * (1.+z)
    center = center.values
    print center
    idx = np.where((wave > center - 20) &(wave < center + 20))[0]
    outdata = data[:, idx]
    outwave = wave[idx]
    if velspace:
        outwave = wl_to_v(outwave, center)
    return outwave, outdata
