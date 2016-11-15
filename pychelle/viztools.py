#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plotting utilities that are neat to have when working with grism.
"""

import numpy as np
import uncertainties.unumpy as unp
import pandas as pd
import matplotlib.pyplot as plt
from lmfit_wrapper import build_model

#plt.rcdefaults()
#plt.rc('text', usetex=True)
#plt.rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'],
#                  'sans-serif':'Helvetica'})
#cf_fonts = {'family': 'serif',}
#plt.rc('font', **cf_fonts)

Paired = ['#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A1C',
          '#FDBF6F', '#FF7F00', '#CAB2D6', '#6A3D9A', '#FFFF99', '#B15928']
Pairdict = dict(
    zip(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', u'i', 'j', 'k', 'l'], Paired)
)


def prettify_axes(ax, lw=0.6):
    # for spine in ['left', 'right', 'top', 'bottom']:
    #     ax.spines[spine].set_linewidth(lw)
    [j.set_linewidth(.6) for j in ax.spines.itervalues()]
    ax.tick_params(labelsize=9, length=2)
    try:
        leg = ax.get_legend()
        leg.get_frame().set_facecolor('0.999')
        leg.get_frame().set_edgecolor('gray')
        leg.get_frame().set_linewidth(0.5)
        leg.get_title().set_size(8)
        leg.set_size(7)
    except:
        pass
    return ax


def component_wise_plot(
        inmodel, pars=['Rows', 'Pos'], ax=None, xerrbar=False, yerrbar=False,
        legend=True, errbarkwargs={'lw':1., 'ecolor':'0.8'},
        plotkwargs={'ms':5, 'mec':'0.4', 'mew':.5,}):
    """ Plots the different components color-coded.

    Params:
    -------
    model : pandas.DataFrame
        Must be 2-layer multiindex'd grism model,
        that is, the "transition" layer removed.
    pars : list of strings
        Names of 1 or 2 columns in the input frame to plot against each other.
        If only 1 is passed, it is assumed that the other is the pixel row nr.
    ax : matplotlib.Axes
        If not passed, a new one will be spawned.
    cont : bool
        Err, what was that for again?
    """
    if ax is None:
        try:
            ax = plt.gca()
        except:
            fig, ax = plt.subplots(1, 1)
    model = inmodel.copy()
    model.loc[pd.isnull(model.Identifier), 'Identifier'] = 'x'
    model['Complabel'] = model.Identifier.values
    model.Identifier = model.Identifier.map(ord) - 97
    # print set(model.Complabel)
    model = model.set_index('Complabel', append=True)
    if 'capsize' in errbarkwargs.keys():
        capsize = errbarkwargs['capsize']
    else:
        capsize=5
    if 'Component' in model.index.names:
        model = model.reset_index('Component')
    if 'Row' in model.columns:
        model['Rows'] = model.Row
    else:
        tmp = model.reset_index(level=0, drop=False)['Rows']\
            .map(lambda x: int(x.split('-')[0]))
        model['Rows'] = tmp.values
    components = set(model.index.get_level_values('Complabel')) - {'x'} # model.index.levels[1]
    for comp in components:
        tmp = model.swaplevel(0, 1).loc[comp].sort_index()
        cnum = tmp.Identifier[0] % 12
        # print cnum, comp
        if not xerrbar:
            xerr = np.ones_like(tmp[pars[0]]) * np.nan
        else:
            xerr = tmp[pars[0]+'_stddev']
        if not yerrbar:
            yerr = np.ones_like(tmp[pars[0]]) * np.nan
        else:
            yerr = tmp[pars[1]+'_stddev']
        ax.errorbar(
            tmp[pars[0]].values, tmp[pars[1]].values, xerr=xerr, yerr=yerr,
            fmt='none',capsize=0, **errbarkwargs
        )
        ax.errorbar(
            tmp[pars[0]].values, tmp[pars[1]].values, xerr=xerr, yerr=yerr,
            fmt='none', ecolor=Paired[cnum], lw=0, capsize=3,
            mew=errbarkwargs['lw'] + .2
        )
    for comp in components:
        tmp = model.swaplevel(0, 1).loc[comp].sort_index()
        cnum = tmp.Identifier[0] % 12
        if legend:
            label = comp
        else:
            label = '_nolabel'
        ax.plot(
            tmp[pars[0]], tmp[pars[1]], 's', color=Paired[cnum], label=label,
            **plotkwargs#Paired[cnum]
        )
    if legend:
        leg = ax.legend(numpoints=1, ncol=2)
        leg.draggable()
    #del(model, tmp)
    return ax


def SII_doublet_plot(spectrum, lines='S II', ax=None):
    model = spectrum.model.copy()
    if lines == 'O II':
        line1 = [line for line in model.index.levels[0]
                 if line.startswith('[O II]_3729')][0]
        line2 = [line for line in model.index.levels[0]
                 if line.startswith('[O II]_3727')][0]
    elif lines == 'S II':
        line1 = [line for line in model.index.levels[0]
                 if line.startswith('[S II]_6717')][0]
        line2 = [line for line in model.index.levels[0]
                 if line.startswith('[S II]_6731')][0]
    else:
        raise ValueError("Only S II6717 and O II 3727")

    fig, ax = plt.subplots(1, 1)
    doublet_plot(spectrum, ax=ax, lines=[line1, line2])

    return


def doublet_plot(spectrum, ax=None, lines=None):
    model = spectrum.model.copy()
    if lines is None:
        line1 = [line for line in model.index.levels[0]
                if line.startswith('[S II]_6717')][0]
        line2 = [line for line in model.index.levels[0]
                if line.startswith('[S II]_6731')][0]
    else:
        line1 = lines[0]
        line2 = lines[1]
    flux1 = spectrum.flux.loc[line1].copy()
    flux2 = spectrum.flux.loc[line2].copy()
    flux1.Flux = unp.uarray(flux1.Flux, flux1.Flux_stddev)
    flux2.Flux = unp.uarray(flux2.Flux, flux2.Flux_stddev)
    ratio = (flux1 / flux2).sort_index()
    ratio =ratio.join(spectrum.model.loc[line1]['Identifier'], how='inner')
    ratio.Flux_stddev = unp.std_devs(ratio.Flux)
    ratio.Flux_stddev = unp.std_devs(ratio.Flux)
    ratio.Flux = unp.nominal_values(ratio.Flux)
    # tmp = ratio.reset_index(level=0, drop=False)['Rows'].map(lambda x: int(x.split('-')[0]))
    # ratio['Rows'] = tmp.values
    #return ratio
    print ratio
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    component_wise_plot(ratio, pars=['Flux', 'Rows'], xerrbar=True)
    # plt.show()
    return ax


def view_doublet(spectrum, row, lines='S II'):
    model = spectrum.model.copy()
    if lines == 'S II':
        line1 = [l for l in model.index.levels[0]
                 if l.startswith('[S II]_6717')][0]
        line2 = [l for l in model.index.levels[0]
                 if l.startswith('[S II]_6731')][0]
    return


def fit_inspect(view, lines=None, row=44, ax=None, wavemode='wave'):
    import linephysics as ls
    reload(ls)
    import lmfit_wrapper as lw
    spectrum = view.Spectrum
    if ax is None:
        ax = plt.gca()
    if len(lines) == 2:
        wave, pd = ls.fit_doublet(
            view, lines=lines, pars_only=True, rows=[row]
        )
    elif len(lines) == 1:
        wave, pd = ls.fit_doublet(
            view, lines=[lines[0], lines[0]], pars_only=True, rows=[row]
        )
        pars = pd[row]['Pars']
        for k in pars.keys():
            if k.endswith('blue'):
                pars.pop(k)
            # EXPERIMENTAL
            elif k.endswith('red'):
                kk = '_'.join(k.split('_')[:-1])
                pars[kk] = pars[k]
                pars.pop(k)
            # ------------
    else:
        raise ValueError(
            "A list of either 1 or 2 valid transitions must be passed"
        )

    md = lw.build_model(pd[row]['Pars'], wave, data=pd[row]['Data'], model_only=True)
    # print pd[row]['Data'].max(), 'DATAMAX'
    ax.plot(wave, pd[row]['Data'], 'k-', drawstyle='steps-mid')
    # ax.plot(wave, md['data'], 'k-', drawstyle='steps-mid')
    for k in md.keys():
        if k in ['errs', 'wave']:
            continue
        if k == 'data':
            continue
        elif k == 'model':
            ax.plot(md['wave'], md[k], '-', color='orange', lw=1.5, label='model')
            #ax.plot(md['wave'], (pd[row]['Data'] - md[k]) / pd[row]['Errs'] - 3., 'b-')
            #ax.axhline(y=-4, color='b')
            #ax.axhline(y=-2, color='b')
        else:
            ax.plot(md['wave'], md[k], '--', label=k)
    ax.legend(frameon=False).draggable()
    #plt.show()
    return md


def show_residuals(
        wave, data, model, line, errs=None, ax=None, snr=False,
        divide_by_model=False, divide_by_data=True, vmin=None, vmax=None,
        halfwidth=20., contsubtract=True, cbar=True, cmap='RdGy',
        wavemode='lambda', viz=True, midline=None, lcol='k'):
    """ Shows 2D residual map of a model.
    """

    from helper_functions import wl_to_v

    try:
        from evaluate_model import evaluate_transition
    except ImportError:
        raise

    singlet = True
    # Set center of plot
    if midline is None:
        midline = model.loc[line, 'Line center'][0]
    else:
        singlet = False
    # print 'The thing:', midline

    ys = np.arange(1, data.shape[0] + 1) - .5
    # cenwave = model.loc[line, 'Line center'][0]
    widx = np.where((wave > midline - halfwidth) &
                    (wave < midline + halfwidth))[0]
    data = data[:, widx]
    wave = wave[widx]
    vels = wl_to_v(wave, midline)
    if errs is not None:
        errs = errs[:, widx]

    simdata, resids = evaluate_transition(
        data, wave, model.loc[line], cont_subtract=contsubtract, viz=False
    )
    if snr:
        resids /= (data / errs)
    elif divide_by_model:
        resids /= simdata
    elif divide_by_data:
        resids /= data
        #resids = np.absolute(resids)
    elif errs is not None:
        resids /= errs

    if vmax is None:
        vmax = np.median(resids) + 2 * resids.std()

    if vmin is None:
        vmin = - vmax

    if ('lambda'.startswith(wavemode.lower())) |\
            ('wavelength'.startswith(wavemode)):
        warr = wave
    elif 'velocity'.startswith(wavemode.lower()):
        warr = vels
    if viz:
        if ax is None:
            ax = plt.gca()
        ax.pcolormesh(warr, ys, resids, cmap=cmap, vmin=vmin, vmax=vmax,
                      edgecolor='face')
        ax.axis((warr.min(), warr.max(), ys.min(), ys.max()))
        if singlet:
            if 'velocity'.startswith(wavemode):
                ml = 0
            else:
                ml = midline
            ax.axvline(x=ml, linestyle=':', color=lcol)
        #ax.setI#
        if cbar:
            fig = plt.gcf()
            fig.colorbar(mpbl, orientation='horizontal')
    residdict = {'Data':data, 'Residuals':resids, 'Errors':errs,
                 'Wave':wave, 'Waveidx': widx, 'Velocity': vels,
                 'Model':simdata}
    return ax, residdict


def show_residuals_doublet(
        view, lines='O II', ax=None, vmin=-20, vmax=20, cbar=False,
        wavemode='lambda', halfwidth=20, cmap='RdGy', lcol='k',
        divide_by_model=False):
    """ `lines` can be 'S II', 'O II' or a list of two lines contained in the
    model index
    """

    from helper_functions import wl_to_v
    try:
        from evaluate_model import evaluate_transition
    except ImportError:
        raise

    if ax is None:
        ax = plt.gca()

    spectrum = view.Spectrum
    model = spectrum.model

    if lines == 'S II':
        lines  = [l for l in model.index.levels[0]
                  if l.startswith('[S II]_6717')]
        lines += [l for l in model.index.levels[0]
                  if l.startswith('[S II]_6731')]
    if lines == 'O II':
        lines  = [l for l in model.index.levels[0]
                  if l.startswith('[O II]_3727')]
        lines += [l for l in model.index.levels[0]
                  if l.startswith('[O II]_3729')]

    print lines
    lincen1 = model.loc[lines[0], 'Line center'][0]
    lincen2 = model.loc[lines[1], 'Line center'][0]
    midline = model.loc[[lines[0], lines[1]], 'Line center'].mean()
    waveidx = np.where((spectrum.wavl > midline - halfwidth) &
                       (spectrum.wavl < midline + halfwidth))[0]
    data = spectrum.data[:, waveidx]
    wave = spectrum.wavl[waveidx]
    vels = wl_to_v(wave, midline)
    if 'velocity'.startswith(wavemode):
        warr = vels
    else:
        warr = wave

    simdata, resids = evaluate_transition(
            data, wave, model.loc[lines[0]], cont_subtract=False, viz=False
    )
    simdata2, resids = evaluate_transition(
        resids, wave, model.loc[lines[1]], cont_subtract=True, viz=False
    )
    simdata += simdata2
    resids = data - simdata
    ax.pcolormesh(
       warr, np.arange(simdata.shape[0])+.5,
       resids / data,  #simdata, #.mean(),
       cmap=cmap, vmin=vmin, vmax=vmax,
    )
    if 'velocity'.startswith(wavemode):
        lincen1 = wl_to_v(lincen1, midline)
        lincen2 = wl_to_v(lincen2, midline)
    print lincen1, lincen2
    ax.axvline(lincen1, color=lcol, ls=':')
    ax.axvline(lincen2, color=lcol, ls=':')
    return ax, data, simdata#, residdict


def example_decomposition(target='ESO', UVview=None, VISview=None, row=42,
                          axes=None, legpos='center right'):
    ''' Target can be ESO, Haro11B, Haro11C
    '''
    import grism
    from helper_functions import wl_to_v
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    if target.lower() == 'eso':
        objname = 'ESO 338'
        UVfile = './Products/ESO338-rectified-UVB_fluxed.fits'
        VISfile = './Products/ESO338-rectified-VIS_fluxed.fits'
        modelfile = './ESO338Model.csv'
    elif target.lower() == 'haro11b':
        objname = 'Haro 11 B'
        UVfile = './Products/Haro11B-rectified-UVB_fluxed.fits'
        VISfile = './Products/Haro11-B-Halpha-fluxed.fits'
        modelfile = './Haro11BModel-fluxed.csv'
    elif target.lower() == 'haro11c':
        objname = 'Haro 11 C'
        UVfile = './Products/Haro11-C-rectified-UVB-fluxed.fits'
        VISfile = './Products/Haro11-C-rectified-Halpha-fluxed.fits'
        modelfile = './Haro11CModel-fluxed.csv'

    if UVview is None:
        UVspec = grism.load_2d(UVfile)
        UVspec.load_model(modelfile)
        UVview = grism.Show2DSpec(UVspec)
    else:
        UVspec = UVview.Spectrum

    if VISview is None:
        VISspec = grism.load_2d(VISfile)
        VISspec.load_model(modelfile)
        VISview = grism.Show2DSpec(VISspec)
    else:
        VISspec = VISview.Spectrum

    annloc = (.06, .88)
    row = str(row)
    row = row.split('-')[0]
    lines = ['O II', 'H_alpha', '[O III]_5007']
    OIIlines = [line for line in UVspec.model.index.levels[0]
                if line.startswith('[O II]_3727')]
    OIIlines += [line for line in UVspec.model.index.levels[0]
                 if line.startswith('[O II]_3729')]
    Haline = [l for l in UVspec.model.index.levels[0]
              if l.startswith('H_alpha')]
    OIIIline = [l for l in UVspec.model.index.levels[0]
                if l.startswith('[O III]_5007')]

    if axes is None:
        fig = plt.figure()#figsize=(3.5, 4.0))
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222, sharey=ax1)
        ax3 = plt.subplot(212, )
        cntnr = fig.add_subplot(
            111, frame_on=False, xticks=[], yticks=[],
            xticklabels=[], yticklabels=[],
        )
        cntnr.set_xlabel(
            u'Velocity [km s$^{-1}$]', fontsize=10, labelpad=15, family='serif'
        )
        cntnr.set_ylabel(
            u'Flux density [$10^{16}$ erg cm$^{-2}$ s$^{-1}$ Å $^{-1}$]',
            fontsize=10, labelpad=20, family='serif'
        )
    else:
        fig = plt.gcf()
        ax1, ax2, ax3 = axes[0], axes[1], axes[2]
    ax1.tick_params(labelbottom='off', labeltop='on')
    ax2.tick_params(labelleft='off', labeltop='on', labelbottom='off')

    # H_alpha
    out1 = fit_inspect(VISview, lines=Haline, ax=ax1, row=int(row))
    Hawave = float(Haline[0].split()[-1].replace('(', '').replace(')', ''))
    top1 = out1['model'].max()
    Haround = int(round(Hawave, -1))
    ax1.set_xticks([-400, 0, 400])
    ax1.annotate(r'\sffamily H$\alpha$', annloc, xycoords='axes fraction',
                 size=9)
    print 'H alpha plotted'
                        #bbox_to_anchor=[0, .0], )

    # O III 5007
    out = fit_inspect(UVview, lines=OIIIline, ax=ax2, row=int(row))
    OIIIwave = float(OIIIline[0].split()[-1].replace('(', '').replace(')', ''))
    OIIIround = int(round(OIIIwave, -1))
    top2 = out['model'].max()
    top = max(top1, top2)
    print('TOP: '+ str(top) + '\n')
    ax1.plot(out1['wave'], out1['data']-out1['model'] + top*1.15,
                     color='gray', zorder=1, drawstyle='steps-mid')
    ax1.plot([Hawave-8, Hawave+8], [top*1.15, top*1.15],
             color='gray', ls='--', lw=.6)
    ax2.plot(out['wave'], out['data']-out['model'] + top*1.15,
                     color='gray', zorder=1, drawstyle='steps-mid')
    ax2.plot([OIIIwave-8, OIIIwave+8], [top*1.15, top*1.15],
             color='gray', ls='--', lw=.6)
    ax2.set_xticks([-400, 0, 400])
    ax2.annotate('\sffamily [O III] 5007', annloc, xycoords='axes fraction',
                 size=9)
    print 'O III 5007 plotted'

    # O II 3726+29
    out = fit_inspect(UVview, lines=OIIlines, ax=ax3, row=int(row))
    OIIwave1 = float(OIIlines[0].split()[-1].replace('(', '').replace(')', ''))
    OIIwave2 = float(OIIlines[1].split()[-1].replace('(', '').replace(')', ''))
    OIIwave = np.array([OIIwave1, OIIwave2]).mean()
    top3 = out['model'].max()
    ax3.plot(out['wave'], out['data']-out['model'] + top3 * 1.15,
                     color='gray', zorder=1, drawstyle='steps-mid')
    ax3.plot([OIIwave-8, OIIwave+8], [top3*1.15, top3*1.15],
             color='gray', ls='--', lw=.6)
    ax3.annotate('\sffamily [O II] 3726+29', annloc, xycoords='axes fraction',
                 size=9)

    midwaves = [Hawave, OIIIwave, OIIwave]

    for j, ax in enumerate([ax1, ax2, ax3]):
        ax.get_legend().set_visible(False)
        if ax is ax1:
            spectrum = VISspec
        else:
            spectrum = UVspec
        #ax.set_ylim()
        legstuff = ax.get_legend_handles_labels()
        handles, labels = legstuff[0], legstuff[1]
        newlabels=[]
        for i, lab in enumerate(labels):
            if lab == 'model':
                handles[i].set_linewidth(.8)
                modeldata = handles[i].get_data()
                modwave = modeldata[0]
                modflux = modeldata[1]
                top = modflux.max()# * 1e16  # handles[i].get_data()[1].max() * 1e16
                continue
            lab = lab.split('_')[0]
            cnum = (ord(lab) - 97) % 12
            handles[i].set_color(Paired[cnum])
            handles[i].set_linewidth(1.0)
            handles[i].set_linestyle('-')
            newlabels.append(lab)
        for i, line in enumerate(ax.get_lines()):
            line.set_data(
                wl_to_v(line.get_data()[0], midwaves[j]),
                line.get_data()[1]# * 1.e16
            )
        ax.set_xlim(-600, 600)
        if ax is ax2:
            uselabels = newlabels
            usehandles = handles
        ax.set_ylim(-.1*top, 1.4*top)
        prettify_axes(ax)

    ax3.set_xlim(-900, 900)
    ax3.set_xticks([-800, -400, 0, 400, 800])
    leg = ax3.legend(
        usehandles, uselabels, loc=legpos, # title='\sffamily {} \n \sffamily row {}'.format(objname, row),
        title='\sffamily {}'.format(objname),#Row {}'.format(row), fancybox=True, shadow=True,
        frameon=False, fontsize=7, markerscale=1.2, labelspacing=0.2,
        handletextpad=.6, handlelength=2
    )
    leg.get_frame().set_lw(.6)
    if axes is None:
        fig.subplots_adjust(wspace=.02, hspace=.02, left=.14, right=.95, top=.95)
        #ax.label_outer()
    # fig.savefig('./Figures/example-decomp.pdf')
    # fig.savefig('./Figures/example-decomp.png', dpi=300)
    #plt.show()
    return out
