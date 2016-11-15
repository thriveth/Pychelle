#!/usr/bin/env python
# -*- coding: utf-8 -*-

import grism
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from uncertainties import ufloat as ufl
from uncertainties import nominal_value, std_dev
from uncertainties.unumpy import uarray, nominal_values, std_devs
from helper_functions import v_to_deltawl, wl_to_v, _extract_1d
import lmfit_wrapper as lw
# reload(lw)

def fit_doublet(
    view=None, spectrum=None,  ampvar=None, sigvar=1., cenvar=1., report=True,
    method='leastsq', lines=None, freeze=['sigma'], tie=['pos'], viz=False,
    rows=None, pars_only=False):
    """
    Fits two lines as one model, to allow fitting a doublet.
    """
    if (view is None) & (spectrum is None):
        print 'Either spectrum of view must be given. Aborting.'
        return
    elif view is None:
        view = Show2DSpec(spectrum)
    elif spectrum is None:
        spectrum = view.Spectrum
    else:
        print('Redundant information given. Ignoring spectrum and using view.')
        spectrum = view.Spectrum
    v = view  # Laziness; less to type

    # Check that a model is loaded
    try:
        tmp = spectrum.model.columns[0]
    except:
        raise AttributeError("Input spectrum must have a model loaded")
    del tmp

    # If rows not explicitly set, set them
    if rows is None:
        rows = range(1, 92)
        therows = list(
            set(
                spectrum.model.index.get_level_values('Rows')
                .map(lambda x: int(x.split('-')[0]))
            )
        )
        print therows

    # set default doublet to fit; find the relevant indices and names in model
    # index
    if lines is None:
        lines = ['[O II]_3727', '[O II]_3729']  # Hack solution, works for now.

    newlines = []
    indices = []
    for line in lines:
        # Find the line names in index matching the input
        full_line_name = [\
            lin for lin in spectrum.model.index.levels[0]\
            if lin.startswith(line)]
        if len(full_line_name) != 1:
            raise ValueError("Line name match not unique")
        else:
            newlines += full_line_name
        # Now, find the actual DataFrame indices that define the dataframe
        b = [a for a in spectrum.model.index.tolist() if a[0].startswith(line)]
        indices += b
    # Check that we do indeed have two lines
    if len(newlines) != 2:
        raise ValueError("The number of lines must be exactly 2.")
    print "Now fitting lines {} and {}".format(newlines[0], newlines[1])
    # Combine components by identifier, not auto-assigned name.
    model = spectrum.model.copy()
    # First, temporarily assign the continuum an identifier
    for l in newlines:
        for row in range(1, 92):
            rownum_str = '{}-{}'.format(row, row)
            model.set_value((l, rownum_str, 'Contin'), 'Identifier', 'Contin')
    # Then, index by identifier instead of autogen'd component name
    model = model.set_index('Identifier', append=True)\
        .reset_index(level='Component')
    model = model.drop(
        ['Fitranges', 'Pos_lock', 'Sigma_lock', 'Ampl_lock',
         'RedChi2', 'Ampl_stddev', 'Pos_stddev', 'Sigma_stddev'], axis=1
    )
    # Now, combine the two lines TODO: Maybe find better suffixes?
    left_frame = model.loc[newlines[0]]
    right_frame = model.loc[newlines[1]]
    combined = left_frame.join(
        right_frame, how='inner', lsuffix='_blue', rsuffix='_red'
    )

    wave_offset = combined['Line center_red'][0] - \
        combined['Line center_blue'][0]

    parsdict = {}
    outdf_red = pd.DataFrame()
    outdf_blue = pd.DataFrame()
    red_params = [par for par in combined.columns if par.endswith('_red')]
    blue_params = [par for par in combined.columns if par.endswith('_blue')]

    # =========================================================================
    #   Now, cycle through rows, fit and append result to outoutput dataframe
    for therow in combined.index.levels[0]:
        if (int(therow.split('-')[0]) not in rows) and (rows is not None):
            continue
        print('\n \n Now fitting rows {} using method {} \n'.format(therow, method))
        v.LineLo = int(therow.split('-')[0])
        v.LineUp = int(therow.split('-')[1])
        # Extract data and errors to fit:
        data, errs = _extract_1d(v.data, v.errs, v.LineLo, v.LineUp)
        # Only use data from the ranges set in the model
        fran = spectrum.model.Fitranges.loc[(newlines[0], therow, 'Contin')]
        idxs = []
        for ran in fran:
            idxs += np.where(
                (spectrum.wavl > ran[0]) & (spectrum.wavl < ran[1]))[0].tolist()
        data = data[idxs]
        errs = errs[idxs]
        wave = spectrum.wavl[idxs]

        if viz:
            plt.plot(wave, data, 'k-', drawstyle='steps-mid')
            plt.plot(wave, errs, '-', color='green', drawstyle='steps-mid')

        # Set parameter limits
        tofit = combined.loc[therow].copy()
        tofit['AmpMax'] = data.max() * 2.
        tofit['AmpMin'] = -tofit.AmpMax * .1
        tofit['SigMin'] = 0.1
        tofit['SigMax'] = 15.
        tofit['WavMin'] = -cenvar  # 10
        tofit['WavMax'] = cenvar  # 10
        tofit.Pos_blue += tofit['Line center_blue']
        tofit.Pos_red += tofit['Line center_red']

        # Construct Parameter collection from DataFrame
        p = lw.lf.Parameters()
        p.clear()
        #return tofit  # XXX DEBUG
        for comp in tofit.index:
            if comp == 'Contin':
                varval = tofit.ix[comp]['Ampl_blue'] * 1.0001
                p.add('Contin_Ampl', value=varval, min=-10., max=10000.,)
                continue
            else:
                for col in tofit.columns:
                    if '_stddev' in col:
                        continue
                    name = comp + '_' + col
                    value = tofit.loc[comp][col]
                    if col.startswith('Pos'):
                        color = col.split('_')[1]
                        # varmin = tofit.loc[comp]['Line center'+'_'+color] + \
                        #     tofit.loc[comp]['WavMin']  # value - 10
                        # varmax = tofit.loc[comp]['Line center'+'_'+color] + \
                        #     tofit.loc[comp]['WavMax']  # value + 10
                        varmin = tofit.loc[comp][col] + \
                            tofit.loc[comp]['WavMin']  # value - 10
                        varmax = tofit.loc[comp][col] + \
                            tofit.loc[comp]['WavMax']  # value + 10
                        vary = 'pos' not in freeze
                    elif col.startswith('Sigma'):
                        varmin = tofit.loc[comp]['SigMin']
                        varmax = tofit.loc[comp]['SigMax']
                        vary = 'sigma' not in freeze
                    elif col.startswith('Ampl'):
                        varmin = 1e-50#0.001  # tofit.loc[comp]['AmpMin']
                        varmax = tofit.loc[comp]['AmpMax']
                        vary = 'ampl' not in freeze
                    else:
                        continue
                    p.add(name, value, min=varmin, max=varmax, vary=vary)

        # Now set parameters that are tied to others by mathematical
        # expressions.
        exprdict = {}
        for i, comp in enumerate(tofit.drop('Contin').index):
            if i == 0:
                amp1 = tofit.loc[comp].Ampl_blue
                sig1 = tofit.loc[comp].Sigma_blue
                wl1 = tofit.loc[comp].Pos_blue
                refname = comp
            else:
                coeff = tofit.loc[comp]['Ampl_blue'] / amp1
                posdiff = tofit.loc[comp]['Pos_blue'] - wl1
                sigcoef = tofit.loc[comp]['Sigma_blue'] / sig1
                ampl_expr = '{}_Ampl_blue * {}'.format(refname, coeff)
                pos_expr = '{}_Pos_blue + {}'.format(refname, posdiff)
                sig_expr = '{}_Sigma_blue * {}'.format(refname, sigcoef)
                if 'ampl' in tie:
                    exprdict[comp+'_Ampl_blue'] = ampl_expr
                if 'pos' in tie:
                    exprdict[comp+'_Pos_blue'] = pos_expr
                if 'sigma' in tie:
                    exprdict[comp+'_Sigma_blue'] = sig_expr
            exprdict[comp+'_Pos_red'] = '{}_Pos_blue + {}'.format(
                comp, wave_offset)
            exprdict[comp+'_Sigma_red'] = '{}_Sigma_blue'.format(comp)
            # exprdict[comp+'_Ampl_red'] = comp

        for key in exprdict.keys():
            com = p[key]
            com.set(expr=exprdict[key])

        if pars_only:
            parsdict[v.LineLo] = {'Pars':p, 'Data':data}
            continue

        # Now: action! Fitting happens here.
        result = lw.fit_it(p, [wave, data, errs])
        if report:
            lw.lf.report_fit(result)
        if viz:
            model = lw.build_model(p, wave)  # return model array
            plt.plot(wave, model, color='orange')

        # Now the rather tedious task of separating red and blue parameters and
        # converting them back to DataFrames with the right format to fit
        # seamlessly back into the spectrum model
        red_keys = [k for k in result.params.keys() if k.endswith('_red')]
        blue_keys = [k for k in result.params.keys() if k.endswith('_blue')]
        par_red = result.params.copy()
        for key in par_red.keys():
            if key.endswith('_blue'): par_red.pop(key)
            if key.endswith('_red'): result.params.pop(key)
        blue_result = lw.params_to_grism(result, output_format='df')
        blue_result.index.names = ['Identifier']
        result.params = par_red
        red_result = lw.params_to_grism(result, output_format='df')
        red_result.index.names = ['Identifier']
        blue_result['Transition'] = newlines[0]
        blue_result['Rows'] = therow
        blue_result = blue_result.set_index('Rows', append=True).set_index('Transition', append=True)
        red_result['Transition'] = newlines[1]
        red_result['Rows'] = therow
        red_result = red_result.set_index('Rows', append=True).set_index('Transition', append=True)
        blue_result.index = blue_result.index.reorder_levels(
            ['Transition', 'Rows', 'Identifier']
        )
        red_result.index = red_result.index.reorder_levels(
            ['Transition', 'Rows', 'Identifier']
        )
        comp = combined.Component_red.loc[therow]
        comp.columns = ['Component']
        red_result.set_index(comp, append=True, inplace=True)
        red_result.reset_index('Identifier', inplace=True)
        red_result.index.names=['Transition', 'Rows', 'Component']
        blue_result.set_index(comp, append=True, inplace=True)
        blue_result.reset_index('Identifier', inplace=True)
        blue_result.index.names=['Transition', 'Rows', 'Component']
        outdf_blue = outdf_blue.append(blue_result)
        outdf_red = outdf_red.append(red_result)

    if pars_only:
        return wave, parsdict

    outdf_blue.loc[outdf_blue.Identifier == 'Contin', 'Identifier'] = np.nan
    outdf_red.loc[outdf_red.Identifier == 'Contin', 'Identifier'] = np.nan
    outdf_blue.Pos -= tofit['Line center_blue'][0]
    outdf_red.Pos -= tofit['Line center_red'][0]
    spectrum.model.update(outdf_blue)
    spectrum.model.update(outdf_red)
    spectrum.model.loc[
        spectrum.model.index.get_level_values('Component') == 'Contin',
        'Identifier'] == np.nan  # loc[]Identifier.isnull(), 'Identifier'] = 'x'
    plt.show()
    return  result, outdf_blue, outdf_red


def fit_OIII(view=None, spectrum=None,  ampvar=None, sigvar=.2,
             cenvar=None, freeze=[], tie=[], method='leastsq', rows='all'):
    """ Fitting OIII with Halpha as a template requires some but not too much
    wiggle allowed in position as well as line width.

    Parameters:
    -----------

    view : grism.show2dspec.Show2DSpec
        Must have an [O III] line selected and ready for fitting.
    spectrum : grism.spectrum2d.Spectrum2D
        The spetrum to analyze. Either a view or spectrum must be given,
        if both are given, will use the view.
    ampvar, sigvar, cenvar : float
       The allowed "wiggle" for the three fit parameters. units of `sigvar`
       should be a multiplicative constant, `cenvar` should be given in km/s
       for now.
    freeze, tie : list
        Lists of strings of parameters that should be frozen (tied).
        Valid values are 'ampl', 'sigma', 'pos'.
    method : str
        Minimizatuion method to be passed on to lmfit.
    rows : str or list
        Valid string: 'all'. Otherwise, must be a list of integers.
    """
    if (view is None) & (spectrum is None):
        print 'Either spectrum of view must be given. Aborting.'
        return
    elif view is None:
        view = Show2DSpec(spectrum)
    elif spectrum is None:
        spectrum = view.Spectrum
    else:
        print('Redundant information given. Ignoring spectrum and using view.')
        spectrum = view.Spectrum

    if rows == 'all':
        rows = np.arange(spectrum.data.shape[0]) + 1

    transit = spectrum.transition
    if not '[O III]' in transit:
        print "This function is for [O III] only."
        print('Consider yourself warned')
        #return
    v = view

    if cenvar is None:
        cenvar = v_to_deltawl(15, spectrum.Center)
    else:
        cenvar = v_to_deltawl(cenvar, spectrum.Center)

    for s in v.model.index.levels[1]:
        nums = s.split('-')
        if not int(nums[0]) in rows:
            continue
        if len(v.model.loc[(spectrum.transition, s)].drop('Contin')) < 1:
            continue
        v.LineLo = int(nums[0])
        v.LineUp = int(nums[1])
        # if not v.LineLo in rows:
        #     continue
        print('Now fitting rows {} using method {}'.format(s, method))
        lp = v.prepare_modeling()
        lp.create_fit_param_frame()

        ### Set special limits for OIII as templated with Halpha
        wave_wiggle = v_to_deltawl(15., lp.line_center)
        lp.tofit['WavMin'] = lp.tofit.Pos - wave_wiggle
        lp.tofit['WavMax'] = lp.tofit.Pos + wave_wiggle
        lp.tofit['SigMin'] = lp.tofit.Sigma - lp.tofit.Sigma * sigvar
        lp.tofit.SigMin[lp.tofit.SigMin < 0.1] = 0.1
        lp.tofit['SigMax'] = lp.tofit.Sigma + lp.tofit.Sigma * sigvar
        lp.tofit.SigMax[lp.tofit.SigMax < 0.1] = 0.1 + lp.tofit.Sigma * sigvar
        # print(lp.tofit)

        lp.load_parameters_to_fitter()

        # print(lp.params)

        exprdict = {}

        ###   Generate mathematical expressions, set these, if 'tie' passed.
        ###                           ------------
        for i, compo in enumerate(lp.tofit.drop('Contin').index):
            for f in freeze:
                lp.params[compo+'_{}'.format(f.capitalize())].set(vary=False)
            if i == 0:
                amp1 = lp.tofit.loc[compo]['Ampl']
                wl1 = lp.tofit.loc[compo]['Pos']
                sig1 = lp.tofit.loc[compo]['Sigma']
                refname = compo
            else:
                coeff = lp.tofit.loc[compo]['Ampl'] / amp1
                posdiff = lp.tofit.loc[compo]['Pos'] - wl1
                sigcoef = lp.tofit.loc[compo]['Sigma'] / sig1
                ampl_expr = '{}_Ampl * {}'.format(refname, coeff)
                pos_expr = '{}_Pos + {}'.format(refname, posdiff)
                sig_expr = '{}_Sigma * {}'.format(refname, sigcoef)

                if 'ampl' in tie:
                    exprdict[compo+'_Ampl'] = ampl_expr
                if 'pos' in tie:
                    exprdict[compo+'_Pos'] = pos_expr
                if 'sigma' in tie:
                    exprdict[compo+'_Sigma'] = sig_expr

        # print exprdict
        for key in exprdict.keys():
            com = lp.params[key]
            com.set(expr=exprdict[key])
        # print lp.params
        v.lp = lp
        print 'Now fitting rows: {}'.format(s)
        v.lp.fit_with_lmfit(method=method, conf='conf')
        v.process_results()
        # v._build_model_plot()  #  Does this slow things down?
        print(
            'Succesfully fitted rows {} using method {}\n \n'.format(
                s, method))

    # Don't alter any data in-place
    # transframe = spectrum.model.loc[transition].copy()
    outframe = lp.output
    v._build_model_plot()
    return outframe


def EBV_calzetti(HaHb):
    ''' Computes E(B-V) based on the method of Domínguez et al. 2012,
    ArXiv 1206.1867, assuming a reddening curve from Calzetti(2000).
    (NB! Explanation in Atek 2012 is good!)

    With Calzetti 2000 values of k(alpha) and k(beta), we get:

        E(B-V) = 1.97 log10((Ha/Hb) / 2.86)
    '''
    EBV = 1.97* unp.log10(HaHb / 2.86)
    return EBV


def O3N2(OIII5007, NII6583, Halpha, Hbeta):
    ''' Calculate O3N3 from Yin et al. 2007:

    O3N2 = log[([O iii]λ5007/Hβ)/([N ii] λ6583/Hα)]
    '''
    O3N2 = unp.log10((OIII5007 / Hbeta) /
                    (NII6583 / Halpha))
    return O3N2


def R23(OII3726_29, OIII4959, OIII5007, Hbeta):
    ''' Compute R23 from Pilyugin 2000 as referred in Yin 2007:

    R23= ( [OII 3727+3729] + [OIII 4960+5008] ) / Hb
    '''
    return (OII3726_29 + OIII4959 + OIII5007) / Hbeta


## XXX FIXME This function gives strange values when called from Jupyter
# Notebook. How can this be? Exact same procedure ran diretctly in the notebook
# seems to give correct values! Very strange! Invesigate! Until then, do not
# use this function!
def P(OIII4959, OIII5007, OII372629):
    ''' Compute  the ionization parameter P from Pilyugin 2001(a)

    P = (I_4959 + I_5007) / (I_4959 + I_5007 + I_3726 + I 3729)
    '''
    P = (OIII4959 + OIII5007) / (OIII4959 + OIII4959 + OII372629)
    return P


def N2(NII6583, Halpha):
    ''' Compute logarithmic N2 ratio.
    '''
    return unp.log10(NII6583 / Halpha)


def OH_from_N2P(N2, P):
    ''' Calculate O/H by the P-method as described in Yin 2007:
    '''
    N2, P = N2.values, P.values
    OH = np.zeros_like(P)
    for i, p in enumerate(P):
        if P[i] < .65:
            ohn2p = 9.332 + 0.998 * N2[i]
        elif P[i] < .8:
            ohn2p = 9.457 + 0.976 * N2[i]
        else:
            ohn2p = 9.514 + 0.916 * N2[i]
        OH[i] = ohn2p
    return OH


def OH_from_O3N2(O3N2):
    ''' Computes Oxygen abundance from O3N2 ratio.

    Yin et al. 2007:
    12 + log(O/H) = 8.203 + 0.630 × O3N2 − 0.327 × O3N2², (eq.10);
    '''
    return 8.203 + 0.630 * O3N2 - 0.327 * O3N2 ** 2


def OH_from_R23(R23):
    ''' Oxygen abundance from  R23 (Yin 2007).
    This calibration is suitable for R23 between 0.4 and 1, and for
    abundances between 7.0 and 7.9
    '''
    return 6.486 + 1.401 * unp.log10(R23)


def TeO3(spectrum, ebv=0.1, mode='components'):
    """ Mode can be components, total"""
    from pyastrolib.astro import ccm_unred
    flux = spectrum.flux
    flux = flux.drop('Contin', level='Component')
    flux.sort_index(inplace=True)
    model = spectrum.model
    model = model.drop('Contin', level='Component')
    model.sort_index(inplace=True)
    flux['Flur'] = ccm_unred(model['Line center'], flux.Flux, ebv)

    Line4363 = [l for l in model.index.levels[0] if l.startswith('[O III]_4363')][0]
    Line4959 = [l for l in model.index.levels[0] if l.startswith('[O III]_4959')][0]
    Line5007 = [l for l in model.index.levels[0] if l.startswith('[O III]_5007')][0]

    M4363 = model.loc[Line4363]
    M4959 = model.loc[Line4959]
    M5007 = model.loc[Line5007]

    I4363 = flux.loc[Line4363]
    I4959 = flux.loc[Line4959]
    I5007 = flux.loc[Line5007]

    if mode == 'total':
        I4363 = I4363.drop('Contin', level=1).groupby(level=0).sum()
        I4959 = I4959.drop('Contin', level=1).groupby(level=0).sum()
        I5007 = I5007.drop('Contin', level=1).groupby(level=0).sum()

    # TODO: Is this iteration correct
    Te = pd.DataFrame(
        3.297 / np.log(
            ((I4959.Flur + I5007.Flur) * 1**0.05) / (I4363.Flur * 7.76))# * 1e4
    )#, columns=['Te'])
    Te['Te'] = pd.DataFrame(
        3.297 / np.log(
            ((I4959.Flur + I5007.Flur) * Te.Flur**0.05) / (I4363.Flur * 7.76))
    ) * 1e4

    Te['Row'] = Te.index.get_level_values('Rows')\
        .map(lambda s: int(s.split('-')[0]))
    if mode != 'total':
        Te['Identifier'] = M4363.Identifier
    else:
        Te['Identifier'] = 'x'
    return Te# , (I5007 + I4959) / I4363


def Te_O3(OIII4363, OIII4959, OIII5007, Hbeta):
    ''' Computing T_e from [O III] λλ 4363, 4959, 5007,
    using the method from Mas-Hesse's code (any better options out there?)
    '''
    Te = pd.DataFrame(OIII4363.copy())
    OIII4363 /= Hbeta
    OIII4959 /= Hbeta
    OIII5007 /= Hbeta
    # First time
    # Te['Te4'] = 3.297 / (
    #     ((OIII4959 + OIII5007) * 1 ** 0.05) / (OIII4363 * 7.76)
    # )  # DEBUG
    Te['Te4'] = 3.297 / unp.log(
        ((OIII4959 + OIII5007) * 1 ** 0.05) / (OIII4363 * 7.76)
    )
    Te.loc[(Te['Te4'] < 0.), 'Te4'] = np.nan
    Te['Te4'] = 3.297 / unp.log(
        ((OIII4959 + OIII5007) * Te['Te4'] ** 0.05) / (OIII4363 * 7.76)
    )
    Te['Te'] = Te['Te4'] * 1.e4
    #return Te
    Te = Te.drop('OIII4363', axis=1)
    #print Te #Te.to_string()#.head()
    return Te


def OH_from_TeO3(fluxes, Te=None):
    ''' Again, translated from Miguel's code.
    '''
    if Te is None:
        Te = Te_O3(fluxes['OIII4363'], fluxes['OIII4959'],
                fluxes['OIII5007'], fluxes['Hbeta'])
    if not 'Te4' in Te.columns:
        Te['Te4'] = Te['Te'] * 1.e4
    Oplus = 7.34e-7 * (fluxes['OII3726'] + fluxes['OII3729'])/fluxes['Hbeta']\
        * unp.exp(3.9/Te['Te4'])
    O2plus = (1.3e-6 / Te['Te4']**0.38) * unp.exp(2.9/Te['Te4'])\
        * 4.2 * fluxes['OIII4959']/fluxes['Hbeta']
    return 12. + unp.log10(Oplus + O2plus)


def density_from_OII_OIII(spectrum):
    try:
        import pyneb as pn
    except:
        print('PyNeb must be installed to run this function')
        return

    flux = spectrum.flux
    flux['UrFlux'] = ccm_unred(flux['Line center'])

    return


def abundance_Te_OIII(spectrum, ebv, n=1e0, mode='components'):
    """ Input frame of Te's computed from OIII lines,
    output similar frame of abundances
    """
    # TODO Get formulas from Miguel's code.
    OII26 = [l for l in spectrum.model.index.levels[0]
            if l.startswith('[O II]_3727') ][0]
    OII29 = [l for l in spectrum.model.index.levels[0]
            if l.startswith('[O II]_3729') ][0]
    I3726 = spectrum.flux.loc[OII26]
    #I3726.Flux = uarray(I3726.Flux.values, I3726.Flux_stddev.values)
    I3729 = spectrum.flux.loc[OII29]
    #I3729.Flux = uarray(I3729.Flux.values, I3729.Flux_stddev.values)
    Te = TeO3(spectrum, ebv, mode=mode)
    print I3729
    #print Te
    Te['OII'] = 7.34e-7 * (I3726.Flux + I3729.Flux) * np.exp(3.9 / Te['Te'])
    Te['OIII']
    return Te


def component_wise_fit_LLE(inframe, column, frac=0.2, method='lowess'):
    '''    'method' can be 'lowess' or 'LLE'.
    '''
    from statsmodels.nonparametric.smoothers_lowess import lowess
    frame = inframe.swaplevel('Row', 'Identifier')
    frame['Identifier'] = frame['Identifier'].map(ord) - 97
    for comp in frame.index.levels[0]:
        cframe = frame.loc[[comp]]#.dropna(subset=[column])
        print 'Component: ', comp
        cnum = cframe['Identifier'][0] % 12
        rows = cframe.index.get_level_values('Row')
        if method == 'LLE':
            LLE = KernelReg(frame.loc[comp][column], rows, 'c', bw='cv_ls')
            means, mfx = LLE.fit()
        elif method == 'lowess':
            LLE = lowess(cframe[column], rows, it=10, missing='none',
                         frac=frac)
            means = LLE[:,1]
        frame.loc[[comp], column+'_means'] = means
        plt.plot(frame.loc[comp][column], rows, 's',
                 color=Paired.hex_colors[cnum], zorder=1)
        plt.plot(means, rows, '-', lw=3, color=Paired.hex_colors[cnum],
                 zorder=2)
    frame[column+'_LLEresids'] = frame[column] - frame[column+'_means']
    return frame[[column, column+'_means', column+'_LLEresids']]


def robust_stddev(inframe, column, sigmas=3):
    '''expects a frame generated by component_wise_FIT_LLE

    '''
    frame = inframe.swaplevel('Row', 'Identifier')
    outframe = frame.groupby(level='Identifier').std()
    for comp in frame.index.levels[0]:
        num_outliers = 1
        itercount = 0
        cframe = frame.loc[comp].copy()
        Stddev = outframe.loc[comp][column+'_LLEresids']
        while num_outliers > 0:
            idx = np.where(cframe[column+'_LLEresids'].abs() > 3 * Stddev)[0]
            cframe[cframe[column+'_LLEresids'].abs() > sigmas * Stddev] = np.nan
            Stddev = cframe[column+'_LLEresids'].std()
            num_outliers = len(idx)
            itercount += 1
            if itercount > 3:
                break
        outframe.set_value(comp, column+'_stddev', Stddev)
        print outframe
    return  outframe[[column+'_stddev']]
