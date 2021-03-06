#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" A wrapper module to use lmfit as a backend for fitting a grism model.
So far, it works for single-row(-set), single transition only.
Maybe more sophisticated ways to do it can be implemented later.

Functions
---------
load_params()
    Takes grism-generated DataFrame as input, loads values into an lmfit
    Parameters() object.
build_model()
    The mathematical function to be minimized.
    This function is not necessary to call manually when fitting, but can be
    practical to call for generating a plot of the model for a given set of
    parameters.
fit_it
    Takes as minimum a Parameters() set and a wavelength array x as arguments,
"""

import scipy as sp
import pandas as pd

try:
    import lmfit as lf
except ImportError:
    print 'Could not find working lmfit installation.'
    raise


def load_params(indf):
    """ Takes a pandas.DataFrame generated by grism and reads it into an
    lmfit.Parameters object.
    """
    df = indf.copy()
    p = lf.Parameters()
    p.clear()
    # print('DEBUG LMFITWRAPPER: POS', df.Pos)
    df.set_value('Contin', 'Identifier', 'x')
    if not 'SigMin' in df.columns:
        df['SigMin'] =  0.1
    if not 'SigMax' in df.columns:
        df['SigMax'] = 30.
    if not 'AmpMin' in df.columns:
        df['AmpMin'] = 0.001
    if not 'AmpMax' in df.columns:
        df['AmpMax'] = 5000.
    if not 'WavMin' in df.columns:
        df['WavMin'] = df.Pos - 10.
    if not 'WavMax' in df.columns:
        df['WavMax'] = df.Pos + 10.
    # print('DEBUG LMFITWRAPPER: POS', df.Pos)
    # print('DEBUG LMFITWRAPPER: WAVMIN', df.WavMin)
    df.Pos += df['Line center']

    # print df
    for comp in df.index:
        if comp == 'Contin':
            varval = df.ix[comp]['Ampl'] + .0001
            p.add('Contin_Ampl', value=varval, min=-10., max=10000.,
                  vary=sp.invert(df.loc[comp]['Lock']))
            continue
        else:
            for col in df.columns:
                if '_stddev' in col:
                    continue
                if col in ['Ampl', 'Sigma', 'Pos', 'Gamma',]:
                    #name = df.ix[comp]['Identifier']+'_'+col
                    name = comp + '_' + col
                    value = df.ix[comp][col]
                    if col == 'Pos':
                        varmin = df.loc[comp]['Line center'] + \
                            df.loc[comp]['WavMin']  # value - 10
                        varmax = df.loc[comp]['Line center'] + \
                            df.loc[comp]['WavMax']  # value + 10
                        vary = sp.invert(df.loc[comp]['Lock'][0])
                    elif col == 'Sigma':
                        varmin = df.loc[comp]['SigMin']
                        varmax = df.loc[comp]['SigMax']
                        vary = sp.invert(df.loc[comp]['Lock'][1])
                    elif col == 'Ampl':
                        varmin = 0.001  # df.loc[comp]['AmpMin']
                        varmax = df.loc[comp]['AmpMax']
                        vary = sp.invert(df.loc[comp]['Lock'][2])
                    else:
                        varmin = None
                        varmax = None
                    p.add(name, value, min=varmin, max=varmax, vary=vary)
    return p


def build_model(params, x, data=None, error=None, model_only=False):
    """ Dynamically defines the model from a lmfit.Parameters() instance.
    Then calculates the residuals by subtracting the data and model.
    This is the function that will be minimized by minimize()
    Returns either:
        * A model generated by an input Parameters() object, with no attempt
        being made at minimizing if either data nor errors are provided, OR
        * An unweighted set of residuals (data - value) if no errors are
        provided, OR
        * A inverse-variances-weighted set of residuals if errors are provided.
    """
    # Because parameters cannot be hierarcically grouped, we need to create a
    # list of components and then cycle through that.
    modeldict = {'wave': x, 'data': data, 'errs': error}
    IDs_done = []
    col_done = []
    for par in params:
        parsplit = par.split('_')
        iden = parsplit[0]
        kind = parsplit[1]
        if len(parsplit) > 2:
            color = parsplit[2]
            if not '_' + color in col_done:
                col_done.append('_' + color)
        if not iden in IDs_done:
            IDs_done.append(iden)
    if len(col_done) == 0:
        col_done.append('')

    model = sp.zeros_like(x)
    model += params['Contin_Ampl'].value
    # print col_done # DEBUG

    for comp in IDs_done:
        if comp.startswith('Contin'):
            continue
        else:
            comp = comp.split('_')[0]
            for col in col_done:
                ampl = params[comp + '_Ampl' + col].value
                pos = params[comp + '_Pos' + col].value
                sigma = params[comp + '_Sigma' + col].value
                line = sp.stats.norm.pdf(x, pos, sigma) \
                        * sigma * sp.sqrt(2 * sp.pi) * ampl
                model += line
                modeldict[comp+col] = line

    modeldict['model'] = model
    if model_only:
        return modeldict
    if data is None:
        return model
    elif error is None:
        return (model - data)
    else:
        return sp.sqrt((model - data) ** 2. / error ** 2.)


def fit_it(params, args, method='leastsq', conf='covar', kwargs=None):
    """ Carries out the fit.

    Parameters
    ----------
    params : lmfit.Parameters() instance
        Call load_params to generate.
    args : tuple
        Arguments to pass to the function to minimize. Must contain a
        wavelength array as first argument, optional second and third argument
        will be interpreted as data and errors, respectively.
        arrays are optional.
    kwargs : tuple
        keyword arguments, will be passed directly to the lmfit.minimize()
        function. See lmfit docs for options.

    Returns
    -------
    result : lmfit.Minimizer() object
    """
    # For testing:
    x = args[0]
    data = args[1]
    errs = args[2]
    earlymodel = build_model(params, x)
    # Now: fitting.
    result1 = lf.minimize(
        build_model, params, args=args, method=method
    )
    outresult = result1
    if method != 'leastsq':
        result2 = lf.minimize(
            build_model, result1.params, args=args, method='leastsq'
        )
        outresult = result2
    if not outresult.errorbars:
        result3 = lf.minimize(
            build_model, outresult.params, args=args, method='leastsq'
        )
        outresult = result3
    if conf == 'conf':
        if (outresult.leastsq()) and (not outresult.errorbars):
            lf.conf_interval()
    return outresult


def params_to_grism(result, output_format='dict'):
    """ Reads lmfit Params() object back into grism-readable formats.
    Parameters
    ----------
    output_format : str
        Can either be 'dict' (default), which is read into the line profile
        editor, or 'df', a pandas dataframe which can be inserted back into a
        model dataframe by whatever means one wants.
    """
    params = result.params
    tempdf = pd.DataFrame()

    for thekey in params:
        compinfo = thekey.split('_')
        compo = compinfo[0]
        param = compinfo[1]
        value = params[thekey].value
        stddev = params[thekey].stderr
        tempdf = tempdf.set_value(compo, param, value)
        tempdf = tempdf.set_value(compo, '{}_stddev'.format(param), stddev)

    #Include reduced chi square:
    tempdf = tempdf.set_value('Contin', 'RedChi2', result.redchi)

    if output_format == 'dict':
        tempdf = tempdf.T.to_dict()

    return tempdf

