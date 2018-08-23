#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import scipy as sp
import scipy.constants as con
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
#            Convenience- and helper functions
# ============================================================================

def hello_there():
    os.system('notify-send -i ipython "IPython process done! "')
    os.system('play --no-show-progress --null --channels 1 synth {} sine {}'.format(.6, 550))


def _extract_1d(data, errs, LineLo, LineUp):
    """ For internal use (or is it?) """
    Lines = [LineLo - 1, LineUp]
    fitdata = data[Lines[0]:Lines[1], :]
    fiterrs = errs[Lines[0]:Lines[1], :] ** 2  # Using variances
    Length = fitdata.shape[1]
    lowerl, upperl = (Length * .2, Length * .8)
    profile = sp.absolute(fitdata[:, lowerl:upperl].sum(1))
    pronorm = profile.sum()
    profile /= pronorm
    profile = profile.reshape(-1, 1)
    weights = sp.tile(profile, fitdata.shape[1])
    data_weighted = fitdata * weights
    errs_weighted = fiterrs * weights ** 2
    data_1D = data_weighted.sum(0)
    errs_1D = sp.sqrt(errs_weighted.sum(0))
    return data_1D, errs_1D


def wl_to_v(wave, wl0):
    """ Converts a wavelength range and a reference wavelength to a
    velocity range.
    Velocities are in km/s.
    """
    # λ / λ0 = 1 + v / c =>
    #      v = (1 + λ / λ0) * c
    v = (wave / wl0 - 1.) * con.c / 1000.
    return v


def v_to_wl(v, wl0):
    """ Converts a velocity range and a reference wavelength to a
    wavelength range.
    Velocities are in km/s.
    """
    # λ / λ0 = 1 + v / c =>
    #      λ = λ0 * (1 + v / c)
    wave = wl0 * (1. + v / (con.c / 1000.))
    return wave


def v_to_deltawl(v, wl0):
    """ Gives delta-lambda as function of v and lambda-0.
    Velocities are in km/s
    """
    #  v = c Δλ / λ0 =>
    # Δλ = v λ0 / c
    delta_l = v * wl0 / (con.c / 1000.)
    return delta_l


def deltawl_to_v(deltawl, wl0):
    """ Gives delta-lambda as function of v and lambda-0.
    Velocities are in km/s
    """
    #  v = c Δλ / λ0 =>
    # Δλ = v λ0 / c
    v = (con.c / 1000.) * deltawl / wl0
    return v


def fwhm_to_sigma(fwhm):
    ''' FWHM must be of numpy.ndarray compatible type'''
    sigma = fwhm / (2. * np.sqrt(2. * np.log(2.)))
    return sigma


def sigma_to_fwhm(sigma):
    ''' sigma must be of numpy.ndarray compatible type'''
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    return fwhm


def air_to_vacuum(airwl, nouvconv=True):
    """
    Copied from Astropysics:
    http://packages.python.org/Astropysics/
    To not have this module as dependency as that is the only one necessary.

    Returns vacuum wavelength of the provided air wavelength array or scalar.
    Good to ~ .0005 angstroms.

    If nouvconv is True, does nothing for air wavelength < 2000 angstroms.

    Input must be in angstroms.

    Adapted from idlutils airtovac.pro, based on the IAU standard
    for conversion in Morton (1991 Ap.J. Suppl. 77, 119)
    """
    airwl = sp.array(airwl, copy=False, dtype=float, ndmin=1)
    isscal = airwl.shape == tuple()
    if isscal:
        airwl = airwl.ravel()
    # wavenumber squared
    sig2 = (1e4 / airwl) ** 2
    convfact = 1. + 6.4328e-5 + 2.94981e-2 / (146. - sig2) +  \
        2.5540e-4 / (41. - sig2)
    newwl = airwl.copy()
    if nouvconv:
        convmask = newwl >= 2000
        newwl[convmask] *= convfact[convmask]
    else:
        newwl[:] *= convfact
    return newwl[0] if isscal else newwl


def vacuum_to_air(vacwl, nouvconv=True):
    """
    Also copied from Astropysics, see air_to_vacuum() above.
    Returns air wavelength of the provided vacuum wavelength array or scalar.
    Good to ~ .0005 angstroms.

    If nouvconv is True, does nothing for air wavelength < 2000 angstroms.

    Input must be in angstroms.

    Adapted from idlutils vactoair.pro.
    """
    vacwl = np.array(vacwl, copy=False, dtype=float)
    isscal = vacwl.shape == tuple()
    if isscal:
        vacwl = vacwl.ravel()

    # wavenumber squared
    wave2 = vacwl*vacwl

    convfact = 1.0 + 2.735182e-4 + 131.4182/wave2 + 2.76249e8/(wave2*wave2)
    newwl = vacwl.copy()/convfact

    if nouvconv:
        # revert back those for which air < 2000
        noconvmask = newwl < 2000
        newwl[noconvmask] *= convfact[noconvmask]

    return newwl[0] if isscal else newwl


def _velspace_translate(wl, oldwl, newwl):
    return v_to_wl(wl_to_v(wl, oldwl), newwl)


def transition_from_existing(model, exist_trans, new_trans,
                             new_lincen, LSF=[0., 0.]):
    """ This function creates a new transition model using an existing one as
    template. This is practical for e.g. Balmer lines, in which the velocity
    distribution and widths should be more or less consistent for all
    transitions, since they come from same physical source.

    The arguments are:
    * model:        A pandas dataframe, containing at least the source model.
    * exist_trans:  A string with the name of the source transition. Should be
                    identical to the one under which it is stored in the model.
    * new_trans:    A string giving the desired name for the new transition.
    * new_lincen:   float or floatable giving the line center of the new
                    transition.
    """
    # TODO: Find a safer way to input the transition name string so it'll play
    # nice with the Transition() class. Maybe it's not even a problem.
    try:
        exist_trans in model.index.levels[0]
    except NameError:
        print("The transition {} was not found in the model"\
            .format(exist_trans))

    # Copy the original transition model with only two lower levels of indexing
    # (omitting the one containing the transition name)
    tmp = model.ix[exist_trans].copy()
    # To add new transition name to multiindex, we must first add it as
    # ordinary data column:
    tmp['Transition'] = new_trans
    # We then append this column as the bottom level of the multiindex object
    # (prepending is not possible)
    tmp = tmp.set_index('Transition', append=True)
    # Reorder the index levels, so they become Transition -> Row  -> Component.
    tmp.index = tmp.index.reorder_levels([2, 0, 1])
    # Index juggling done, now we adjust the wavelength offsets of the
    # centroids to match in velocity space rather than wavelength space,
    # because not doing this is wrong. Afterwards, convert velocities to
    # wavelength relaive to new center wavelength. Now the wavelengths are not
    # wrong anymore.
    thingie = model.ix[exist_trans]  # Debug
    wlspace = (
        model.ix[exist_trans]['Pos']
        + model.ix[exist_trans]['Line center']) \
        / model.ix[exist_trans]['Line center'] \
        * new_lincen\
        - new_lincen
    # wlwidth = (
    #     model.ix[exist_trans]['Sigma']
    #     + model.ix[exist_trans]['Line center']) \
    #     / model.ix[exist_trans]['Line center'] \
    #     * new_lincen\
    #     - new_lincen
    widthvel = wl_to_v(
        model.loc[exist_trans, 'Sigma'] +
        model.loc[exist_trans, 'Line center'],
        model.loc[exist_trans, 'Line center']
    )
    print(wlspace.loc['44-44'])
    print(widthvel.loc['44-44'])  # DEBUG
    wlwidth = v_to_deltawl(widthvel, new_lincen)
    print(wlwidth['44-44'])  # DEBUG
    tmp['Pos'] = wlspace.values
    tmp['Line center'] = new_lincen
    tmp['Sigma'] = wlwidth.values
    tmp.Sigma = np.sqrt(tmp.Sigma**2 - LSF[0]**2 + LSF[1]**2)  # "real" sigma.
    tmp.Sigma[tmp.Sigma < 0.1] = 0.1
    # tmp.Sigma = np.sqrt(tmp.Sigma**2 + LSF[1]**2)  # XXX Should this not be plus!?


    # Fit ranges should be --cleared-- copied in velspace:
    old_lincen = model.loc[exist_trans]['Line center'][0]
    if 'Fitranges' in tmp.columns:
        tmp.Fitranges = np.nan
        tmp.Fitranges = tmp.Fitranges.astype(object)
    # if "new" line already in model, overwrite rather than creating multiple
    # versions:
    if new_trans in model.index:
        print(new_trans)
        model = model.drop(new_trans, level=0)
    tmp = model.append(tmp)
    tmp = tmp.sort_index()
    print('Model of transition {} successfully created \n with {} as template'\
        .format(new_trans, exist_trans))
    return tmp


def set_model(spec, dataframe, transition, centerwl=None):
    """ Sets the model of a given Show2Dspec instance (and its parent
    Spectrum2D instance - mayaps the latter is actually enough?) and updates
    the relevant state variables to reflect this in the GUI.
    """
    # Should this respect or overwrite any existing model? Or have a boolean to
    # set (more work for me?)
    # Also, return new object or operate in-place?
    # Usage suggests this should be an instance method of Spectrum2D.
    spec.model = dataframe
    if centerwl is not None:
        centerwl = sp.float64(centerwl)
        spec.Center = centerwl
    if 'Dummy' in spec.model.index.levels[0].to_series().sum():
        dataframe = dataframe.drop('Dummy', level=0)
    return spec


def select_ranges(wave, data):
    mask = np.zeros_like(wave).astype(bool)

    def _onselect(xmin, xmax):
        mask[(wave > xmin) & (wave < xmax)] = True
        ax.axvspan(xmin, xmax, color='lightblue')
        plt.draw()
        return

    def _on_ok(event):
        idx = np.where(mask is True)
        fitwave = wave[idx]
        fitdata = data[idx]
        #fiterrs = errs[idx]
        plt.close(fig)
        return

    fig, ax = plt.subplots(1, 1)
    axfit = plt.axes([0.9, 0.84, 0.09, 0.06])
    fitit = Button(axfit, 'OK')
    fitit.on_clicked(_on_ok)
    ax.plot(wave.flatten(), data.flatten(), color='black', drawstyle='steps-mid')
    span = SpanSelector(ax, _onselect, 'horizontal')
    plt.show()
    return mask


def fit_with_sherpa(model, data, trans, rows,
                    ranges=[], errs=None, shmod=1, method='levmar'):
    """ This is probably going to be one of the slightly more complicated
    functions, so it'll probably need a relatively good and comprehensive
    docstring.

    Parameters
    -----------
    model : pandas DataFrame object.
        Must either have a 3-level index as created by grism, or at least a
        one-level index identifying the different components. If a three-level
        index is passed, it must either contain only one value of the
        Transition and Row values, or the 'trans' and 'row' kwargs must be
        passed in the call.
    data : numpy array or pandas Series or DataFrame.
        The data is assumed to have at least two columns containing first
        wavelength, then data. An optional third column can contain errors.
        If no such column exists, the errors are assumed to be 1.
    trans : string
        The desired value of the 'Transition' level of the model dataframe.
    rows : string
        The desired value of the 'Rows' level of the model index.
    shmod : integer
        Identifier for the Sherpa model in case one wants to have more models
        loaded in memory simultaneously. If not, it will be overwritten each
        time fit_with_sherpa() is run.
        Default: 1
    ranges : list of (min, max) tuples.
        The determines which wavelength ranges will be included in the fit. If
        an empty list is passed, the entire range of the data passed will be
        used.
        Default: [] (Empty list).
    method : string
        optimization to be used by Sherpa. Can be either 'levmar', 'neldermead'
        or 'moncar'. See Sherpa documentation for more detail.
        Default: 'levmar'.
    """
    # Should this perhaps be an instancemethod of one of the major classes
    # instead? On the upside it would mean direct access to all said class's
    # attributes, which is good because it means less mandatory input
    # parameters. On the downside, it is not generally useable. I want things
    # to be as general as possible. But not at any price. Cost/benifit analysis
    # not yet conclusive.

    # First of all, check if Sherpa is even installed.
    try:
        import sherpa.astro.ui as ai
        import sherpa.models as sm
    except ImportError:
        print(" ".join("The Sherpa fitting software must be installed to use \
            this functionality.".split()))
        raise

    # Sherpa isn't good at staying clean, need to help it.
    for i in ai.list_model_ids():
        ai.delete_model(shmod)
        ai.delete_data(shmod)
    # Load data first, 'cause Sherpa wants it so.
    if data.shape[0] == 2:
        ai.load_arrays(shmod, data[:, 0], data[:, 1])
    if data.shape[0] > 2:
        ai.load_arrays(shmod, data[:, 0], data[:, 1], data[:, 2])
    # Initialize model by setting continuum
    Contin = sm.Const1D('Contin')
    Contin.c0 = model.xs('Contin')['Ampl']
    ai.set_model(shmod, Contin)

    for i in model.index:
        if i == 'Contin':
            continue
        else:
            # use the identifier as letter (good idea?)
            name = model.ix[i]['Identifier']
            comp = ai.gauss1d(name)
            comp.ampl = model.ix[i]['Ampl']
            comp.pos = model.ix[i]['Pos']
            comp.fwhm = model.ix[i]['Sigma']
            ai.set_model(shmod, ai.get_model(shmod) + comp)
            ai.show_model(shmod)  # For testing...
    print('  ')
    print(ai.get_model(shmod))

    # Set ranges included in fit.
    # First, unset all.
    ai.ignore_id(shmod)
    if len(ranges) == 0:
        ai.notice_id(shmod)
    else:
        for r in ranges:
            ai.notice_id(shmod, r[0], r[1])

    # Set optimization algorithm
    ai.set_method(method)
    # Create copy of model
    new_model = model.copy()
    # Perform the fit:
    ai.fit(shmod)
    print(ai.get_fit_results())
    print(model)
    return new_model


def redshift_ned(name):
    """Queries NED for heliocentric redshift if possible.
    `name` can be any string understood by Astroquery's NED module.
    """
    try:
        from astroquery.ned import Ned
    except ImportError:
        raise
    try:
        z = Ned.query_object(name)['Redshift']
    except:
        raise ValueError('Object not found')

    return z


def flux(model, lines=[]):
    """ Calculates the fluxes of emission lines in a grism model dataframe.
    """
    uncert = False
    try:
        import uncertainties.unumpy as u
        uncert = True
    except ImportError:
        S = ''.join(["Could not import module 'uncertainties', \n",
                      "Uncertainties not computed."])
        raise ImportWarning(S)
    if type(lines) is not list:
        raise TypeError("Keyword argument 'lines' must be list-like.")
    df = model[['Ampl', 'Ampl_stddev',
                        'Pos', 'Pos_stddev',
                        'Sigma', 'Sigma_stddev']]
    df = df.sort_index()
    if len(lines) > 0:
        idx = model.index
        if set(idx.levels[0]).intersection(set(lines)) != set(lines):
            raise ValueError(
                "One or more input lines does not exist. \n Aborting.")
        df = df.loc[lines]
    if uncert:
        df['Ampl'] = u.uarray(df.Ampl, df.Ampl_stddev)
        df['Sigma'] = u.uarray(df.Sigma, df.Sigma_stddev)
        df['Pos'] = u.uarray(df.Pos, df.Pos_stddev)
    df['Flux'] = df.Sigma * df.Ampl * np.sqrt(2 * np.pi)
    if uncert:
        df['Flux_stddev'] = u.std_devs(df.Flux)
        df.Flux = u.nominal_values(df.Flux)
    if uncert:
        return df[['Flux', 'Flux_stddev']]
    else:
        return df['Flux']


def equivalent_width(wave, data, cont, wlstep=None):
    '''Calculates the equivalent width of a spectral line given a data set
    and a continuum. Works only for evenly spaced data.
    '''
    if wlstep is None:
        wlstep = np.hstack(wave[0], wave[:-1]) - \
            np.hstack(wave[1:], wave[-1])
    elif np.isscalar(wlstep):
        wlstep = wlstep
    return ((1. - data / cont) * wlstep).sum()


def EW_from_lmfit(model, wave, stddev=None, which='emit', MC=False):
    ''' Computes EW of a spectral line.
    `which` can be 'emit', 'absorb' or 'noabsorb'.
    `iter` is an integer, counts how many draws to sample the errors.
    '''
    num_iter = 1000 if MC else 1

    abscen = model.params['abs_center']
    absamp = model.params['abs_amplitude']
    abssig = model.params['abs_sigma']
    abscenerr = model.params['abs_center'].stderr
    absamperr = model.params['abs_amplitude'].stderr
    abssigerr = model.params['abs_sigma'].stderr
    if (np.array([abscenerr, absamperr, abssigerr]) == 0.).any():
        num_iter = 1
    data = model.data
    comps = model.eval_components()
    if which == 'absorb':
        data = comps['abs_']
    cont = comps['linear']  # model.eval_components()['linear']
    if which == 'emit':
        cont += comps['abs_']
    EW = {}
    for i in range(num_iter):
        if i > 0:
            data = data + np.random.normal(scale=model.weights**(-0.5))
            if which == 'emit':
                cont = comps['linear'] + model.eval_components(
                    abs_center = abscen + np.random.normal(scale=abscenerr),
                    abs_amplitude = absamp + np.random.normal(scale=absamperr),
                    abs_sigma = abssig + np.random.normal(scale=abssigerr)
                )['abs_']
        EW[i] = equivalent_width(wave, data, cont, wlstep=wave.ptp()/len(wave))
    EW = pd.DataFrame.from_dict(EW)
    return EW


def component_mean( spectrum, datacols=['Flux', 'Flux_stddev', 'Row'],
                   groupby='Identifier', mode='total'):
    ''' Takes a weighted average for all components in a given grism model.
    Thge returned DataFrame contains the specified column plus other useful
    columns like the averaged "Rown Number" (which is now a float).

    Parameters
    ----------
    model : pandas.DataFrame
        Model from `grism` Spectrum2D instance
    datacols : list
        List of strings containing the names of columns to average.
    groupy : str
        Name of index level to group by.
    weightcol : str
        String containing name of column to use as weights. If None, data is
        weighted evenly.

    Returns
    -------
    pandas.DataFrame
    '''
    import uncertainties.unumpy as unp
    modl = spectrum.model.copy().drop('Contin', level='Component')#[datacols]
    modl = modl.join(spectrum.flux)
    modl['Flux'] = unp.uarray(modl.Flux, modl.Flux_stddev)
    modl['Row'] = modl.index.get_level_values('Rows')\
        .map(lambda S: int(S.split('-')[0])).astype(np.float64)
    modl = modl[datacols+[groupby]]
    #modl = modl[['Flux']+[groupby]]
    modl = modl.set_index(groupby, append=True)
    # print modl.columns
    grouped = modl.groupby(level=['Transition', groupby])
    out = grouped.apply(flux_weighted_average, operate_on=datacols)#.aggregate(flux_weighted_average)
    out['Identifier'] = out.index.get_level_values('Identifier')
    return out  # grouped  #out  # grouped  #modl


def flux_weighted_average(grouped, operate_on=['Flux']):
    ''' Weightscol is a string containing col name for the weights
    '''
    import uncertainties.unumpy as unp
    weights = grouped.Flux.map(unp.nominal_values).values  # [weightscol]
    for col in grouped.columns:
        # print col  # DEBUG
        grouped[col] *= weights / weights.sum()
    return grouped.sum(0)

# =============================================================================
#     Import atom transition list taken from galaxy_lines.dat of Astropysics
#        fame. This is done at load time to be available module-wide, instead
#        of  per-instance (is there a better way to do that?)
#        Also, more extensive list is needed!
# =============================================================================

def load_lines_series():
    """ Loads the transitions list into a pandas.Series object. """
    # TODO: possibly extend to a DataFrame including oscillator strengths and
    # Gamma-factors so the voigt profile module can be incorporated into this
    # later?
    lines_srs = pd.read_table('galaxy_lines.dat', sep=',', index_col=0)
    lines_srs.index = lines_srs.index.map(str.strip)
    lines_srs.columns = lines_srs.columns.map(str.strip)
    lines_srs = lines_srs.apply(air_to_vacuum)
    print("List of lines loaded")
    return lines_srs


def bisquare(x):
    y = (1 - x**2)**2
    y[x < -1] = 0
    y[x > 1] = 0
    return y


def measure_fwhm(data, wave=None, cont=None, xtype='vel'):
    ''' 'Data' must be dict with keys 'data' and 'wave'
    '''
    import numpy as np
    # wave = data['wave']
    # data = data['data']
    if cont is None:
        cont = np.median(data)
    linedata = data - cont
    hiidx = np.where(linedata > linedata.max()/2.)
    fwhm = wave[hiidx].max() - wave[hiidx].min()
    return fwhm


def MC_it(function, data, errs, iters=1000, **kwargs):
    for j in range(iters):
        perturb = [np.random.normal(scale=errs.flatten()[i])
                   for i in range(len(errs.flatten()))]
        pertarr = np.array(perturb).reshape(errs.shape)
        indata = data + pertarr
        out = function(indata, **kwargs)
        if j == 0:
            outarr = out
        else:
            outarr = np.dstack((outarr, out))
    output = (outarr.mean(2), outarr.std(2))
    return output, outarr


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, np.sqrt(variance))
