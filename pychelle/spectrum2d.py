#!/usr/bin/env python
# encoding: utf-8

import scipy as sp
import numpy as np
import pandas as pd
import ast
from traits.api import Array, Float, Dict, List, Range, Any, HasTraits, Str
from .helper_functions import air_to_vacuum, redshift_ned
from .helper_functions import flux as fluxes


class Spectrum2D(HasTraits):
    """Object that handles the 2D Spectrum data, creates other relevant
    objects like a wavelength array, and also keeps track of the
    guessed and fitted models as they are created. The latter in the
    shape of Dicts of instances of the Transistion() class and some
    similar container for the fitting output.
    The class takes as input a data-array, an error array and a FITS header. If
    wanting to open from a FITS file directly, use the load_2d() helper
    function.
    """
    # TODO: Maybe restructure for more flexibility, so e.g. a FITS-header is
    #       not mandatory and so that wavelength array is given in the helper
    #       function rather than in the constructor module of this class.
    objname = Str()
    wavl = Array()
    instrument_profile = Array()
    LSF = Array()
    wavlmin = Float()
    wavlmid = Float()
    wavlmax = Float()
    data = Array()
    errs = Array()
    ModelsDict = Dict
    line_spectra = Dict
    transit_dict = Dict
    transit_list = List(['None'])
    fitranges = List([])
    z = Float

    # =========================================================================
    #     Active state variables: Holding information on the currently active
    #     state (line numbers, transition etc.) as opposed to in formation
    #     containing variables that contain the model, this simply tells other
    #     objects which one is currently being operated on. Works as a kind of
    #     communication relay between different modules and the user.

    transition = Str
    lines_sel = Str

    def _transition_default(self):
        return 'None'

    def _lines_sel_default(self):
        return '1-1'



    # =========================================================================
    #     Add parameters and Traits that depend on the data loaded. Build
    #     skeleton Pandas DataFrame object to hold model and fits.

    def __init__(self, data, errs, header, z=0., LSF=None):
        self.data = data
        self.errs = errs
        self.z = z
        self.add_trait('LineLo', Range(1, self.data.shape[0]))
        self.add_trait('LineUp', Range(1, self.data.shape[0]))
        self.header = header
        exes = sp.arange(data.shape[1])
        wavl = sp.array((exes * header['CDELT1'] + header['CRVAL1']))  # * 10.)
        wavl = air_to_vacuum(wavl)  # Convert from air to vacuum wavelengths.
        self.wavlmin = float(wavl.min())
        self.wavlmax = float(wavl.max())
        self.wavlmid = sp.mean([wavl.min(), wavl.max()])
        self.LSF = np.zeros_like(wavl)
        self.add_trait(
            'Center', Range(self.wavlmin, self.wavlmax, self.wavlmid))
        self.wavl = wavl
        idx = pd.MultiIndex.from_arrays(
            [['Dummy'], ['Dummy'], ['Dummy']],
            names=['Transition', 'Rows', 'Component'])
        self.model = pd.DataFrame(
            [[0., 0., 0., 0., '0', [(0.0), (0.0)]]], index=idx,
            columns=['Pos', 'Sigma', 'Ampl', 'Line center', 'Identifier', 'Fitranges'])
        self.model = self.model.sortlevel(level='Transition')

    # =======================================================================
    #   Public methods
    # =======================================================================

    def load_model(self, filename):
        """ Loads a model from a file, then uses self.set_model() to do all the
        other necessary stuff.

        Arguments
        ---------

        * filename (string):
            String with the path to the .csv file containing the model.
        """
        model = pd.read_csv(
            filename, index_col=['Transition', 'Rows', 'Component']
        )
        if 'Fitranges' in model.columns:
            model.Fitranges = model.Fitranges.map(
                ast.literal_eval, na_action='ignore'
            )
            if model.Fitranges.dtype != object:
                model.Fitranges = model.Fitranges.astype(object)
        if 'Ampl_lock' in model.columns:
            model.Ampl_lock = model.Ampl_lock.astype(bool)
        self.set_model(model)
        return

    def set_model(self, model):
        """ imports an existing model and updates relevant attributes like
        transit_list etc. in order for the thingie to work.

        Arguments
        ---------
        * model (pandas.DataFrame object):
            Dataframe containing the model to be imported.
        """
        print(model.index.levels[0].values.sum())
        if 'Dummy' in model.index.levels[0].values.sum():
            model = model.drop('Dummy', level=0)
            print('Dummy!')
        if not 'Ampl_lock' in model.columns:
            model['Ampl_lock'] = False
        if not 'Sigma_lock' in model.columns:
            model['Sigma_lock'] = False
        if not 'Pos_lock' in model.columns:
            model['Pos_lock'] = False
        if not 'Fitranges' in model.columns:
            model['Fitranges'] = pd.Series([[] for x in model.index.labels])
        self.model = model
        self.transit_list = ['None'] + model.index.levels[0].tolist()
        return

    def find_redshift(self, name=None):
        if name == None:
            self.z = redshift_ned(self.objname).data.data[0]
        else:
            self.z = redshift_ned(name).data.data[0]
        print(('Found redshift to be z = {}'.format(self.z)))

    # =========================================================================
    #  Properties
    # =========================================================================

    @property
    def flux_from_model(self):
        return fluxes(self.model, lines=[])

    flux = flux_from_model  # Alias

    @property
    def flux_robust(self):
        """Not yet implemented
        """
        # TODO: Implement!
        raise NotImplementedError("Not yet implemented")

    @property
    def EW_model(self):
        """Not yet implemented
        """
        # TODO: Implement!
        raise NotImplementedError("Not yet implemented")

    @property
    def EW_robust(self):
        """Not yet implemented
        """
        # TODO: Implement!
        raise NotImplementedError("Not yet implemented")

    EW = EW_model  # Alias


    # =========================================================================
    #       Handlers for change of traits.

    # Update the edges of the overlay rectangle when the chosen line numbers
    #     are changed.

    def _LineLo_changed(self):
        if self.LineUp < self.LineLo:
            self.LineUp = self.LineLo
        self.lines_sel = '{}-{}'.format(self.LineLo, self.LineUp)

    def _LineUp_changed(self):
        if self.LineLo > self.LineUp:
            self.LineLo = self.LineUp
        self.lines_sel = '{}-{}'.format(self.LineLo, self.LineUp)

    def _transition_changed(self):
        print('Now working on transition: {0}'.format(self.transition))
