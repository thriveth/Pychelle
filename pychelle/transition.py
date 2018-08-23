#!/usr/bin/env python
# encoding: utf-8

# IMPORTS:
# NumPy and other functionality
import scipy as sp
# Advanced, labelled data structures
import pandas as pd
# Astronomy utilities
from astropy import units as u
# Traits - for the model handling and GUI
from traits.api import HasTraits, Dict, Bool, DelegatesTo,\
    PrototypedFrom, Instance, Button, Str, Enum
from traitsui.api import View, Group, HGroup, Item, EnumEditor
from traitsui.menu import LiveButtons, ModalButtons
from chaco.api import ArrayPlotData, GridContainer
# Other pychelle modules:
from .helper_functions import load_lines_series, air_to_vacuum, vacuum_to_air
from .spectrum2d import Spectrum2D


class Transition(HasTraits):
    # When based on Pandas, do we even need this class as anything but a viewer
    # class? I don't think so.
    # ...but it's a pretty good viewer class, that is something too.
    """ A Transition-class to keep track of lines/components for a given
    transition defined by its species and wavelength, with possible common
    aliases like e.g. H-Alpha
    """
    # We need some information about the spectrum we're working with:
    spectrum = Instance(Spectrum2D)
    wavelength = PrototypedFrom('spectrum', prefix='Center')
    model = PrototypedFrom('spectrum')
    line_spectra = DelegatesTo('spectrum')
    Center = DelegatesTo('spectrum')
    wavl = DelegatesTo('spectrum')
    CompMarkers = Dict()
    from_existing = Bool(False)
    fit_now = Bool(False)
    transit_list = DelegatesTo('spectrum')
    select_existing = Str
    Builddatabutton = Button  # For debugging only
    z = DelegatesTo('spectrum')  # Or what? How to handle this?
    exes = []
    wyes = []
    cens = []
    wids = []
    amps = []

    def _lookup_nearby_lines(self):
        ''' Still need to figure out how best to do this.
        '''
        try:
            from astroquery.atomic import AtomicLineList, Transition
        except:
            raise ImportError("Could not find a working `astroquery` install.")
        labwl = self.wavelength / (1. + self.z)
        query_range = [(labwl - 5.) * u.AA, (labwl + 5.) * u.AA]
        try:
            Q = AtomicLineList.query_object(
                query_range,
                'AIR',
                transitions=Transition.nebular,
                nmax=4
            )
        except:
            raise LookupError("Could not perform line lookup or found nothing")
        Q = pd.DataFrame(Q.as_array())
        Q.SPECTRUM += \
            '_' + Q['LAMBDA AIR ANG'].map(np.around).astype(int).astype(str)
        Q['Lambda_0'] = air_to_vacuum(Q['LAMBDA AIR ANG'])
        Q = Q[['SPECTRUM', 'Lambda_0']]
        return

    def _build_trans_list(self):
        """
        When given the Center wavelength value, find the nearby transitions
        in the system transitions list and let user choose which one to add to
        the user's transition list.
        """
        lines_srs = load_lines_series()
        #lookup_success = False
        #try:
        #    Loclines = self._lookup_nearby_lines()
        #    lines_srs.merge(Loclines, left_index=True, right_on='SPECTRUM')
        lines_selection = pd.DataFrame(lines_srs, columns=['Lambda_0'])
        lines_selection['Lambda_obs'] = \
            lines_selection['Lambda_0'] * (1. + self.z)
        lines = lines_selection[
            sp.absolute(lines_selection['Lambda_obs'] -
                        self.wavelength) <= 35.]
        # Turn this into a list of strings that Enum understands
        choices = ['%s  (%.2f)' %
                   (i, lines.ix[i]['Lambda_obs']) for i in lines.index]
        if len(choices) > 0:
            self.add_trait('choices', Enum(choices))
            return True
        else:
            choices.append('')
            self.add_trait('choices', Enum(choices))
            print('No lines listed in this neighbourhood.')
            return False

    def _z_changed(self):
        # TODO: Implement!
        pass

    def show_trans_model(self):
        self._build_plot_data()
        # This to be a conv function rather than a method? I think possibly so.
        # Main functionality of this can just be a View() in Show2DSpec. But
        # should it?

    def __init__(self, spectrum=None, z=0.):
        if spectrum is not None:
            super(Transition, self).__init__(spectrum=spectrum)
        self.Succes = self. _build_trans_list()

    # =========================================================================
    #     Define the plot that is the main part of the components editor. This
    #     requires an ArrayPlotData object to be defined, which again requires
    #     a bunch of data arrays to be defined and imported from the parent
    #     objects, so this may take a little while to implement.
    #     FIXME: Possibly remove, as most is now in the 2d viewer class.

    container = GridContainer(
        padding=30,
        bgcolor="sys_window",
        spacing=(15, 15),
        shape=(3, 1)
    )

    transitiondata = ArrayPlotData(
        wl=wavl,
    )

    # =========================================================================
    #     This class has two different views called at two different points in
    #     the data reduction - one before and one after fitting.

    view = View(
        Item('choices'),
        Item('wavelength'),
        Item('line_spectra'),
        buttons=ModalButtons
    )

    Choose = View(
        Group(
            HGroup(
                Item('choices', label='Select line'),
                Item('from_existing'),
                Item('select_existing', enabled_when='from_existing',
                     editor=EnumEditor(name='transit_list')),
                Item('fit_now', enabled_when='from_existing'),
            ),
            Item('z', label='Redshift'),
            show_border=True,
        ),
        buttons=LiveButtons,
        kind='livemodal',
        title='Pychelle - add new transition'
    )
