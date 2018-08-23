#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2012-2015 Thoeger Emil Rivera-Thorsen.
# Distributed under the terms of the GNU General Public License v. 3
# http://www.gnu.org/licenses/gpl-3.0.html

""" Standalone version of the profile builder class used in pychelle.
The module consists of a class and a convenience function to load it quicker.

Classes:
========

ProfileEditor
-------------

Args:
-----

wavlens : numpy array
    wavelength (or velocity) array.

indata : numpy array
    Fluxes.

inerrs : numpy array
    Array of standard deviations.

linecen : float
    Wavelength to consider center of the transition. Use 0 in velocity space.

fitrange:    List of tuples.
    Optional list of (min, max) tuples of wavelength/velocity ranges to
    include in fit. If none given, will fall back on linecen +/- 15.

Functions:
==========

load_profile
------------

Convenience function.

Args:
-----

data : pandas.DataFrame
    Run this module as program to see the format of dataframe.

"""

# IMPORTS:
# NumPy and other functionality
import scipy as sp
import numpy as np
# Gaussian and other functions
import scipy.stats as stats
# Collection of constants in SI units
import scipy.constants as con
# Advanced, labelled data structures
import pandas as pd
# Traits - for the model handling and GUI
from traits.api import HasTraits, Float, List, Dict, Bool, Array, DelegatesTo,\
        PrototypedFrom, Instance, Button, Str, Range, Enum, Int, Property, \
        on_trait_change, Any
from traitsui.api import View, Group, HGroup, VGroup, Item, Spring, EnumEditor
from traitsui.menu import OKButton,  CancelButton, RevertButton, LiveButtons,\
    ModalButtons, UndoButton, ApplyButton
# Enable & chaco - for plotting in the model editor GUI
from enable.component_editor import ComponentEditor
from chaco.example_support import COLOR_PALETTE
from chaco.chaco_plot_editor import ChacoPlotItem
from chaco.api import ArrayPlotData, PlotLabel, Plot, HPlotContainer, \
        GridContainer, ImageData, bone, gist_heat, gist_rainbow, DataRange1D, \
        ScatterInspectorOverlay, ColorBar, LinearMapper, Legend, \
        color_map_name_dict, KeySpec, create_line_plot, LinePlot, \
        add_default_axes, add_default_grids, OverlayPlotContainer, \
        ArrayDataSource, PolygonPlot
from chaco.tools.api import ScatterInspector, ZoomTool, PanTool, \
        BroadcasterTool, LegendTool, RangeSelection, RangeSelectionOverlay
from .profiles import gauss, voigt, lorentz
from .paired import Paired2 as Paired
from .paired import Paired as Paired2


class ProfileEditor(HasTraits):
    """ The line profile intitial guess editor class.
    This is the line profile editor module.
    It can be used to provide initial guesses to any fitting package according
    to the user's tastes.

    Usage: ProfileEditor(wave, data, errors, center)

    * wave: one-dimensional wavelength-like array, can be either
            wavelength or velocity units.
    * data: one-dimensional spectrum.
    * errors: one-dimensional noise spectrum.
    * center: Float. Central wavelength. 0 if in velocity space.
    """
    # TODO: Implement the model in a different way. Make a class for each, add
    # them as new instances, use DelegatesTo for the important parts, maybe
    # even not, maybe just use the 'object.MyInstance.attribute' notation and
    # only store the plotdata of the given model in this class...?
    # Probably needs a new way to store the parameters, though. Either the
    # Components Dict can take an extra "Kind" keyword in it, or restructure
    # the whole thing into a DataFrame object...? The latter will require a
    # major amount of work. On the other hand, it could mean much better
    # modularity.

    CompNum = Int(1)
    Components = Dict
    Locks = Dict
    # FitSettings = Dict
    CompoList = List()
    CompType = Enum([ 'Gauss', 'Absorption Voigt' 'Absorption Gauss'])

    x = Array
    mod_x = Array
    Feedback = Str

    sigmin = .1
    sigmax = 30.
    Sigma = Range(sigmin, sigmax)
    #Centr = Range(-100., 100., 0.)
    #Heigh = Range(0., 200000., 15)
    N = Range(1e12, 1e24, 1e13)
    b_param = Range(0., 200., 10.)

    # Define vars to regulate whether the above vars are locked
    #    down in the GUI:
    LockSigma = Bool()
    LockCentr = Bool()
    LockHeigh = Bool()
    LockConti = Bool()
    LockN = Bool()
    LockB = Bool()

    #continuum_estimate = Range(0., 2000.)
    plots = {}
    plotrange = ()
    resplot = Instance(Plot)
    Model = Array
    Resids = Property(Array, depends_on='Model')
    y = {}
    # Define buttons for interface:
    add_profile = Button(label='Add component')
    remove_profile = Button(label='Remove selected')
    Go_Button = Button(label='Fit model')
    plwin = Instance(GridContainer)
    select = Str
    line_center = Float()

    # Non-essentials, for use by outside callers:
    linesstring = Str('')
    transname = Str('')

    def _line_center_changed(self):
        self.build_plot()

    def _get_Resids(self):
        intmod = sp.interp(self.x, self.mod_x, self.Model)
        resids = (self.indata - intmod) / self.errs
        return resids

    def _Components_default(self):
        return {'Contin': [self.continuum_estimate, np.nan],
                'Comp1': [0., .1, 0., 'a', np.nan, np.nan, np.nan, ]}
    # Center, Sigma, Height, Identifier, Center-stddev, sigma-stddev,
    # ampl-stddev

    def _CompType_default(self):
        return 'Gauss'

    def _Locks_default(self):
        return {'Comp1': [False, False, False, False]}

    def _CompoList_default(self):
        return ['Comp1']

    def _y_default(self):
        return {}

    def _select_default(self):
        return 'Comp1'

    def build_plot(self):
        print('Building plot...')
        fitrange = self.fitrange  # Just for convenience
        onearray = Array
        onearray = sp.ones(self.indata.shape[0])
        minuses = onearray * (-1.)

        # Define index array for fit function:
        self.mod_x = sp.arange(self.line_center - 50.,
                               self.line_center + 50., .01)
        self.Model = sp.zeros(self.mod_x.shape[0])

        # Establish continuum array in a way that opens for other, more
        #   elaborate continua.
        self.contarray = sp.ones(self.mod_x.shape[0]) * \
                self.Components['Contin'][0]
        self.y = {}

        for comp in self.CompoList:
            self.y[comp] = gauss(  # x, mu, sigma, amplitude
                self.mod_x,
                self.Components[comp][0] + self.line_center,
                self.Components[comp][1],
                self.Components[comp][2]
            )

        self.Model = self.contarray + self.y[self.select]

        broca = BroadcasterTool()

        # Define the part of the data to show in initial view:
        plotrange = sp.where((self.x > self.line_center - 30) &
                             (self.x < self.line_center + 30))
        # Define the y axis max value in initial view (can be panned/zoomed):
        maxval = float(self.indata[fitrange].max() * 1.2)
        minval = maxval / 15.
        minval = abs(np.median(self.indata[fitrange])) * 1.5
        maxerr = self.errs[fitrange].max() * 1.3
        resmin = max(sp.absolute(self.Resids[self.fitrange]).max(), 5.) * 1.2
        cenx = sp.array([self.line_center, self.line_center])
        ceny = sp.array([-minval, maxval])
        cenz = sp.array([-maxval, maxval])
        # Gray shading of ignored ranges
        rangelist = np.array(self.rangelist)
        grayx = np.array(rangelist.flatten().repeat(2))
        grayx = np.hstack((self.x.min(), grayx, self.x.max()))
        grayy = np.ones_like(grayx) * self.indata.max() * 2.
        grayy[1::4] = -grayy[1::4]
        grayy[2::4] = -grayy[2::4]
        grayy = np.hstack((grayy[-1], grayy[:-1]))

        # Build plot of data and model
        self.plotdata = ArrayPlotData(
            wl=self.x,
            data=self.indata,
            xs=self.mod_x,
            cont=self.contarray,
            ones=onearray,
            minus=minuses,
            model=self.Model,
            errors=self.errs,
            ceny=ceny,
            cenz=cenz,
            cenx=cenx,
            Residuals=self.Resids,
            grayx=grayx,
            grayy=grayy,
        )

        # Add dynamically created components to plotdata
        for comp in self.CompoList:
            self.plotdata.set_data(comp, self.y[comp])
        olplot = GridContainer(shape=(2, 1), padding=10,
                               fill_padding=True,
                               bgcolor='transparent',
                               spacing=(5, 10))
        plot = Plot(self.plotdata)
        plot.y_axis.title = 'Flux density'
        resplot = Plot(self.plotdata, tick_visible=True, y_auto=True)
        resplot.x_axis.title = 'Wavelength [Å]'
        resplot.y_axis.title = 'Residuals/std. err.'

        # Create initial plot: Spectrum data, default first component,
        #   default total line profile.

        self.comprenders = []

        self.datarender = plot.plot(
            ('wl', 'data'), color='black',
            name='Data',
            render_style='connectedhold'
        )

        self.contrender = plot.plot(
            ('xs', 'cont'), color='darkgray',
            name='Cont'
        )

        self.modlrender = plot.plot(
            ('xs', 'model'), color='blue',
            line_width=1.6, name='Model'
        )

        self.centrender = plot.plot(
            ('cenx', 'ceny'),
            color='black',
            type='line',
            line_style='dot',
            name='Line center',
            line_width=1.
        )

        self.rangrender = plot.plot(
            ('grayx', 'grayy'),
            type='polygon',
            face_color='lightgray',
            edge_color='gray',
            face_alpha=0.3,
            alpha=0.3,
        )

        # There may be an arbitrary number of gaussian components, so:
        print('Updating model')
        for comp in self.CompoList:
            self.comprenders.append(
                plot.plot(
                    ('xs', comp),
                    type='line',
                    color=Paired[self.Components[comp][3]],  # tuple(COLOR_PALETTE[self.CompNum]),
                    line_color=Paired[self.Components[comp][3]],  # tuple(COLOR_PALETTE[self.CompNum]),
                    line_style='dash',
                    name=comp
                )
            )

        # Create panel with residuals:
        resplot.plot(('wl', 'Residuals'), color='black', name='Resids')
        resplot.plot(('wl', 'ones'), color='green')
        resplot.plot(('wl', 'minus'), color='green')
        resplot.plot(('cenx', 'cenz'), color='red',
                     type='line',
                     line_style='dot',
                     line_width=.5)
        resplot.plot(('grayx', 'grayy'),  # Yes, that one again
                     type='polygon',
                     face_color='lightgray',
                     edge_color='gray',
                     face_alpha=0.3,
                     alpha=0.3,)
        plot.x_axis.visible = False

        # Set ranges to change automatically when plot values change.
        plot.value_range.low_setting,\
            plot.value_range.high_setting = (-minval, maxval)
        plot.index_range.low_setting,\
            plot.index_range.high_setting = (self.line_center - 30.,
                                             self.line_center + 30.)
        resplot.value_range.low_setting,\
            resplot.value_range.high_setting = (-resmin, resmin)
        resplot.index_range.low_setting,\
            resplot.index_range.high_setting = (plot.index_range.low_setting,
                                                plot.index_range.high_setting)
        #resplot.index_range = plot.index_range  # Yes or no? FIXME
        plot.overlays.append(ZoomTool(plot, tool_mode='box',
                                      drag_button='left',
                                      always_on=False))

        resplot.overlays.append(ZoomTool(resplot, tool_mode='range',
                                         drag_button='left',
                                         always_on=False))

        # List of renderers to tell the legend what to write
        self.plots['Contin'] = self.contrender
        self.plots['Center'] = self.centrender
        self.plots['Model'] = self.modlrender
        for i in sp.arange(len(self.comprenders)):
            self.plots[self.CompoList[i]] = self.comprenders[i]

        # Build Legend:
        legend = Legend(component=plot, padding=10, align="ur")
        legend.tools.append(LegendTool(legend, drag_button="right"))
        legend.plots = self.plots
        plot.overlays.append(legend)
        olplot.tools.append(broca)
        pan = PanTool(plot)
        respan = PanTool(resplot, constrain=True, constrain_direction='x')
        broca.tools.append(pan)
        broca.tools.append(respan)
        plot.overlays.append(ZoomTool(plot, tool_mode='box',
                                      always_on=False))
        olplot.add(plot)
        olplot.add(resplot)
        olplot.components[0].set(resizable='hv', bounds=[500, 400])
        olplot.components[1].set(resizable='h', bounds=[500, 100])
        self.plot = plot
        self.resplot = resplot
        self.plwin = olplot
        self.legend = legend
        self.plotrange = plotrange

    def __init__(self, wavlens, indata, inerrs, linecen,
                 fitrange=None, fitter='lmfit', crange=[-100., 100.]):
        halfrange = 30.
        self.fitter = fitter
        self.x = wavlens
        wavmin = float(linecen - halfrange)
        wavmax = float(linecen + halfrange)
        self.fitrange = fitrange
        # print self.fitrange
        if fitrange is None:
            self.fitrange = [()]
        fitrange = []
        if len(self.fitrange) == 0:
            self.fitrange = [
                (self.line_center - halfrange, self.line_center + halfrange)]
        self.rangelist = self.fitrange
        if len(self.fitrange) > 0:
            print('Nonzero fitranges given: ', self.fitrange)
            for ran in self.fitrange:
                rmin, rmax = ran[0], ran[1]
                fitrange += sp.where((self.x > rmin) & (self.x < rmax))
            fitrange = sp.hstack(fitrange[:])
            fitrange.sort()
        self.fitrange = fitrange
        # Now the rest of the things
        self.indata = indata
        self.add_trait('Centr', Range(min(crange), max(crange), 0.))
        ### Set top and bottom data values and fit value ranges for amplitude:
        ###            ------------
        ampmin = float(-indata.std())
        ampmax = float(indata.max() + 2 * indata.std()) * 4.
        self.add_trait(
            'Heigh',
            Range( ampmin, ampmax, 0.)
        )
        ### same, for continuum:
        self.add_trait(
            'continuum_estimate',
            Range( ampmin, ampmax / 4., 0.)
        )
        ### Now add traits to represent fit limits.
        ### Then think about appropriate GUI for them
        ### So far, they will set sane default values and be scriptable.
        ###              ---------------
        self.add_trait('ampfitmax', Range(0, ampmax, ampmax))
        self.add_trait('ampfitmin', Range(ampmin, 0, ampmin))
        self.add_trait('contfitmax', Range(0, ampmax, ampmax))
        self.add_trait('contfitmin', Range(ampmin, 0, ampmin))
        ### The below version needs working on but could be
        ### the beginning of an interative interface.
        ### Commented out for now.
        ###         ---------------
        # self.add_trait('wavfitmax', Range(0, wavmax, wavmax))
        # self.add_trait('wavfitmin', Range(wavmin, 0, wavmin))
        ### Instead, we just set some sensible values.
        self.add_trait('wavfitmax', Range(0, 10., 10.))
        self.add_trait('wavfitmin', Range(-10., 0, -10.))
        self.add_trait('sigfitmax', Range(0.1, 20., 20.))
        self.add_trait('sigfitmin', Range(0.1, 20, 0.1))
        ### Add dict representing fit settings, now that
        ### all information needed is available.
        self.ampmin = ampmin
        self.ampmax = ampmax
        #self._Components_default()
        self.errs = inerrs
        self.line_center = linecen
        # Define index array for data:
        self.build_plot()

    ### =======================================================================
    #     Reactive functions: What happens when buttons are pressed, parameters
    #     are changes etc.

    # Add component to model

    def _add_profile_fired(self):
        """ Add new component to model
        """
        self.CompNum += 1
        next_num = int(self.CompoList[-1][-1]) + 1
        Name = 'Comp' + str(next_num)
        self.CompoList.append(Name)
        print("Added component nr. " + Name)
        self.Components[Name] = [0., .1, 0., chr(self.CompNum+96),
                                 np.nan, np.nan, np.nan]
        self.Locks[Name] = [False, False, False, False]
        self.select = Name
        # And the plotting part:
        #    Add y array for this component.
        # self.y[self.select] = stats.norm.pdf(
        #     self.mod_x,
        #     self.Centr + self.line_center,
        #     self.Sigma) * self.Sigma * sp.sqrt(2. * sp.pi) * self.Heigh
        self.y[self.select] = gauss(
            self.mod_x,
            self.Centr + self.line_center,
            self.Sigma,
            self.Heigh
        )
        self.plotdata[self.select] = self.y[self.select]
        render = self.plot.plot(('xs', self.select), type='line',
                                line_style='dash',
                                color=Paired[self.Components[Name][3]],  # tuple(COLOR_PALETTE[self.CompNum]),
                                line_color=Paired[self.Components[Name][3]],  # tuple(COLOR_PALETTE[self.CompNum]),
                                name=Name)
        self.plots[self.select] = render
        self.legend.plots = self.plots
        return

    def _remove_profile_fired(self):
        """ Remove the ~~last added~~ currently selected component.
        """
        if len(self.CompoList) > 1:
            comp_idx = self.CompoList.index(self.select)
            oldName = self.select  # 'Comp' + str(self.CompNum)
            newName = self.CompoList[comp_idx - 1]  # 'Comp' + str(self.CompNum - 1)
            ### newName = 'Comp' + str(self.CompNum - 1)
            self.plot.delplot(oldName)
            self.plotdata.del_data(oldName)
            del self.y[oldName]
            del self.plots[oldName]
            del self.Components[oldName]
            del self.Locks[oldName]
            self.select = newName
            print('Removed component nr. ' + str(self.CompNum))
            self.legend.plots = self.plots
            self.CompoList.pop(comp_idx)
            self.CompNum -= 1
        else:
            print('No more components to remove')

    ##=========================================================================
    #    Here follows the functionality of the GO button, split up into one
    #    function per logical step, so it is easier to script this and do some
    #    non-standard tinkering like e.g. setting odd fit constraints in the
    #    model for this transition etc.
    ##=========================================================================

    def set_fit_data(self):
        # Make sure no data arrays belonging to the parent class are altered.
        x = self.x.copy()
        data = self.indata.copy()
        errs = self.errs.copy()
        if len(self.fitrange) > 0:
            x = x[self.fitrange]
            data = data[self.fitrange]
            errs = errs[self.fitrange]
        return x, data, errs

    def create_fit_param_frame(self):
        tmpdict = self.Components.copy()
        tmpdict.pop('Contin')
        tofit = pd.DataFrame.from_dict(tmpdict).T
        tofit.columns = ['Pos', 'Sigma', 'Ampl', 'Identifier',
                         'Pos_stddev', 'Sigma_stddev', 'Ampl_stddev']
        tofit.set_value('Contin', 'Ampl', self.Components['Contin'][0])
        tofit['Line center'] = self.line_center
        tofit.set_value('Contin', 'Lock', self.LockConti)
        for lines in list(self.Components.keys()):
            if lines == 'Contin':
                continue
            tofit.set_value(lines, 'Lock', self.Locks[lines][:3])
            tofit.set_value(lines, 'AmpMax', self.ampfitmax)
            tofit.set_value(lines, 'AmpMin', self.ampfitmin)
            tofit.set_value(lines, 'SigMax', self.sigfitmax)
            tofit.set_value(lines, 'SigMin', self.sigfitmin)
            tofit.set_value(lines, 'WavMax', self.wavfitmax)
            tofit.set_value(lines, 'WavMin', self.wavfitmin)
        self.tofit = tofit

    def load_parameters_to_fitter(self, fitter='lmfit'):
        if fitter == 'lmfit':
            try:
                from . import lmfit_wrapper as lw
            except ImportError:
                print('Could not import LMfit')
                return
            self.params = lw.load_params(self.tofit)

    def fit_with_lmfit(self, method='lbfgsb', conf='covar', report=True):
        try:
            from . import lmfit_wrapper as lw
        except ImportError:
            print('Could not import LMfit')
            return
        x, data, errs = self.set_fit_data()
        result = lw.fit_it(
            self.params,
            args=(
                self.x[self.fitrange],
                self.indata[self.fitrange],
                self.errs[self.fitrange]),
                method=method)
        if report:
            lw.lf.report_fit(result)
        output = lw.params_to_grism(result, output_format='df')
        output['Identifier'] = self.tofit['Identifier']
        output.set_value('Contin', 'Identifier', sp.float64('nan'))
        output['Pos'] -= self.tofit['Line center']
        outdict = {}
        for i in output.index:
            row = output.ix[i]
            if i == 'Contin':
                outdict[i] = [row['Ampl'], row['Ampl_stddev'], row['RedChi2']]
            else:
                outdict[i] = [row['Pos'], row['Sigma'], row['Ampl'],
                              row['Identifier'], row['Pos_stddev'],
                              row['Sigma_stddev'], row['Ampl_stddev']]
        self.Components = outdict
        self.import_model()
        self.result = result
        self.output = output

    def _Go_Button_fired(self):
        # Transform the internal dict holding the model to a Pandas dataframe
        # that the lmfit wrapper will digest:
        print(('Now fitting lines {}'.format(self.linesstring)))
        self.create_fit_param_frame()
        self.load_parameters_to_fitter()
        if self.fitter == 'lmfit':
            self.fit_with_lmfit()
        else:
            raise NotImplementedError('Only LMfit backend implemented so far.')

        print(('Successfully fitted lines {} \n \n '.format(self.linesstring)))

    ##=========================================================================
    #    END of GO button functionality.
    ##=========================================================================

    # Define what to do when a new component is selected.
    def _select_changed(self):
        # First, show the values of current component in sliders!
        self.Centr = self.Components[self.select][0]
        self.Sigma = self.Components[self.select][1]
        self.Heigh = \
            min(self.ampmax, max(self.Components[self.select][2], self.ampmin))
        self.LockCentr = self.Locks[self.select][0]
        self.LockSigma = self.Locks[self.select][1]
        self.LockHeigh = self.Locks[self.select][2]
        self.plot.request_redraw()
        return

    # Every time one of the parameters in the interactive window is changed,
    #   write the change to the parameters list of the selected component.
    #   Do this one-by-one, as it is otherwise going to mess up the
    #   creation and selection of components.

    def _Centr_changed(self):
        self.Components[self.select][0] = self.Centr
        self.update_plot()
        return

    def _Sigma_changed(self):
        self.Components[self.select][1] = self.Sigma
        self.update_plot()
        return

    def _Heigh_changed(self):
        self.Components[self.select][2] = self.Heigh
        self.update_plot()
        return

    def _continuum_estimate_changed(self):
        self.Components['Contin'][0] = self.continuum_estimate
        self.update_plot()

    def _LockCentr_changed(self):
        self.Locks[self.select][0] = self.LockCentr
        return

    def _LockSigma_changed(self):
        self.Locks[self.select][1] = self.LockSigma
        return

    def _LockHeigh_changed(self):
        self.Locks[self.select][2] = self.LockHeigh
        return

    ###========================================================================
    # Define the graphical user interface

    view = View(
        Group(
            Group(
                VGroup(
                    Item('plwin', editor=ComponentEditor(),
                         show_label=False, springy=True),
                    Group(
                        HGroup(
                            Item('Centr', label='Center',
                                 enabled_when='LockCentr==False'),
                            Item('LockCentr', label='Lock'),
                        ),
                        HGroup(
                            Item('Sigma', label='Sigma',
                                 enabled_when='LockSigma==False'),
                            Item('LockSigma', label='Lock'),
                        ),
                        HGroup(
                            Item('Heigh', label='Strength ',
                                 enabled_when='LockHeigh==False'),
                            Item('LockHeigh', label='Lock'),
                        ),
                        HGroup(
                            Item('continuum_estimate',
                                 enabled_when='LockConti==False',
                                 label='Contin.  ',
                                 springy=True
                                 ),
                            Item('LockConti', label='Lock'),
                            springy=True,
                            show_border=False,
                        ),
                        show_border=True,
                        label='Component parameters'),
                        springy=True
                    ),
                    show_border=True,
                ),
            Group(Item('add_profile'),
                  Item('remove_profile'),
                  Item('Feedback', style='readonly'),
                  Item('Feedback', style='readonly'),
                  Item(
                      name='select', editor=EnumEditor(name='CompoList'),
                      style='custom'
                  ),
                  Item('Go_Button'),
                  orientation='vertical',
                  show_labels=False,
                  show_border=True),
            show_border=True,
            orientation='horizontal'),
        resizable=True,
        height=700, width=1000,  # ),
        buttons=[UndoButton, ApplyButton, CancelButton, OKButton],
        close_result=True,
        kind='livemodal',  # Works but not perfect.
        title="Pychelle - line profile editor"
    )

    compview = View(
        HGroup(Item('sigfitmin'), Item('sigfitmax')),
        HGroup(Item('ampfitmin'), Item('ampfitmax')),
        HGroup(Item('wavfitmin'), Item('wavfitmax')),
        title="Pychelle - component fit settings"
    )

    def import_model(self):
        ''' Once lpbuilder's Components dict is set; use this to set
        the state variables of the LPbuilder instance.
        '''
        self.CompoList = sorted(self.Components.keys())[:-1]
        print(self.CompoList, list(self.Components.keys()))
        #import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
        self.CompNum = len(self.CompoList)
        for com in self.CompoList:
            self.Locks[com] = [False] * 4
        self.continuum_estimate = self.Components['Contin'][0]
        self.select = self.CompoList[-1]
        self._select_changed()
        self.build_plot()
        self.update_plot()
        print('    ')

    def update_plot(self):
        self.y[self.select] = gauss(
            self.mod_x,
            self.Centr + self.line_center,
            self.Sigma,
            self.Heigh
        )
        ys = sp.asarray(list(self.y.values())).sum(0)
        self.contarray = sp.ones(self.mod_x.shape[0]) * self.continuum_estimate
        self.Model = self.contarray + ys
        self.plotdata.set_data('cont', self.contarray)
        self.plotdata.set_data(self.select, self.y[self.select])
        self.plotdata.set_data('model', self.Model)
        self.plotdata.set_data('Residuals', self.Resids)
        self.update_resid_window()  # Uncomment to keep static yscale on resids

    @on_trait_change('Resids')
    def update_resid_window(self):
        resmin = max(sp.absolute(self.Resids[self.fitrange]).max(), 5.) * 1.2
        self.resplot.value_range.low_setting,\
            self.resplot.value_range.high_setting = (-resmin, resmin)
        self.resplot.request_redraw()

###===========================================================================
#            Convenience- and helper functions
###===========================================================================

def load_profile(dataframe, centroid):
    wave = dataframe['wave'].values
    data = dataframe['data'].values
    errs = dataframe['errs'].values
    lb = ProfileEditor(wave, data, errs, centroid)
    lb.configure_traits()
    return lb


###============================================================================
#             What to do if script is called from command line rather than
#             imported from an external python module or shell.
###============================================================================

if __name__ == '__main__':
    df = pd.read_csv('testdata1d.dat')
    pe = load_profile(df, 6699.5)

