#!/usr/bin/env python
# encoding: utf-8

import scipy as sp
import numpy as np
import pandas as pd
import astropy.io.fits as pf
from traits.api import Instance, DelegatesTo, List, Str,\
    Range, Bool, HasTraits, Button, Enum, Array
from traitsui.api import View, Item, Group, HGroup, VGroup,\
    EnumEditor, Spring, LiveButtons, CheckListEditor, UItem
from traitsui.menu import OKButton, CancelButton, RevertButton,\
    UndoButton
from enable.component_editor import ComponentEditor
from chaco.api import ArrayPlotData, PlotLabel, Plot, HPlotContainer, \
    ImageData, bone, DataRange1D, create_line_plot,\
    ScatterInspectorOverlay, ColorBar, LinearMapper, \
    color_map_name_dict, OverlayPlotContainer, add_default_axes, \
    add_default_grids, ColorMapper
from chaco.tools.api import ScatterInspector, ZoomTool, \
    RangeSelection, RangeSelectionOverlay
from spectrum2d import Spectrum2D
from transition import Transition
from helper_functions import load_lines_series, _extract_1d, \
    transition_from_existing
from lpbuilder import ProfileEditor
from paired import Paired


def load_2d(filename, objname=None, redshift=False):
    """ Convenience function to open open a Spectrum2D instance."""
    HDUList = pf.open(filename, ignore_missing_end=True)
    data = HDUList[0].data
    head = HDUList[0].header
    # Load errors, set to 1. everywhere if error spec not present.
    if(len(HDUList) > 1):
        errs = HDUList[1].data
    else:
        print "No error spectrum present"
        print "Set all errors to 1."
        errs = sp.ones_like(data)
    if len(data.shape) < 2:
        print "This is a one-dimensional spectrum and cannot be opened \
                in the 2D-viewer"
    datathingie = Spectrum2D(data, errs, head)
    if objname is not None:
        datathingie.objname = objname
    if redshift:
        if objname is None:
            raise ValueError('Cannot find redshift without object name.')
        else:
            datathingie.find_redshift()
    return datathingie


def view_2d(Spectrum, Center=None):
    """ Initializes the 2d-view of the given Spectrum2D() object. """
    the_view = Show2DSpec(Spectrum=Spectrum)
    if Center is not None:
        the_view.Center = Center
    the_view.configure_traits(view='main')
    return the_view


def fit_transition_to_other(view=None, spectrum=None, transition=None,
                            freeze=['pos', 'sigma'], tie=[], method='leastsq',
                            verbose=False, rows='all'):
    """Fits a transition to a template transition.
    + Line widths are kept (almost?) constant.
    + Centroids are allowed to vary slightly, but may be tied up to fixed
      inter-component distance. The entire line can move, but the components
      cannot move relative to each other.
    + Amplitudes are by default allowed to move freely but might be frozen or
      tied to each other.

    Keyword arguments
    -----------------
    view : pychelle.Show2DSpec
        Current 2D-view object being worked on, and its spectrum.
    spectrum : pychelle.Spectrum2D
        Spectrum2D object, if no view is created yet. The function will create
        a new Show2dSpec object automatically.
    transition : str
        ### NB! Currently, this doesn't actually do anything: ###
        String containing the transition name as it appears in the Spectrum
        model index. Defaults to the `transition` property of the passed
        spectrum.
    freeze : list
        The subset of the parameters 'ampl', 'pos' or 'sigma' that should be
        frozen in the fit.
    tie : list
        Same as freeze, only the subset that should be allowed to move with
        the constraints set in the expressions below.

    Either view or spectrum must be given.

    Returns: A modified copy of the current model, with the selected transition
    fitted to the data according to the rules defined above.
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

    v = view  # Less typing!

    # Not actually used for anything. Should it be?
    if transition is None:
        transition = spectrum.transition

    if rows == 'all':
        rows = [int(s.split('-')[0]) for s in spectrum.model.index.levels[1]]

    # Cycle through all rows / row intervals in model index; fit to data.
    for s in v.model.index.levels[1]:
        nums = s.split('-')
        if int(nums[0]) not in rows:
            continue
        if len(v.model.loc[(spectrum.transition, s)].drop('Contin')) < 1:
            continue
        v.LineLo = int(nums[0])
        v.LineUp = int(nums[1])

        print('\n \n Now fitting rows {} using method {}'.format(s, method))
        lp = v.prepare_modeling()

        # Now do the stuff that the Go button in LPbuilder does (more or less):
        lp.create_fit_param_frame()
        if verbose:
            print('Parameters to fit: \n', lp.tofit)
        lp.load_parameters_to_fitter()
        # print(lp.params)

        exprdict = {}
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
        for key in exprdict.keys():
            com = lp.params[key]
            com.set(expr=exprdict[key])
        print(lp.params)
        v.lp = lp
        print 'Now fitting rows: {}'.format(s)
        v.lp.fit_with_lmfit(method=method, conf='conf')
        v.process_results()
        v._build_model_plot()
        print('Succesfully fitted rows {} using method {}\n \n'.format(s, method))

    # Don't alter any data in-place
    # transframe = spectrum.model.loc[transition].copy()
    outframe = lp.output
    return outframe


lines_srs = load_lines_series()


class SetFitRange(HasTraits):
    spectrum = Instance(Spectrum2D)
    data = DelegatesTo('spectrum')
    errs = DelegatesTo('spectrum')
    Center = DelegatesTo('spectrum')
    wavl = DelegatesTo('spectrum')
    fitranges = DelegatesTo('spectrum')
    add = Button('Add')
    reset = Button('Reset')

    def _update_plot(self):
        del(self.container.plot_components[2:])
        mv = self.markerval
        for i in self.fitranges:
            ys = sp.array([mv, mv])
            xs = sp.asarray(i)
            # print xs, type(xs), xs.dtype
            plot = create_line_plot((xs, ys), color='orange', width=5)
            plot.value_mapper = self.plot.value_mapper
            plot.index_mapper = self.plot.index_mapper
            self.container.add(plot)
            self.container.request_redraw()
            # print i
        return

    def _add_fired(self):
        if type(self.plot.active_tool.selection) == tuple:
            range_select = self.plot.active_tool
            ranges = range_select.selection
            self.fitranges.append(ranges)
            self._update_plot()
            self.plot.active_tool.deselect()

    def _reset_fired(self):
        del(self.fitranges[:])
        del(self.container.plot_components[2:])
        # print self.fitranges
        xs = sp.array([0, 1])
        ys = sp.array([0, 0])
        plot = create_line_plot((xs, ys))
        # Hack to make sure plot is properly updated:
        self.container.add(plot)
        self.container.remove(plot)
        self.container.request_redraw()
        self.plot.active_tool.deselect()

    def __init__(self, spectrum):
        super(SetFitRange, self).__init__(spectrum=spectrum)
        self.rows = self.data.shape[0]
        # self.rangexs = sp.array([])
        # self.rangeys = sp.array([])
        if spectrum.transition != 'None':
            try:
                has_range = self.spectrum.model.notnull().get_value(
                    (self.spectrum.transition,
                     self.spectrum.lines_sel,
                     'Contin'),
                    'Fitranges'
                )

                if not has_range:
                    raise ValueError
                else:
                    the_range = self.spectrum.model.get_value(
                        (self.spectrum.transition,
                         self.spectrum.lines_sel,
                         'Contin'),
                        'Fitranges'
                    )
                    self.fitranges = the_range
            except KeyError:
                self.fitranges = []
            except ValueError:
                self.fitranges = []

        data1d, errs1d = _extract_1d(self.data, self.errs, 1, self.rows-1)
        data1d, errs1d = _extract_1d(
            self.data, self.errs, spectrum.LineLo, spectrum.LineUp
        )
        plotindex = sp.where(
            (self.wavl > self.Center - 50) & (self.wavl < self.Center + 50))
        self.markerval = data1d[plotindex].max() * 1.05
        container = OverlayPlotContainer(
            padding=40, bgcolor="white",
            use_backbuffer=True,
            border_visible=True,
            fill_padding=False
        )
        self.plot = create_line_plot(
            (self.wavl[plotindex], data1d[plotindex]),
            color='black',)
        add_default_grids(self.plot)
        add_default_axes(self.plot)
        self.plot.value_range.set_bounds(data1d[plotindex].max() * -.1,
                                         data1d[plotindex].max() * 1.1)
        self.plot2 = create_line_plot(
            (self.wavl[plotindex], errs1d[plotindex]),
            color='green',)
        self.plot2.value_mapper = self.plot.value_mapper
        self.plot2.index_mapper = self.plot.index_mapper
        container.add(self.plot)
        container.add(self.plot2)
        self.plot.active_tool = RangeSelection(
            self.plot,)
        self.container = container
        self.plot.overlays.append(RangeSelectionOverlay(component=self.plot))

        if len(self.fitranges) > 0:
            self._update_plot()

    view = View(
        Item('container', editor=ComponentEditor(), show_label=False),
        HGroup(
            Item('add', show_label=False),
            Item('reset', show_label=False),),
        buttons=LiveButtons,
        kind='livemodal',
        resizable=True,
        height=700, width=900,
    )


class Show2DSpec(HasTraits):
    """The class that displays a Spectrum2D instance.

    A module to view and select regions to fit from the 2D-spectrum.
    Selected regions are forwarded to the ProfileEditor class where they can be
    modelled and later (in-program or outside?) can be fitted by a fitting
    backend, e.g. Sherpa.
    It takes as input a 2D numpy array and a PyFITS header object
    (other options later?)
    """

    Spectrum = Instance(Spectrum2D)
    data = DelegatesTo('Spectrum')
    errs = DelegatesTo('Spectrum')
    header = DelegatesTo('Spectrum')
    wavl = DelegatesTo('Spectrum')
    wavlmid = DelegatesTo('Spectrum')
    wavlmin = DelegatesTo('Spectrum')
    wavlmax = DelegatesTo('Spectrum')
    LineLo = DelegatesTo('Spectrum')
    LineUp = DelegatesTo('Spectrum')
    Lines = DelegatesTo('Spectrum')  # Dict of linenums in str and flt form
    LSF = DelegatesTo('Spectrum')
    Center = DelegatesTo('Spectrum')
    line_spectra = DelegatesTo('Spectrum')
    transit_dict = DelegatesTo('Spectrum')
    transit_list = DelegatesTo('Spectrum')
    model = DelegatesTo('Spectrum')
    transition = DelegatesTo('Spectrum')
    fitranges = DelegatesTo('Spectrum')
    add_trans = Button(label='New transition')
    specplot = Instance(Plot)
    fit_this = Button(label='Guess / Fit')
    # Whether to show color range editor
    ColorRange = Bool(False)
    Interact = Bool(True)
    line_sel_lock = Bool(False)
    show_model_comps = Bool(True)
    ShowContin = Button(label='Show/Edit continuity plots')
    ShowColran = Button(label='Show/Edit')
    ShowFitran = Button(label='Show/Edit')
    ColorScale = Enum(['Linear', 'Sqrt', 'Log'])
    colormaps = color_map_name_dict
    colormaps_name = Enum(sorted(colormaps.keys()))
    # For continuity plot window:
    set_label = Button(label='Set identifier label')
    remove_comp = Button(label='Remove selected')
    unselect_all = Button(label='Clear selections')
    apply_to_all_transitions = Bool(False)
    all_labels = List(editor=CheckListEditor(values=[], cols=1,))
    the_label = Str()

    def _build_model_plot(self):
        ''' This helper method builds the model plot and rebuilds it when the
        model is changed. First, construct the x and y values:
        '''
        rowcoords = []
        rowthicks = []
        # Get row number labels, and get list of the mean of each of them and
        # the width of the rows.
        for row in self.model.drop('Dummy', level=0).drop('Contin', level=2).\
                index.get_level_values(1):
            rowcoords.append(sp.float64(row.split('-')).sum() / 2.)
            rowthicks.append(sp.float64(row.split('-')).ptp())
        rowcoords = sp.array(rowcoords) - 0.5
        rowthicks = sp.array(rowthicks)

        # Get the identifier tags, map them t integers for colormapping in
        # plot (or other mapping as one would like):
        id_colors = (self.model.drop('Dummy', level=0).drop('Contin', level=2)
                     .Identifier.map(ord).values - 97) % 12 + 1
        pos_frame = self.model.drop('Dummy', level=0)\
            .drop('Contin', level=2)
        pos_array = pos_frame.Pos.values + pos_frame['Line center'].values
        cont_series = self.model.drop('Dummy', level=0).loc[
            self.model.drop('Dummy', level=0).Identifier.isnull()]

        # Check if a transition is selected. If yes, then create continuity
        # plot data arrays consisting of the selected transition alone;
        # otherwise, create an empty plot.
        if self.transition in self.model.index.levels[0]:
            current_pos_frame = pos_frame.loc[self.transition]
            current_pos_array = current_pos_frame.Pos.values\
                + current_pos_frame['Line center'].values
            current_cont = cont_series.loc[self.transition]
            curr_cont = current_cont.Ampl.loc[
                current_cont.Identifier.isnull()
            ]
            curr_id = (pos_frame.loc[self.transition].Identifier.map(ord)
                       .values - 97) % 12 + 1
            current_ys = current_pos_frame.index.droplevel(1).map(
                lambda x: x.split('-')[0]).astype(float) - .5
            amp_array = self.model.loc[self.transition]\
                .drop('Contin', level=1).Ampl.values
            sig_array = self.model.loc[self.transition]\
                .drop('Contin', level=1).Sigma.values
            cont_y = cont_series.loc[self.transition].index.droplevel(1)\
                .map(lambda x: x.split('-')[0]).astype(float) - 0.5
            cont_array = curr_cont.values
        else:
            current_pos_array = np.array([])
            sig_array = np.array([])
            amp_array = np.array([])
            cont_array = np.array([])
            cont_y = np.array([])
            current_ys = np.array([])
            curr_id = np.array([])
            curr_cont = np.array([])

        # Inject into the class' ArrayPlotData object, so it's available for
        # all Plot instances that read this.
        self.plotdata.set_data('model_y', rowcoords)
        self.plotdata.set_data('model_x', pos_array)
        self.plotdata.set_data('model_w', rowthicks)
        self.plotdata.set_data('model_colors', id_colors)
        self.plotdata.set_data('model_amp', amp_array)
        self.plotdata.set_data('model_sig', sig_array)
        self.plotdata.set_data('contin_amp', cont_array)
        self.plotdata.set_data('contin_y', cont_y)
        self.plotdata.set_data('curr_pos', current_pos_array)
        self.plotdata.set_data('curr_y', current_ys)
        self.plotdata.set_data('curr_id', curr_id)

        # Update ranges for the continuety plots:
        #  First, check if a transition is selected. Otherwise, create empty
        #  plots with generic [0, 1] ranges.
        Posrange = np.array([[current_pos_array.min(), pos_array.max()]
                            if len(current_pos_array) > 1
                            else [0., 1.]][0])
        Amprange = np.array([[amp_array.min(), amp_array.max()]
                            if len(amp_array) > 1
                            else [0., 1.]][0])
        Sigrange = np.array([[sig_array.min(), sig_array.max()]
                            if len(sig_array) > 1
                            else [0., 1.]][0])
        # Now, set the ranges.
        self.Posplot.index_range.set_bounds(
            Posrange.min()
            - Posrange.ptp() * .1,
            Posrange.max()
            + Posrange.ptp() * .1
        )
        self.Ampplot.index_range.set_bounds(
            Amprange.min()
            - Amprange.ptp() * .1,
            Amprange.max()
            + Amprange.ptp() * .1
        )
        self.Sigplot.index_range.set_bounds(
            Sigrange.min()
            - Sigrange.ptp() * .1,
            Sigrange.max()
            + Sigrange.ptp() * .1)
        return

    def _transition_default(self):
        return ''

    def __init__(self, Spectrum, center=None):
        """ Non-passive object needs contructor to know and manipulate its own
        Traits. """
        # Make sure this constructor function can access parent spec Traits.
        super(Show2DSpec, self).__init__(Spectrum=Spectrum)
        # After Use newly accessed self awareness to construct new Traits
        self.transit_list.extend(self.transit_dict.keys())
        self.all_labels = self.model.drop('Dummy', level=0)\
            .drop('Contin', level=2)['Identifier'].unique().tolist()
        # print self.all_labels
        self.add_trait(
            'val_max',
            Range(
                -0., float(self.data[30:-30, :].max()),
                float(self.data[30:-30, 1000:2000].max())))
        self.add_trait(
            'val_min',
            Range(0., float(self.data[30:-30, :].max()), 0.))
        # Create arrays for plotting and minor helper-arrays, define misc.
        # values etc.
        wyes = sp.arange(self.data.shape[0])

        # Data for image plot, initial.
        self.imgdata = ImageData(data=self.data[:, :-1], value_depth=1)

        # Data for different color scales:
        self.lindata = self.data[:, :-1]  # , value_depth=1
        tempdata = self.data.copy()
        tempdata[tempdata < 0] = 0.
        self.sqrtdata = np.sqrt(tempdata[:, :-1].copy())  # , value_depth=1
        tempdata[tempdata == 0.] = 0.001
        self.logdata = np.log10(tempdata[:, :-1].copy())

        linexes = sp.array([self.Center, self.Center])
        loexes = linexes - 30.
        hiexes = linexes + 30.
        linwyes = sp.array([wyes.min(), wyes.max()])
        indrange = sp.array(
            [self.wavlmin, self.wavlmax, self.wavlmax, self.wavlmin])
        valrange = sp.array(
            [self.LineUp, self.LineUp, self.LineLo - 1, self.LineLo - 1])
        model_x = sp.array([0.])
        model_y = sp.array([0.])
        model_w = sp.array([0.])
        model_amp = sp.array([0.])
        contin_amp = sp.array([0.])
        contin_y = sp.array([0.])
        curr_y = sp.array([0.])
        model_sig = sp.array([0.])
        dummy = sp.array([0.])
        model_colors = sp.array([0.])
        curr_pos = sp.array([0.])
        curr_id = sp.array([0.])

        # ==============================================================
        # Define ArrayPlotData object and the plot itself.

        # Plot data object:
        self.plotdata = ArrayPlotData(
            dummy=dummy,
            imagedata=self.imgdata,
            exes=linexes,
            loex=loexes,
            hiex=hiexes,
            wyes=linwyes,
            polexes=indrange,
            polwyes=valrange - .0,
            model_x=model_x,
            model_y=model_y,
            model_w=model_w,
            curr_pos=curr_pos,
            curr_id=curr_id,
            curr_y=curr_y,
            model_amp=model_amp,
            contin_amp=contin_amp,
            contin_y=contin_y,
            model_sig=model_sig,
            model_colors=model_colors
        )
        # Plot object containing all the rest.
        myplot = Plot(self.plotdata)
        # Define image plot of the 2D data
        self.my_plot = myplot.img_plot(
            "imagedata",
            name='Image',
            colormap=bone,  # gist_heat,
            xbounds=self.wavl[:]  # -1],
        )[0]
        #  Normalise the colorbar to the value range of the data.
        self.my_plot.value_range.set_bounds(self.val_min, self.val_max)
        # Define the rectangle overlay showing wich lines are selected.
        self.selected = myplot.plot(
            ('polexes', 'polwyes'),
            name='Selected rows',
            type='polygon',
            face_color=(0.5, 0.5, 0.9) + (0.3,),
            edge_width=0.3,
            edge_color=(1., 1., 1.) + (1.,),
            edge_alpha=1.,
        )[0]

        # Once defined, add these to the plot object.
        myplot.add(self.my_plot)
        myplot.add(self.selected)

        # Marker for the center line
        center_marker = myplot.plot(
            ('exes', 'wyes'),
            type='line',
            color='green', line_width=1.5, alpha=1.)
        # Lower and upper limits for the 1D plot shown with markers.
        lower_marker = myplot.plot(
            ('loex', 'wyes'),
            type='line',
            color='yellow',
            line_width=.3)
        higher_marker = myplot.plot(
            ('hiex', 'wyes'), type='line',
            color='yellow', line_width=.3)

        # Model plot.
        # For some reason, all elements of a Plot() instance by default share
        # the same mappers. This has the slightly bizarre consequence that we
        # have to assign the model plot the wrong colormapper (which will then
        # be valid globally, i.e. for the image plot), and then after the fact
        # assign a different color_mapper to the actual model plot.
        # Yes, seriously.
        self.centroids_plot = myplot.plot(
            ('model_x', 'model_y', 'model_colors'),
            type='cmap_scatter',
            vmin=97,
            vmax=123,
            color_mapper=bone,  # gist_heat,
            name='centroids',
            marker_size=2.5,
            outline_color='transparent',
        )
        # Set fixed color range based on ColorBrewer 'Paired' sequence
        paired_mapper = ColorMapper.from_palette_array(
            [Paired[x] for x in sorted(Paired.keys())][:12],
            # Paired.values()[:12],
            range=DataRange1D(low=1, high=12),
            steps=12
        )

        self.paired_mapper = paired_mapper
        self.centroids_plot[0].color_mapper = paired_mapper  # (

        # =====================================================================
        # And now: the parameter-space continuity plot.

        ContCont = HPlotContainer(
            use_backbuffer=True,
            resizable='hv',
            bgcolor='transparent',
            spacing=-50,
            padding_top=20,
        )

        Posplot = Plot(self.plotdata)
        Sigplot = Plot(self.plotdata)
        Ampplot = Plot(self.plotdata)

        # posplot = Posplot.plot(  # Same as 'centroids'! Different container.
        #     ('model_x', 'model_y', 'model_colors'),
        #     type='cmap_scatter',
        #     color_mapper=paired_mapper,  # gist_rainbow,
        #     marker_size=4,
        #     outline_color='gray',
        #     name='Positions',
        #     bgcolor='whitesmoke',
        #     # bgcolor='lavender',
        # )

        posplot = Posplot.plot(  # Same as 'centroids'! Different container.
            ('curr_pos', 'curr_y', 'curr_id'),
            type='cmap_scatter',
            color_mapper=paired_mapper,  # gist_rainbow,
            marker_size=4,
            outline_color='gray',
            name='Positions',
            bgcolor='whitesmoke',
            # bgcolor='lavender',
        )
        ampplot = Ampplot.plot(
            ('model_amp', 'curr_y', 'curr_id'),
            type='cmap_scatter',
            color_mapper=paired_mapper,  # gist_rainbow,
            marker_size=4,
            name='Amplitudes',
            bgcolor='cornsilk',
            index_scale='log',
            # bgcolor='white',
        )

        contplot = Ampplot.plot(
            ('contin_amp', 'contin_y'),
            type='scatter',
            color='black',
            name='Continuum',
        )

        sigplot = Sigplot.plot(
            ('model_sig', 'curr_y', 'curr_id'),
            type='cmap_scatter',
            color_mapper=paired_mapper,  # gist_rainbow,
            marker_size=4,
            name='Sigma',
            bgcolor='lavender',
            # bgcolor='thistle',
            # bgcolor='white',
        )

        Posplot.title = 'Centroid positions'
        Posplot.value_axis.title = 'Row #'
        Posplot.index_axis.title = 'Wavelength [Å]'

        Ampplot.title = 'Amplitudes'
        Ampplot.index_axis.title = 'Amplitude [flux]'

        Sigplot.title = 'Line widths'
        Sigplot.index_axis.title = 'Line width [Å]'

        ContCont.add(Posplot)
        ContCont.add(Ampplot)
        ContCont.add(Sigplot)
        ContCont.overlays.append(
            PlotLabel(
                ' '.join(
                    "Select Points on Centroids plot and assign them a label. \
                    Zoom by using the mouse wheel or  holding Ctrl and \
                    dragging mouse to mark zoom region. Use ESC to revert \
                    zoom.".split()),
                component=ContCont,
                overlay_position='top'))
        # Attach some tools to the plot
        Posplot.overlays.append(ZoomTool(Posplot))
        Sigplot.overlays.append(ZoomTool(Sigplot))
        Ampplot.overlays.append(ZoomTool(Ampplot))

        # Add ScatterInspector tool and overlay to the Posplot part.
        posplot[0].tools.append(ScatterInspector(posplot[0]))
        overlay = ScatterInspectorOverlay(
            posplot[0],
            hover_color="red",
            hover_marker_size=5,
            selection_marker_size=4,
            selection_color="transparent",
            selection_outline_color="white",
            selection_line_width=1.5)

        posplot[0].overlays.append(overlay)
        Posplot.value_range.set_bounds(wyes.min(), wyes.max())
        Posplot.index_range.set_bounds(model_x.min() * .9,
                                       model_x.max() * 1.1)
        Ampplot.value_range = Posplot.value_range
        Ampplot.index_range.set_bounds(model_amp.min() * .9,
                                       model_amp.max() * 1.1)
        Sigplot.value_range = Posplot.value_range
        Sigplot.index_range.set_bounds(model_sig.min() * .9,
                                       model_sig.max() * 1.1)
        self.Posplot = Posplot
        self.posplot = posplot
        self.Ampplot = Ampplot
        self.Sigplot = Sigplot
        self.ContCont = ContCont

        # Create the colorbar, handing in the appropriate range and colormap
        colormap = self.my_plot.color_mapper
        colorbar = ColorBar(
            index_mapper=LinearMapper(range=colormap.range),
            color_mapper=colormap,
            plot=self.my_plot,
            orientation='v',
            resizable='v',
            width=25,
            padding=20)
        colorbar.padding_top = myplot.padding_top
        colorbar.padding_bottom = myplot.padding_bottom
        container = HPlotContainer(use_backbuffer=True)
        container.add(myplot)
        container.add(colorbar)
        container.bgcolor = "sys_window"
        self.container = container
        self.specplot = myplot
        # If a center is given in the call, set this.
        if center is not None:
            self.Center = center
        # Set the wavelength range to show.
        self.specplot.index_range.low_setting, \
            self.specplot.index_range.high_setting\
            = (self.Center - 55, self.Center + 55)
        self.wyes = wyes  # For debugging
        if len(self.model) > 1:
            self._build_model_plot()
        #                    END __init__()
        # =====================================================================

    # =========================================================================
    #       Handlers for change of traits.
    # =========================================================================

    # Update the edges of the overlay rectangle when the chosen line numbers
    #     are changed.

    def _LineLo_changed(self):
        self.plotdata.set_data(
            'polwyes',
            sp.array([self.LineUp, self.LineUp,
                      self.LineLo - 1, self.LineLo - 1]) + .0)
        try:
            fitrange = self.model\
                .loc[self.transition]\
                .loc['{}-{}'.format(self.LineLo, self.LineUp)]\
                .loc['Contin']['Fitranges']
            if np.isnan(fitrange).any():
                self.fitranges = []
            else:
                self.fitranges = fitrange
        except KeyError:
            self.fitranges = []

    def _LineUp_changed(self):
        self.plotdata.set_data(
            'polwyes',
            sp.array([self.LineUp, self.LineUp,
                      self.LineLo - 1, self.LineLo - 1]) + .0)

    # When Center is changed, move the image accordingly, but make sure the
    # center and wing markers are still in the middle of the plot.

    def _Center_changed(self):
        if self.transition != 'None' and self.line_sel_lock is False:
            self.transition = 'None'
        self.specplot.index_range.low_setting, \
            self.specplot.index_range.high_setting\
            = (self.Center - 55, self.Center + 55)
        self.plotdata.set_data('exes', sp.array([self.Center, self.Center]))
        self.plotdata.set_data(
            'loex', sp.array([self.Center, self.Center]) - 30.)
        self.plotdata.set_data(
            'hiex', sp.array([self.Center, self.Center]) + 30.)
        self.my_plot.request_redraw()

    def _transition_changed(self):
        """Change the Center parameter to that of the selected transition."""
        if self.transition == 'None':
            pass
        else:
            print 'New transition selected: ', self.transition
            transwl = self.transition.split('(')[1][:-1]
            # Make sure the selected line is *not* changed to 'None' when we
            # jump to the center of the newly selected line:
            self.line_sel_lock = True
            self.Center = float(transwl)
            self.line_sel_lock = False
            self.transwl = float(transwl)
            self._build_model_plot()

    # Mostly to deal with a manually set model in the Spectrum2D Instance:
    def _transit_list_changed(self):
        self._build_model_plot()

    # Update color scale when requested from GUI.
    def _val_min_changed(self):
        if self.val_min > self.val_max:
            self.val_max = self.val_min
        self.my_plot.value_range.set_bounds(self.val_min, self.val_max)
        self.my_plot.request_redraw()
        self.specplot.request_redraw()

    def _val_max_changed(self):
        if self.val_min > self.val_max:
            self.val_min = self.val_max
        self.my_plot.value_range.set_bounds(self.val_min, self.val_max)
        self.specplot.request_redraw()

    def _show_model_comps_changed(self):
        clr_range = self.my_plot.color_mapper.range
        if self.show_model_comps is True:
            self.specplot.showplot('centroids')
        if self.show_model_comps is False:
            self.specplot.hideplot('centroids')

        # Ugly hack to make sure plot updates:
        self.plotdata.set_data('model_y', self.plotdata['model_y'] + .5)
        self.plotdata.set_data('model_y', self.plotdata['model_y'] - .5)
        # </ugly hack>
        self.my_plot.request_redraw()
        self.container.request_redraw()
        self.specplot.request_redraw()

    # =========================================================================
    #      Handler for fired button and possibly other events added in future

    def _add_trans_fired(self):
        transition = Transition(spectrum=self.Spectrum)
        if transition.Succes:
            foo = transition.configure_traits(view='Choose')
            transname = transition.choices
            # print transname
            # print ' '.join(transname.split()[:-1])
            # import pdb; pdb.set_trace()  # XXX BREAKPOINT
            transwl = lines_srs.loc[' '.join(transname.split()[:-1])]\
                ['Lambda_0'] * (1 + transition.z)
            # print 'transwl: ', transwl
            if foo:
                print "This is Show2DSpec: transition added '" \
                    + transname + "'"
                self.transition = transname  # *After* setting transit_dict.
                # We don't want duplicate entries in the transition list:
                if transname not in self.transit_list:
                    self.transit_list.append(transname)
                # print 'Transition: ', self.transition
                # If from_existing is selected, transfer and transform existing
                # transition to new.
                if transition.from_existing:
                    oldtrans = transition.select_existing
                    oldwl = self.model['Line center'].dropna().loc[oldtrans][0]
                    LSF_old = np.interp(oldwl, self.wavl, self.LSF)
                    LSF_new = np.interp(transwl, self.wavl, self.LSF)
                    self.model = transition_from_existing(
                        self.model, oldtrans, transname, transwl,
                        LSF=[LSF_old, LSF_new]
                    )

                    if transition.fit_now:
                        try:
                            fit_transition_to_other(self)
                        except:
                            print 'Quick fit did not succeed.'
                            raise

                self._build_model_plot()
            else:
                print 'Cancelled, no new transition added.'
        else:
            print 'Something wrong when adding transition! \n'
        return

    # =========================================================================
    #    Functionality of the "Fit this" button, chopped up in one function per
    #    logical step, to allow more flexible scripting and doing nonstandard
    #    things that are not easily integrated into the GUI.

    def get_fitdata(self):
        data1d, errs1d = _extract_1d(
            self.data, self.errs, self.LineLo, self.LineUp)
        return data1d, errs1d

    def define_fitranges(self):
        if self.transition == 'None':
            transname = 'Center: ' + str(self.Center)
            self.transition = transname
        else:
            transname = self.transition
        Lines = [self.LineLo - 1, self.LineUp]
        linesstring = str(Lines[0] + 1) + '-' + str(Lines[1])
        if (transname, linesstring, 'Contin') in self.model.index:
            fitranges = self.model.get_value(
                (transname, linesstring, 'Contin'), 'Fitranges')
            # print 'Fitranges Show2dspec: ', fitranges
            if type(fitranges) == float:
                fitranges = [(self.Center - 30., self.Center + 30)]
        elif len(self.fitranges) > 0:
            # print 'self.fitranges was longer than 0 but fitranges not in model'
            fitranges = self.fitranges
        else:
            fitranges = [(self.Center - 30., self.Center + 30)]
        return transname, linesstring, fitranges

    def prepare_modeling(self):
        transname, linesstring, fitranges = self.define_fitranges()
        data1d, errs1d = self.get_fitdata()
        lp = ProfileEditor(
            self.wavl, data1d, errs1d, self.Center, fitrange=fitranges
        )
        lp.transname = transname
        lp.linesstring = linesstring
        if len(self.fitranges) == 0:
            self.fitranges = [(self.wavlmin, self.wavlmax)]
        print 'Components before: ', lp.Components
        # Inject existing model into the LineProfile object if there is one.
        if ((transname in self.model.index.levels[0]) and
                (linesstring in self.model.index.levels[1])):
            to_insert = self.model.loc[transname].loc[linesstring]
            # Remove the automatically  created components in lp. model.
            print to_insert.index
            if 'Comp' not in to_insert.index:
                pass
            if 'Comp1' not in to_insert.index:
                if len(to_insert.index) > 1:
                    lp.Components.pop('Comp1')
                    lp.CompoList = []
            # Make sure all columns are there
            for S in ['Pos_stddev', 'Sigma_stddev', 'Ampl_stddev']:
                if S not in to_insert.columns:
                    to_insert[S] = np.nan
            # Insert model into LPbuilder model.
            for i in to_insert.index:
                if len(to_insert.index) == 1:
                    continue
                if i == 'Contin':
                    lp.Components[i][0] = to_insert.xs(i)['Ampl']
                else:
                    lp.Components[i] = list(
                        to_insert[
                            ['Pos', 'Sigma', 'Ampl', 'Identifier',
                             'Pos_stddev', 'Sigma_stddev', 'Ampl_stddev']]
                        .xs(i).values
                    )
        print 'Components after: ', lp.Components
        print lp.CompoList
        lp.import_model()
        return lp

    def process_results(self):
        """ Inserts fit results from LPbuilder into model of Spectrum2D.

        NOTE
        ----
        This method makes in-place changes to the model.
        """
        #  Since there may be more components in the old model than the new,
        #  we cannot overwrite component-wise, so we remove entire submodel
        #  for the current transname-linesstring combo and rewrite it.
        self.model.sortlevel(0)
        transname, linesstring = self.lp.transname, self.lp.linesstring
        if ((transname in self.model.index.levels[0].values.sum()) and
                (linesstring in self.model.index.levels[1].values.sum())):
            self.model = self.model\
                .unstack()\
                .drop((transname, linesstring))\
                .stack()
        for thekey in self.lp.Components.keys():
            if not thekey == 'Contin':
                self.model = self.model.set_value(
                    (transname, linesstring, thekey),
                    'Ampl',
                    self.lp.Components[thekey][2]
                )
                self.model = self.model.set_value(
                    (transname, linesstring, thekey),
                    'Ampl_stddev',
                    self.lp.Components[thekey][6]
                )
                self.model = self.model.set_value(
                    (transname, linesstring, thekey),
                    'Pos',
                    self.lp.Components[thekey][0]
                )
                self.model = self.model.set_value(
                    (transname, linesstring, thekey),
                    'Pos_stddev',
                    self.lp.Components[thekey][4]
                )
                self.model = self.model.set_value(
                    (transname, linesstring, thekey),
                    'Sigma',
                    self.lp.Components[thekey][1]
                )
                self.model = self.model.set_value(
                    (transname, linesstring, thekey),
                    'Sigma_stddev',
                    self.lp.Components[thekey][5]
                )
                self.model = self.model.set_value(
                    (transname, linesstring, thekey),
                    'Identifier',
                    self.lp.Components[thekey][3]
                )
                # Keep track of line center position for each transition:
                self.model = self.model.set_value(
                    (transname, linesstring, thekey),
                    'Line center',
                    self.Center
                )
            else:
                self.model = self.model.set_value(
                    (transname, linesstring, thekey),
                    'Ampl',
                    self.lp.Components[thekey][0]
                )
                self.model = self.model.set_value(
                    (transname, linesstring, thekey),
                    'Ampl_stddev',
                    self.lp.Components[thekey][1]
                )
                self.model = self.model.set_value(
                    (transname, linesstring, thekey),
                    'Fitranges',
                    self.lp.rangelist
                )
                try:
                    self.model = self.model.set_value(
                        (transname, linesstring, thekey),
                        'RedChi2',
                        self.lp.result.redchi
                    )
                except:
                    print ('No fit performed for lines {}, RedChi set to NaN'
                           .format(self.Spectrum.lines_sel)
                    )
                    self.model = self.model.set_value(
                        (transname, linesstring, thekey),
                        'RedChi2', np.nan
                    )
            self.model = self.model.sort()
        # return

    def _fit_this_fired(self):
        # Extract rows to 1D spectrum, send this to
        #     ProfileEditor class:
        print '   '
        print 'Now modelling selected rows and transition:'
        # Is this step really necessary?
        transname, linesstring, fitranges = self.define_fitranges()
        print 'Transition to be modelled:', transname
        print 'Rows selected: {0} to {1}'.format(self.LineLo, self.LineUp)

        self.lp = self.prepare_modeling()
        # Now, ready to rock.
        new_model = self.lp.configure_traits(view='view')
        print 'Line Profile return: ', new_model  # .result
        # When done creating the guessed or fitted model,
        #  insert it into Pandas DataFrame.
        if new_model:  # .result:
            self.process_results()
        self._build_model_plot()
        self.Posplot.request_redraw()
        return

    # End of "Fit this" button functionality.
    # =========================================================================

    # =========================================================================
    #     The different window layouts. So far the main window, the volor range
    #     editor, the continuity plot window and the ID label assigning window.
    # =========================================================================

    # Main window.
    main = View(
        Item(
            'container',
            editor=ComponentEditor(),
            resizable=True,
            show_label=False,
            width=1400
        ),
        Group(
            Group(
                Item('Center', show_label=False, springy=True),
                springy=True,
                show_border=True,
                label='Center'
            ),
            HGroup(
                Item('LineLo', style='custom', label='lower', springy=True),
                Item('LineUp', style='custom', label='Upper', springy=True),
                label='Selected rows',
                show_border=True,
                springy=True
            ),
            HGroup(
                Item('show_model_comps', label='Show'),
                label='Model',
                show_border=True
            ),
            orientation='horizontal'
        ),
        HGroup(
            HGroup(
                Item('ShowContin', show_label=False),
                label='Model parameter inspector',
                show_border=True,
            ),
            HGroup(
                Item('ShowColran', show_label=False),
                label='Color Range',
                show_border=True,
            ),
            HGroup(
                Item('ShowFitran', show_label=False),
                label='Fit wavelength range',
                show_border=True,
            ),
            HGroup(
                Item('transition',
                     editor=EnumEditor(name='transit_list'),
                     label='Choose'
                     ),
                Item('add_trans', show_label=False),
                label='Transition:',
                show_border=True,
            ),
            Spring(),
            Item('fit_this', show_label=False),
            springy=True,
        ),
        buttons=[OKButton, CancelButton],
        resizable=True,
        title='Pychelle - 2D viewer & selector')

    # The color range editor window
    ColranView = View(
        Group(
            VGroup(
                Item('val_min', label='Min:', springy=True),
                Item('val_max', label='Max:', springy=True),
                label='Cut levels',
                show_border=True,
            ),
            HGroup(
                Item('ColorScale', label='Color scale'),  # style='custom'),
                Item('colormaps_name', label='Colormap'),
                label='Scale and colors',
                show_border=True,
            ),
        ),
        title='Edit plot look & feel',
        buttons=[UndoButton, RevertButton, CancelButton, OKButton]
    )

    # The UI window showing the continuity plots and calling the Identifier
    # label assigning window below.
    ContinView = View(
        VGroup(
            Group(
                Item(
                    'ContCont',
                    editor=ComponentEditor(),
                    show_label=False,
                    ),
                '|',
                UItem(
                    'all_labels',
                    show_label=False,
                    style='readonly',
                    label='Show/hide components'
                )
            ),
            HGroup(
                Item('unselect_all', show_label=False),
                Spring(),
                Item('apply_to_all_transitions'),
                Item('set_label', show_label=False),
                Item('remove_comp', show_label=False),
                springy=True,
            ),
        ),
        resizable=True,
        width=1200.,
        height=600.,
        buttons=[UndoButton, RevertButton, CancelButton, OKButton],
        kind='live',
    )

    # Interface to set identifier labels.
    ReassignView = View(
        HGroup(
            Item(
                'all_labels', style='readonly',
                label='Existing',
                show_label=False
            ),
            VGroup(
                Item('the_label', label='Set identifier label',
                     style='custom'),
            )
        ),
        buttons=[CancelButton, OKButton],
        close_result=True,
        kind='livemodal'
    )

    AreYouSureString = 'Pressong OK will permanently delete \n' +\
        'the selected components'
    AreYouSureView = View(
        Item('AreYouSureString', style='readonly'),
        buttons=['OK', 'Cancel']
    )

    def _ShowColran_fired(self):
        self.edit_traits(view='ColranView')

    def _ColorScale_default(self):
        return 'Linear'

    def _ColorScale_changed(self):
        if self.ColorScale == 'Linear':
            self.plotdata.set_data('imagedata', self.lindata)
            self.val_max = float(self.lindata[30:-30, 1000:2000].max())
            self.specplot.request_redraw()
        if self.ColorScale == 'Sqrt':
            self.plotdata.set_data('imagedata', self.sqrtdata)
            self.val_min = 0.
            self.val_max = float(self.sqrtdata[30:-30, 1000:2000].max())
            self.specplot.request_redraw()
        if self.ColorScale == 'Log':
            self.plotdata.set_data('imagedata', self.logdata)
            self.val_max = float(self.logdata[30:-30, 1000:2000].max())
            self.val_min = 0.
            self.specplot.request_redraw()
        return

    def _colormaps_name_default(self):
        return 'gray'

    def _colormaps_name_changed(self):
        print("Selected colormap: {}".format(self.colormaps_name))
        clr_range = self.my_plot.color_mapper.range
        self.my_plot.color_mapper = \
            color_map_name_dict[self.colormaps_name](clr_range)

    def _ShowContin_fired(self):
        self.model = self.model.sort_index()
        self._build_model_plot()
        self.edit_traits(view='ContinView')

    def _ShowFitran_fired(self):
        A = SetFitRange(self.Spectrum)
        A.edit_traits()
        theidx = (self.Spectrum.transition, self.Spectrum.lines_sel, 'Contin')
        # print self.fitranges
        if A:  # and (theidx[0] in self.model.index.levels[0]):
            if not theidx in self.model.index:
                self.model.set_value(theidx, 'Ampl', 0.0)
            # if len(self.model.loc[self.Spectrum.transition].loc[self.Spectrum.lines_sel]) == 1:
            #     self.model.loc[
            #         (self.Spectrum.transition, self.Spectrum.lines_sel, 'Comp1'),
            #         ['Pos', 'Sigma', 'Ampl', 'Identifier']
            #     ] = [0, 0, 0, 'a']
            self.model.set_value(theidx, 'Fitranges', self.fitranges)


    def _unselect_all_fired(self):
        """Clears all selections."""
        self.posplot[0].index.metadata['selections'] = []
        self.posplot[0].value.metadata['selections'] = []


    def _set_label_fired(self):
        self.model = self.model.sort_index()
        self._build_model_plot()
        y_mask = self.posplot[0].value.metadata.get('selections', [])
        x_mask = self.posplot[0].index.metadata.get('selections', [])
        self.all_labels = self.model.drop('Dummy', level=0)\
            .drop('Contin', level=2)['Identifier'].unique().tolist()
        do_it = self.edit_traits(view='ReassignView')
        print('Label to set: {0}'.format(self.the_label))
        # print do_it
        if self.apply_to_all_transitions is True:
            transits = self.Spectrum.model.index.levels[0].tolist()
        else:
            transits = [self.transition]
        model = self.model.copy()#\
        #    .set_index('Identifier', append=True, drop=False)\
        #    .reset_index('Component', drop=False)
        the_index = model.loc[self.transition]\
            .drop('Contin', level=1).index[x_mask]
        for t in transits:
            for ind in the_index:
                model = model.set_value(
                    (t, ind[0], ind[1]), 'Identifier', self.the_label
                )
        # TODO: I think this is the source for my problems. Why change index
        # here??
        self.model = model#.set_index('Component', append=True, drop=True)\
        #     .reset_index('Identifier', drop=True)
        self._build_model_plot()
        self.ContCont.request_redraw()
        self.Posplot.request_redraw()
        self.apply_to_all_transitions = False


    def _remove_comp_fired(self):
        if self.apply_to_all_transitions is True:
            transits = self.Spectrum.model.index.levels[0].tolist()
        else:
            transits = [self.transition]
        y_mask = self.posplot[0].value.metadata.get('selections', [])
        x_mask = self.posplot[0].index.metadata.get('selections', [])
        model = self.model.copy()\
            .set_index('Identifier', append=True, drop=True)\
            .reset_index('Component', drop=False)
        the_index = model.loc[self.transition]\
            .drop('Contin', level=1).index[x_mask]
        for t in transits:
            for ind in the_index:
                model.drop((t, ind[0], ind[1]), inplace=True)
        self.model = model.set_index('Component', append=True)\
            .reset_index('Identifier')
        self._unselect_all_fired()
        self._build_model_plot()
        self.ContCont.request_redraw()
        self.Posplot.request_redraw()
        self.apply_to_all_transitions = False


