#!/usr/bin/env python
# encoding: utf-8

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
    CompNum = Int
    CompNum = 1
    Components = Dict
    Locks = Dict
    CompoList = List()
    ArbitraryFactor = Range(0., .5, .1)
    x = Array
    mod_x = Array
    Feedback = Str
    Sigma = Range(.1, 200.)
    Centr = Range(-100., 100., 0.)
    Heigh = Range(0., 20000.)
    # Define vars to regulate whether the above vars are locked
    #    down in the GUI:
    LockSigma = Bool()
    LockCentr = Bool()
    LockHeigh = Bool()
    LockConti = Bool()
    continuum_estimate = Range(0., 2000.)
    plots = {}
    plotrange = ()
    resplot = Instance(Plot)
    resplot = Plot()
    Model = Array
    Resids = Property(Array, depends_on='Model')
    y = {}

    def _Resids_default(self):
        intmod = sp.interp(self.x, self.mod_x, self.Model)
        resids = (self.indata - intmod) / self.errs
        return resids

    def _get_Resids(self):
        intmod = sp.interp(self.x, self.mod_x, self.Model)
        resids = (self.indata - intmod) / self.errs
        return resids

    def _Components_default(self):
        return {'Contin': self.continuum_estimate,
                'Comp1': [0., .1, 0., 'a']}  # Center, Sigma, Height

    def _Locks_default(self):
        return {'Comp1': [False, False, False, False]}

    def _CompoList_default(self):
        return ['Comp1']

    def _y_default(self):
        return {}

    set_vals = Button(
        label='Set parameters',
        desc="Set component parameters to current values"
    )
    # Define buttons for interface:
    add_profile = Button(label='Add component')
    remove_profile = Button(label='Remove latest')
    Go_Button = Button(label='Show model')
    select = Str

    def build_plot(self):
        global plotdata
        onearray = Array
        onearray = sp.ones(self.indata.shape[0])
        minuses = onearray * (-1.)
        # Define index array for fit function:
        self.mod_x = sp.arange(
            self.line_center - 50.,
            self.line_center + 50., .01
        )
        self.Model = sp.zeros(self.mod_x.shape[0])
        # Establish continuum array in a way that opens for other, more
        #   elaborate continua.
        self.contarray = sp.ones(self.mod_x.shape[0]) * \
                self.Components['Contin']
        self.y = {}

        for comp in self.CompoList:
            self.y[comp] = stats.norm.pdf(
                self.mod_x,
                self.Components[comp][0] + self.line_center,
                self.Components[comp][1]
            ) * self.Components[comp][1] * sp.sqrt(2. * sp.pi) * \
                    self.Components[comp][2]
        self.Model = self.contarray + self.y[self.select]
        broca = BroadcasterTool()
        # Define the part of the data to show in initial view:
        plotrange = sp.where((self.x > self.line_center - 20) &
                             (self.x < self.line_center + 20))
        # Define the y axis max value in initial view (can be panned/zoomed):
        maxval = float(self.indata[plotrange].max() * 1.1)
        minval = maxval / 15.
        maxerr = self.errs[plotrange].max() * 1.3
        resmin = max(self.Resids[plotrange].max(), 5.) * 1.2
        cenx = sp.array([self.line_center, self.line_center])
        ceny = sp.array([-minval, maxval])
        cenz = sp.array([-maxval, maxval])
        # Build plot of data and model
        plotdata = ArrayPlotData(
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
        )
        for comp in self.CompoList:
            plotdata.set_data(comp, self.y[comp])
        olplot = GridContainer(
            shape=(2, 1), padding=10,
            fill_padding=True,
            bgcolor='transparent',
            spacing=(5, 10)
        )
        plot = Plot(plotdata)
        plot.y_axis.title = 'Flux density'
        resplot = Plot(plotdata, tick_visible=True, y_auto=True)
        resplot.x_axis.title = u'Wavelength [Å]'
        resplot.y_axis.title = u'Residuals/std. err.'

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
        # There may be an arbitrary number of gaussian components, so:
        for comp in self.CompoList:
            self.comprenders.append(
                plot.plot(
                    ('xs', comp),
                    type='line',
                    #color='auto',
                    color=tuple(COLOR_PALETTE[self.CompNum]),
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
        plot.x_axis.visible = False

        # Set ranges to change automatically when plot values change.
        plot.value_range.low_setting,\
            plot.value_range.high_setting = (-minval, maxval)
        plot.index_range.low_setting,\
            plot.index_range.high_setting = (self.line_center - 20.,
                                             self.line_center + 20.)
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

    # Select component 1 by default:
    def __init__(self, wavlens, indata, inerrs, linecen, spec2d=None):
        self.line_center = linecen
        # Define index array for data:
        self.x = wavlens
        self.indata = indata
        self.errs = inerrs
        self.build_plot()

    def _select_default(self):
        return 'Comp1'

    ### =======================================================================
    #     Reactive functions: What happens when buttons are pressed, parameters
    #     are changes etc.
    ### =======================================================================

    # Add component to model
    def _add_profile_fired(self):  # FIXME
        global plotdata
        self.CompNum += 1
        Name = 'Comp' + str(self.CompNum)
        self.CompoList.append(Name)
        print "Added component nr. " + Name
        #self.Components[Name]=[0., 1., 0.] # Add booleans for locked or not?
        self.Components[Name] = [0., .1, 0., chr(self.CompNum+96)]
        self.Locks[Name] = [False, False, False, False]
        self.select = Name
        # And the plotting part:
        #    Add y array for this component.
        self.y[self.select] = stats.norm.pdf(
            self.mod_x,
            self.Centr + self.line_center,
            self.Sigma) * self.Sigma * sp.sqrt(2. * sp.pi) * self.Heigh
        plotdata[self.select] = self.y[self.select]
        render = self.plot.plot(('xs', self.select), type='line',
                                line_style='dash',
                                color=tuple(COLOR_PALETTE[self.CompNum]),
                                name=Name)
        self.plots[self.select] = render
        self.legend.plots = self.plots
        return

    # Remove the last added component.
    def _remove_profile_fired(self):
        global plotdata
        if self.CompNum > 1:
            oldName = 'Comp' + str(self.CompNum)
            newName = 'Comp' + str(self.CompNum - 1)
            self.plot.delplot(oldName)
            plotdata.del_data(oldName)
            del self.y[oldName]
            del self.plots[oldName]
            del self.Components[oldName]
            del self.Locks[oldName]
            self.select = newName
            print 'Removed component nr. ' + str(self.CompNum)
            self.legend.plots = self.plots
            self.CompoList.pop()
            self.CompNum -= 1
        else:
            print 'No more components to remove'

    def _Go_Button_fired(self):
        print '''This function is yet to be implemented. \n \
                Maybe this button will be removed completely in favor of \
                another way of doing things. For now, use the OK button and \
                manually import the model to the fitting software of your \
                choice.'''
        print self.Components
        return

    # Define what to do when a new component is selected.
    def _select_changed(self):
        # First, show the values of current component in sliders!
        self.Centr = self.Components[self.select][0]
        self.Sigma = self.Components[self.select][1]
        self.Heigh = self.Components[self.select][2]
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
        self.Components['Contin'] = self.continuum_estimate
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

    plwin = Instance(GridContainer)
    view = View(
        Group(
            Group(
                Item('plwin', editor=ComponentEditor(),
                     show_label=False, springy=True),
                Group(
                    Group(
                        Item('Centr', label='Center',
                             enabled_when='LockCentr==False'),
                        Item('Sigma', label='Sigma',
                             enabled_when='LockSigma==False'),
                        Item('Heigh', label=u'Strength ',
                             enabled_when='LockHeigh==False'),
                        springy=True,
                        show_border=False,
                        orientation='vertical'),
                    Group(
                        Item('LockCentr', label='Lock'),
                        Item('LockSigma', label='Lock'),
                        Item('LockHeigh', label='Lock'),
                        orientation='vertical'),
                    orientation='horizontal',
                    show_border=True,
                    label='Component parameters'),

                HGroup(
                    Item('continuum_estimate',
                         enabled_when='LockConti==False',
                         label='Contin.  ',
                         springy=True
                         ),
                    Item('LockConti', label='Lock'),
                    show_border=True,
                    springy=True
                ),
                show_border=True),
            Group(Item('add_profile'),
                  Item('remove_profile'),
                  Item('Go_Button'),
                  Item('Feedback', style='readonly'),
                  Item('Feedback', style='readonly'),
                  Item(name='select', editor=EnumEditor(name='CompoList'),
                       style='custom'),
                  orientation='vertical',
                  show_labels=False,
                  show_border=True),
            orientation='horizontal'),
        resizable=True,
        height=700, width=1000,  # ),
        buttons=[UndoButton, ApplyButton, CancelButton, OKButton],
        close_result=True,
        kind='livemodal',
        title="Grism - line profile editor")

    def import_model(self):
        global plotdata
        #print '    '
        #print 'This is the model importing method of ProfileEditor: '
        self.CompoList = sorted(self.Components.keys())[:-1]
        self.CompNum = len(self.CompoList)
        for com in self.CompoList:
            self.Locks[com] = [False] * 4
        #print self.Components
        self.continuum_estimate = self.Components['Contin']
        self._select_changed()
        self.build_plot()
        self.update_plot()
        print '    '


    def update_plot(self):
        self.y[self.select] = stats.norm.pdf(
            self.mod_x,
            self.Centr + self.line_center,
            self.Sigma) * self.Sigma * sp.sqrt(2. * sp.pi) * self.Heigh
        ys = sp.asarray(self.y.values()).sum(0)
        self.contarray = sp.ones(self.mod_x.shape[0]) * self.continuum_estimate
        self.Model = self.contarray + ys
        plotdata.set_data('cont', self.contarray)
        plotdata.set_data(self.select, self.y[self.select])
        plotdata.set_data('model', self.Model)
        plotdata.set_data('Residuals', self.Resids)

    @on_trait_change('Resids')
    def update_resid_window(self):
        resmin = max(self.Resids[self.plotrange].max(), 5.) * 1.2
        self.resplot.value_range.low_setting,\
            self.resplot.value_range.high_setting = (-resmin, resmin)
        self.resplot.request_redraw()
