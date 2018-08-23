#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Docstring for the whole thing - yayze?
Dependencies: Numpy, SciPy, Traits, TraitsUI, Chaco, PyFits, Pandas.
"""

# IMPORTS:
# FITS files I/O
#import astropy.io.fits as pf
# First, set gui toolkit (Should maybe test for availablity?)
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt4'
# Advanced, labelled data structures
import pandas as pd
# Traits - for the model handling and GUI
# from traits.api import HasTraits, Float, List, Dict, Bool, Array, DelegatesTo,\
#         PrototypedFrom, Instance, Button, Str, Range, Enum, Int, Property, \
#         on_trait_change, Any
# from traitsui.api import View, Group, HGroup, VGroup, Item, Spring, EnumEditor
# from traitsui.menu import OKButton,  CancelButton, RevertButton, LiveButtons,\
#     ModalButtons, UndoButton, ApplyButton
# # Enable & chaco - for plotting in the model editor GUI
# from enable.component_editor import ComponentEditor
# from chaco.example_support import COLOR_PALETTE
# from chaco.chaco_plot_editor import ChacoPlotItem
# from chaco.api import ArrayPlotData, PlotLabel, Plot, HPlotContainer, \
#         GridContainer, ImageData, bone, gist_heat, gist_rainbow, DataRange1D, \
#         ScatterInspectorOverlay, ColorBar, LinearMapper, Legend, \
#         color_map_name_dict, KeySpec, create_line_plot, LinePlot, \
#         add_default_axes, add_default_grids, OverlayPlotContainer
# from chaco.tools.api import ScatterInspector, ZoomTool, PanTool, \
#         BroadcasterTool, LegendTool, RangeSelection, RangeSelectionOverlay
from .lpbuilder import ProfileEditor, load_profile

# TODO: Create redshift-estimator class/view, with a simple list of
# HI-transitions and maybe a few other strong features as a guide. Just set
# it precisely enough that there will typically be a couple of lines
# available to choose from when adding a new transition.
#
# TODO: Split up in multiple files! One for each of the major classes, one for
# helper functions (where to leave lines_srs?). Plus, I suppose, separate files
# for fitting backends. These should be

from .helper_functions import air_to_vacuum, vacuum_to_air, load_lines_series
from .spectrum2d import Spectrum2D
from .show2dspec import SetFitRange, view_2d, Show2DSpec, load_2d, lines_srs, \
    fit_transition_to_other
from .helper_functions import _extract_1d, wl_to_v, v_to_wl, \
    transition_from_existing, set_model, fit_with_sherpa \

# lines_srs = load_lines_series()

###============================================================================
#             What to do if script is called from command line rather than
#             imported from an external python module or shell.
###============================================================================

if __name__ == '__main__':
    #load_lines_dict()
    My2dSpec = load_2d('Haro11-B-Combined-VIS.fits')
    print('My2dSpec object created')
    Base_Ui = view_2d(My2dSpec, Center=6699.5)
    print(My2dSpec.model)
    print(My2dSpec.line_spectra)
