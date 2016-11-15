# /usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import rc, rcParams, font_manager, cm
from matplotlib.colors import colorConverter
import scipy as sp

pats = 16   # Plot Axis Title Size
sfont = 14  # Small font for e.g. legends
csz = 5     # Error bar cap size
leg_prop = font_manager.FontProperties(size=sfont)
# Set the ColorBrewer 'Set1' colr set as default colors. Because prettier.
colorindices = sp.linspace(0, 256, 9)
# CB_color_cycle = ('#e41a1c', '#377eb8', '#4daf4a',
#                   '#984ea3', '#ff7f00', '#dede00', # '#ffff33',
#                   '#a65628', '#f781bf', '#999999')
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

CB_cycle2 = ()

CUDcolorD = {}  # From http://python-cudtools.googlecode.com/hg/cudtools.py
CUDcolorD['Black']          = (0, 0, 0)           #  cool
CUDcolorD['Vermillion']     = (0.8, 0.4, 0)       # warm
CUDcolorD['Blue']           = (0, 0.45, 0.7)      #  cool
CUDcolorD['Orange']         = (0.9, 0.6, 0)       # warm
CUDcolorD['Bluish Green']   = (0, 0.6, 0.5)       #  cool
CUDcolorD['Reddish Purple'] = (0.8, 0.6, 0.7)     # warm
CUDcolorD['Sky Blue']       = (0.35, 0.7, 0.9)    #  cool
CUDcolorD['Yellow']         = (0.95, 0.9, 0.25)   # warm
CUDcolorOrderL=['Vermillion', 'Blue',
                'Orange', 'Bluish Green', 'Reddish Purple',
                'Sky Blue', 'Yellow', 'Black']
CUDCycle = ((0.8, 0.4, 0),
            (0, 0.45, 0.7),
            (0.9, 0.6, 0),
            (0, 0.6, 0.5),
            (0.8, 0.6, 0.7),
            (0.35, 0.7, 0.9),
            (0.95, 0.9, 0.25))
# plt.rc('axes', color_cycle=[cm.Accent(int(i)) for i in colorindices])
Set1colors = [cm.Set1(int(i)) for i in colorindices]
# Change the tick fontsize for all subsequent figures
rc('xtick', labelsize=11)
rc('ytick', labelsize=11)
rc('lines', linewidth = 1.5)
rc('font', family='serif')
rc('font', size = 12)
#rc('text', usetex=True)
rcParams['axes.formatter.limits'] = (-4,4) # Exponential tick labels outside this log range

def maskplot(index, data, mask, color='black', styles=['solid', 'dotted']):
    """ Creates a plot with masked-out intervals in dotted or custom line style,
    non-masked segments in dashed or custom line style, all in given color.
    Built from the example at
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    """
    points = sp.array([index, data]).T.reshape(-1, 1, 2)
    segments = sp.concatenate([points[:-1], points[1:]], axis=1)
    mask = mask.reshape(-1, 1)
    mask = (sp.ones_like(mask) - mask).astype(bool)
    mask = sp.hstack((mask, mask))
    mask = sp.dstack((mask, mask))
    mask = mask[1:]
    #print segments.shape, mask.shape
    mask1 = sp.ma.masked_where(mask, segments)
    mask2 = sp.ma.masked_where(
        (-mask.astype(float)+1.).astype(bool), segments)

    ls1 = LineCollection(
        mask1,
        colors=color,
        linestyles=styles[0])

    ls2 = LineCollection(
        mask2,
        colors=colorConverter.to_rgba(color),
        linestyles=styles[1])

    return ls1, ls2


def fill_between_steps(x, y1, y2=0, h_align='mid', ax=None, **kwargs):
    ''' Fills a hole in matplotlib: fill_between for step plots.

    Parameters :
    ------------

    x : array-like
        Array/vector of index values. These are assumed to be equally-spaced.
        If not, the result will probably look weird...
    y1 : array-like
        Array/vector of values to be filled under.
    y2 : array-Like
        Array/vector or bottom values for filled area. Default is 0.

    **kwargs will be passed to the matplotlib fill_between() function.

    '''
    # If no Axes opject given, grab the current one:
    if ax is None:
        ax = plt.gca()
    # First, duplicate the x values
    xx = x.repeat(2)[1:]
    # Now: the average x binwidth
    print x
    xstep = sp.repeat((x[1:] - x[:-1]), 2)
    xstep = sp.concatenate(([xstep[0]], xstep, [xstep[-1]]))
    # Now: add one step at end of row.
    xx = sp.append(xx, xx.max() + xstep[-1])

    # Make it possible to chenge step alignment.
    if h_align == 'mid':
        xx -= xstep / 2.
    elif h_align == 'right':
        xx -= xstep

    # Also, duplicate each y coordinate in both arrays
    y1 = y1.repeat(2)#[:-1]
    if type(y2) == sp.ndarray:
        y2 = y2.repeat(2)#[:-1]

    # now to the plotting part:
    ax.fill_between(xx, y1, y2=y2, **kwargs)

    return ax

