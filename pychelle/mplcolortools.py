#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc


def make_colormap(seq, name='CustomMap'):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mc.LinearSegmentedColormap(name, cdict)


def diverge_map(high=(0.565, 0.392, 0.173), low=(0.094, 0.310, 0.635),
                mid='black', name='CustomMap'):
    '''
    low and high are colors that will be used for the two
    ends of the spectrum. they can be either color strings
    or rgb color tuples
    '''
    c = mc.ColorConverter().to_rgb
    if isinstance(low, str): low = c(low)
    if isinstance(high, str): high = c(high)
    return make_colormap([low, c(mid), 0.5, c(mid), high], name=name)


def add_diverging_cmap(high='orange', mid='black', low='blue', name='mymap'):
    """ Creates and adds a diverging colormap  from three specified colors.
    Any color specification understood by matplotlib.colors.ColorConverter
    are valid.
    """
    dvrgmap = diverge_map(high=high, mid=mid, low=low, name=name)

    return dvrgmap

#c = mcolors.ColorConverter().to_rgb
#rvb = make_colormap(
#    [c('red'), c('violet'), 0.33, c('violet'), c('blue'), 0.66, c('blue')])
#N = 1000
#array_dg = np.random.uniform(0, 10, size=(N, 2))
#colors = np.random.uniform(-2, 2, size=(N,))
#plt.scatter(array_dg[:, 0], array_dg[:, 1], c=colors, cmap=rvb)
#plt.colorbar()
#plt.show()
