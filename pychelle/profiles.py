#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy as sp

def lorentz(x, pos, hwhm, ampl):
    """ Lorentz-Cauchy function expressed in terms of x0, gamma and height.

    Parameters
    ----------
    x : numpy.ndarray
        Wavelength array
    pos : float
        Wavelength of centroid.
    hwhm : float
        gamma, or half width half maximum.
    ampl : float
        Height of function peak.

    Returns
    -------
    lor : numpy.ndarray
    """
    lor = ampl * hwhm ** 2 / (4 * (x - pos) ** 2 + hwhm ** 2)
    lor = ampl * hwhm ** 2. / (( x - pos ) ** 2 + hwhm ** 2)
    return lor

# Gauss:
def gauss(x, pos, sig, ampl=1.):
    """ Gaussian distribution expressed in terms of
    centroid position, sigma and peak amplitude.

    Parameters
    ----------
    x : numpy.ndarray
        Wavelength array.
    pos : float
        Wavelength of centroid.
    sig : float
        Sigma of distribution.
    ampl : float
        Height of function peak.

    Returns
    -------
    gau : numpy.ndarray
    """
    gau = ampl * sp.exp(-0.5 * ((x - pos) / sig) ** 2)
    return gau

# Voigt, from Tepper Garcia 2006:
# FIXME: This whole thing depends on knowledge of resonator strengths and which
# atom/molecule we are dealing with. Overkill?
def _get_transition_info(transition='Halpha'):
    """Still need to figure out if I am going to implement this.
    If so, as far as possible build the design on that from vpfit...?
    Call: f, gamma = _get_transition_info(transition)
    """
    resstrengths={'Halpha': 0.695800, 'Hbeta': 0.121800, 'Hgamma': 0.044370}
    gammas = {'Halpha': 6.465e7, 'Hbeta': 2.062e7, 'Hgamma': 9.425e6}
    f = resstrengths[transition]
    g = gammas[transition]
    wl = wavlens[transition]
    return f, g


def _H(a, u):
    # The H(a, x) function of Tepper Garcia 2006
    P  = u**2
    H0 = np.e**(-(u**2))
    Q  = 1.5/u**2
    H  = H0-(a/np.sqrt(np.pi)/P*(H0*H0*(4*P*P+7*P+4+Q)-Q-1))
    print(P, H0, Q)
    return H


def voigt(x, pos, N, b, transition='Halpha'):
    # FIXME: Make me finished!
    """ This one should hopefully calculate the voigt function.
    The output is a multiplicative array that should be applied to the
    continuum.
    """
    #    CGS units:
    cgsc  = 2.998e10
    cgsme = 9.1095e-28
    cgse  = 4.8032e-10
    # Get transition info.
    f, g = _get_transition_info(transition)
    a = x * g / (4 * sp.pi * b)
    Ca = cgse**2/(cgsme * cgsc)*np.sqrt(np.pi**3)/np.pi*f*pos/b
    vp = 1.
    #u = -c / b * ()

    return vp

