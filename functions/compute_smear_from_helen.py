# import the relevant modules
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyraf
import numpy.ma as ma
import pickle
from scipy.spatial import distance
from scipy import ndimage
from copy import copy
from lmfit import Model
from itertools import cycle as cycle
from lmfit.models import GaussianModel, PolynomialModel, ConstantModel
from scipy.optimize import minimize
from astropy.io import fits, ascii
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import poly1d
from sys import stdout
from photutils import CircularAperture
from photutils import EllipticalAperture
from photutils import aperture_photometry
import astropy.wcs.utils as autils
from astropy.wcs import WCS
from PIL import Image

def compute_velocity_smear_from_ratio(ratio_r_psf,
                                      velocity_3rd):
    
    # given the ratio of the size of the PSF to the
    # disk radius of the galaxy, compute the correction
    # factor for the velocity (at 3Rd, whether that is important or not)
    # analytic function for correction comes from H.Johnson

    r = np.arange(0,5.0, 0.001)

    # parameter values
    a_1 = 1.0
    b_1 = 0.18
    c_1 = 1.48
    d_1 = 1.0

    a_2 = 1.0
    b_2 = 0.18
    c_2 = 1.26
    d_2 = 0.88

    curve_1 = []

    curve_2 = []

    for entry in r:

        func = a_1 - (b_1 * np.exp(-c_1 * entry**d_1))

        curve_1.append(func)

    for entry in r:

        func = a_2 - (b_2 * np.exp(-c_2 * entry**d_2))

        curve_2.append(func)

    corr_indx = np.argmin(abs(ratio_r_psf - r))
    corr_factor_1 = curve_1[corr_indx]
    corr_factor_2 = curve_2[corr_indx]

    print 'VELOCITY CORRECTION AT 2.2: %s' %  (velocity_3rd / corr_factor_2)

    return velocity_3rd / corr_factor_2

def compute_velocity_smear_from_ratio_3(ratio_r_psf,
                                        velocity_3rd):
    
    # given the ratio of the size of the PSF to the
    # disk radius of the galaxy, compute the correction
    # factor for the velocity (at 3Rd, whether that is important or not)
    # analytic function for correction comes from H.Johnson

    r = np.arange(0,5.0, 0.001)

    # parameter values
    a_1 = 1.0
    b_1 = 0.18
    c_1 = 1.48
    d_1 = 1.0

    a_2 = 1.0
    b_2 = 0.18
    c_2 = 1.26
    d_2 = 0.88

    curve_1 = []

    curve_2 = []

    for entry in r:

        func = a_1 - (b_1 * np.exp(-c_1 * entry**d_1))

        curve_1.append(func)

    for entry in r:

        func = a_2 - (b_2 * np.exp(-c_2 * entry**d_2))

        curve_2.append(func)

    corr_indx = np.argmin(abs(ratio_r_psf - r))
    corr_factor_1 = curve_1[corr_indx]
    corr_factor_2 = curve_2[corr_indx]

    print 'VELOCITY CORRECTION AT 3.0: %s ' % (velocity_3rd / corr_factor_1)
    print 'VELOCITY CORRECTION AT 2.2: %s' %  (velocity_3rd / corr_factor_2)

    return velocity_3rd / corr_factor_1



#    fig, ax = plt.subplots(1, 1, figsize=(10,10))
#    ax.plot(r, curve_1)
#    ax.plot(r, curve_2, ls='--')
#    plt.show()
#    plt.close('all')

def compute_mean_sigma_smear_from_ratio(ratio_r_psf,
                                        sigma_mean,
                                        vel_bin=1):

    # exactly the same as above this time with a different 
    # analytic function and computing the sigma correction

    r = np.arange(0,5.0, 0.001)

    if vel_bin == 1:

        a = 1.00
        b = 11.50
        c = 4.65
        d = 0.20

    elif vel_bin == 2:

        a = 1.00
        b = 52.85
        c = 5.55
        d = 0.34

    elif vel_bin == 3:

        a = 1.00
        b = 8.74
        c = 3.15
        d = 0.77

    elif vel_bin == 4:

        a = 1.00
        b = 14.15
        c = 3.05
        d = 0.69

    curve = []

    for entry in r:

        func = a + (b * np.exp(-c * entry**d))

        curve.append(func)

    corr_indx = np.argmin(abs(ratio_r_psf - r))
    corr_factor = curve[corr_indx]

    print 'SIGMA MEAN CORRECTION FACTOR: %s' % corr_factor
    print 'SIGMA MEAN CORRECTED: %s' % (sigma_mean / corr_factor)

    return sigma_mean / corr_factor

#    fig, ax = plt.subplots(1, 1, figsize=(10,10))
#    ax.plot(r, curve)
#    plt.show()
#    plt.close('all')

def compute_outer_sigma_smear_from_ratio(ratio_r_psf,
                                         sigma_outer,
                                         vel_bin=1):

    # exactly the same as above this time with a different 
    # analytic function and computing the sigma correction

    r = np.arange(0,5.0, 0.001)

    if vel_bin == 1:

        a = 1.00
        b = 0.53
        c = 8.22
        d = 0.94

    elif vel_bin == 2:

        a = 1.00
        b = 6.98
        c = 7.07
        d = 0.52

    elif vel_bin == 3:

        a = 1.00
        b = 3.27
        c = 4.96
        d = 0.59

    elif vel_bin == 4:

        a = 1.00
        b = 2.06
        c = 3.67
        d = 0.70

    curve = []

    for entry in r:

        func = a + (b * np.exp(-c * entry**d))

        curve.append(func)

    corr_indx = np.argmin(abs(ratio_r_psf - r))
    corr_factor = curve[corr_indx]

    print 'SIGMA EDGES CORRECTION FACTOR: %s' % corr_factor
    print 'SIGMA EDGES CORRECTED: %s' % (sigma_outer / corr_factor)

    return sigma_outer / corr_factor

#    fig, ax = plt.subplots(1, 1, figsize=(10,10))
#    ax.plot(r, curve)
#    plt.show()
#    plt.close('all')

# note the velocity used is the 1d extrapolation to 3Rd
# along the dynamic position angle
#compute_velocity_smear_from_ratio(0.83, 53.8)
#compute_outer_sigma_smear_from_ratio(1.83, 85.98, 3)
#compute_mean_sigma_smear_from_ratio(1.83, 116.65 , 3)
