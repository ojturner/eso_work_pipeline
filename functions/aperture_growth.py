# Houses the methods required to analytically
# determine the half-light radii of a galaxy
# taking into account any neighbours which
# must be masked (in a very simple way)


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


# add the functions folder to the PYTHONPATH
sys.path.append('/disk2/turner/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/functions')

import flatfield_cube as f_f
import psf_blurring as psf
import twod_gaussian as g2d
import rotate_pa as rt_pa

def find_aperture_parameters(stamp,
                             x_low,
                             x_high,
                             y_low,
                             y_high):

    """
    Def:
    Stamp is a fits file containing an extracted stamp around the
    galaxy in question. It is the fits file itself, so will be opened 
    as a fits file. I'm also assuming here that these are the fits files
    returned from Victoria - so I'm opening the first extension.
    """

    image_data = fits.open(stamp)[1].data

    ## galfit plotting

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.imshow(image_data, vmin=0, vmax=5)

    ax.minorticks_on()

    ax.grid(b=True, which='major', color='b', linestyle='-')

    ax.grid(b=True, which='minor', color='r', linestyle='--')

    # plt.show()

    plt.close('all')

    # play the same game as in the reshape script with accepting
    # raw input for the pixel boundaries for gaussian fitting

    cut_data = image_data[x_low:x_high, y_low:y_high]

    fit_data, fit_params = g2d.fit_gaussian(cut_data)

    ## galfit plotting

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.imshow(cut_data, vmin=0, vmax=5)

    y_full, x_full = np.indices(cut_data.shape)
    ax.contour(x_full,
               y_full,
               fit_data,
               4,
               ls='solid',
               colors='black')

    ax.minorticks_on()

    ax.grid(b=True, which='major', color='b', linestyle='-')

    ax.grid(b=True, which='minor', color='r', linestyle='--')

    # plt.show()

    # now simply return the ratio of widths, central position, 
    # and the rotation angle

    central_x = fit_params[3]

    central_y = fit_params[2]

    positions = [(central_y,central_x)]

    width_a = fit_params[4]

    width_b = fit_params[5]

    if width_a < width_b:

        axis_ratio = width_a / width_b

        theta = (-1.*fit_params[6]) - np.pi / 2.0

    else:

        axis_ratio = width_b / width_a

        theta = (-1.*fit_params[6])


    # set the semi and major axis

    a = 3
    b = a * axis_ratio

    aperture = EllipticalAperture(positions, a, b, theta)

    aperture.plot(ax, color='yellow')

    #plt.show()

    #plt.close('all')

    # now actually do the aperture photometry, might want this in a different
    # method

    a = np.arange(0, 50, 0.05)

    b = axis_ratio * a

    sum_array = []

    for major, minor in zip(a,b):

        aperture = EllipticalAperture(positions, major, minor, theta)

        #aperture.plot(ax, color='yellow')

        phot_table = aperture_photometry(cut_data, aperture)

        phot_sum = phot_table['aperture_sum']

        sum_array.append(phot_sum)

    #plt.show()

    plt.close('all')

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    ax.plot(a, sum_array)

    ax.scatter(a, sum_array)

    #plt.show()

    plt.close('all')

    # now determine the half-light radius from this experiment
    # take the median from the index corresponding to a major 
    # axis size of 30 onwards

    t_index = np.argmax(sum_array)

    sum_limit = sum_array[t_index]

    # use half of this maximum to determine 
    # the half light radius

    half_limit = sum_limit / 2.0

    half_index = np.nanargmin(abs(half_limit - sum_array))

    # also return the 90 percent limit

    n_limit = sum_limit * 0.9

    n_index = np.nanargmin(abs(n_limit - sum_array))

    return {'cut_data' : cut_data,
            'fit_data' : fit_data,
            'a_array' : a,
            'sum_array' : sum_array,
            'axis_ratio' : axis_ratio,
            'r_e_pixels' : a[half_index],
            'r_9_pixels' : a[n_index],
            'pa' : theta}