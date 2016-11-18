# rotating the position angle
# to find the maximum velocity gradient 
# which is defined to be 

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
from photutils import aperture_photometry

# add the class file to the PYTHONPATH
sys.path.append('/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/Class')

# Need a function to do the extraction along the KA

def extract(d_aper,
            r_aper,
            pa,
            v_field,
            xcen,
            ycen,
            pix_scale):

    """
    Def: By taking a distance between apertures, and an aperture radius
    measurement, extract velocity measurement along a given position angle
    """

    # define the positions array to store aperture centers in

    positions = []

    # define the x and y dimensions

    xdim = v_field.shape[0] - 2

    ydim = v_field.shape[1] - 2

    # the increment sizes depend on the position angle

    x_inc = d_aper * abs(np.sin((np.pi / 2.0) - pa))

    y_inc = d_aper * abs(np.cos((np.pi / 2.0) - pa))

    # now find the sequence of aperture centres up until the boundaries
    # this is tricky - depending on the PA need to increase and decrease
    # both x and y together, or increase one and decrease the other

    # statements for the maximum likelihood position angle

    if 0 < pa < np.pi / 2.0 or np.pi < pa < 3 * np.pi / 2.0:

        # print 'Top Right and Bottom Left'

        # need to increase x and decrease y and vice versa

        new_x = xcen + x_inc

        new_y = ycen - y_inc

        # while loop until xdim is breached or 0 is breached for y

        while new_x < xdim and new_y > 2:

            # append aperture centre to the positions array

            positions.append((new_y, new_x))

            new_x += x_inc

            new_y -= y_inc

            # print new_x, new_y

        # starting from the left so need to reverse list direction
        # and append the central point

        positions = positions[::-1]

        positions.append((ycen, xcen))

        # now go in the other direction

        new_x = xcen - x_inc

        new_y = ycen + y_inc

        # while loop until xdim is breached or 0 is breached for y

        while new_x > 2 and new_y < ydim:

            # append aperture centre to the positions array

            positions.append((new_y, new_x))

            new_x -= x_inc

            new_y += y_inc

        if np.pi < pa < 3 * np.pi / 2.0:

            positions = positions[::-1]

            # print new_x, new_y

    # deal with the other cases of position angle

    else:

        # print 'Top Left and Bottom Right'

        # need to increase x and increase y and vice versa

        new_x = xcen - x_inc

        new_y = ycen - y_inc

        # while loop until xdim is 2 or ydim is 2

        while new_x > 2 and new_y > 2:

            # append aperture centre to the positions array

            positions.append((new_y, new_x))

            new_x -= x_inc

            new_y -= y_inc

        # starting from the left so need to reverse list direction
        # and append the central point

        positions = positions[::-1]

        positions.append((ycen, xcen))

        # now go in the other direction

        new_x = xcen + x_inc

        new_y = ycen + y_inc

        # while loop until xdim is breached or ydim is breached

        while new_x < xdim and new_y < ydim:

            # append aperture centre to the positions array

            positions.append((new_y, new_x))

            new_x += x_inc

            new_y += y_inc

        if 3 * np.pi / 2.0 < pa < 2 * np.pi:

            positions = positions[::-1]

    # now do the aperture photometry with these positions

    # construct the x_axis for the aperture extraction plot

    x_array = []

    for entry in positions:

        x_array.append(entry[1])

    x_array = np.array(x_array) - xcen

    # print x_array

    x_index = np.where(x_array == 0.0)[0]

    x = np.linspace(-1. * d_aper * x_index,
                        d_aper * (len(x_array) - x_index - 1),
                        num=len(x_array))

    x = x * pix_scale

    # pixel area

    pixel_area = np.pi * r_aper * r_aper

    # the max likelihood extraction parameters

    apertures = CircularAperture(positions, r=r_aper)

    real_phot_table = aperture_photometry(v_field, apertures)

    real_velocity_values = real_phot_table['aperture_sum'].data / pixel_area

#    print 'RT PA REAL VELOCITY VALUES: %s' % real_velocity_values
#    print 'RT PA X: %s' % x
#    print 'POSITIONS: %s' % positions
#    print 'xcen and ycen: %s %s' % (xcen, ycen)
#    fig, ax = plt.subplots(1, 1, figsize=(10,10))
#    ax.imshow(v_field, interpolation='nearest')
#    apertures.plot(ax=ax)
#    plt.show()
#    plt.close('all')

#    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
#    ax.scatter(x,
#            real_velocity_values,
#            color='red',
#            label='max_model')
#    # ax.legend(prop={'size':10})
#    ax.set_xlim(-1.5, 1.5)
#    # ax.legend(prop={'size':5}, loc=1)
#    ax.axhline(0, color='silver', ls='-.')
#    ax.axvline(0, color='silver', ls='-.')
#    ax.set_title('Real Velocity')
#    ax.set_ylabel('velocity (kms$^{-1}$)')
#    ax.set_xlabel('arcsec')
#    plt.show()
#    plt.close('all')
#    fig, ax = plt.subplots(1, 1, figsize=(10,10))
#    ax.imshow(v_field)
#    apertures.plot(ax, color='black')
#    plt.show()
#    plt.close('all')

    return [real_velocity_values, x]

def find_first_valid_entry(velocity_array):

    """
    Def:
    Takes the output from extract and returns the index of the first
    and last values which are not np.nan
    """

    # need at least one valid entry for this to work

    if np.isfinite(velocity_array).any():

        i = 0

        while np.isnan(velocity_array[i]):

            i += 1

        start_index = copy(i)

        # need to have a try here just incase the velocity array
        # reaches the last entry in the array

        j = 0

        while np.isnan(velocity_array[::-1][j]):

            j += 1

        end_index = len(velocity_array) - 1 - j

        return [start_index, end_index]

    else:

        return [0, 0]

# First have a simple function which measures the PA classifier

def pa_statistic(d_aper,
                 r_aper,
                 pa,
                 v_field,
                 xcen,
                 ycen,
                 pix_scale):

    """
    Def:
    Return the statistic for that particular choice of PA
    """
    # first find the velocity values

    vel_array, x = extract(d_aper,
                           r_aper,
                           pa,
                           v_field,
                           xcen,
                           ycen,
                           pix_scale)

    # next determine the start and end indices

    start_i, end_i = find_first_valid_entry(vel_array)

    # Two different cases - 1) if end_i - start_i is less than
    # 6 just find the maximum subtract minimum
    # The *better* way is to take the median of the first three and
    # last three and sum these

    # added complication that some PAs will draw a line from negative to
    # negative. Or positive to positive. So the statistic must be computed
    # with a subtraction

    if end_i - start_i <= 4:

        stat = abs(vel_array[start_i] - (vel_array[end_i]))

    else:

        stat = abs(np.nanmean(vel_array[start_i:start_i + 2]) -
                   np.nanmean(vel_array[end_i - 2:end_i]))

    return stat

def rot_pa(d_aper,
           r_aper,
           v_field,
           xcen,
           ycen,
           pix_scale):

    """
    Def:
    Tie the above methods together using an array of 0 to np.pi 
    in increments of 0.01 radians.
    """

    pa_array = np.arange(0, np.pi, 0.01)

    stat_array = []

    for entry in pa_array:

        stat_array.append(pa_statistic(d_aper,
                                       r_aper,
                                       entry,
                                       v_field,
                                       xcen,
                                       ycen,
                                       pix_scale))

    stat_array = np.array(stat_array)

#    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
#    ax.plot(pa_array, stat_array)
#    plt.show()
#    plt.close('all')
    print pa_array[np.nanargmax(stat_array)]
    return [pa_array[np.nanargmax(stat_array)],
            pa_array,
            stat_array]
