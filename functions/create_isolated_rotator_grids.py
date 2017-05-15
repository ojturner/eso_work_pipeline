# This class houses the methods which are relevant to manual additions to the
# ESO KMOS pipeline Mainly concentrating on two procedures - 1) pedestal
# readout column correction and 2) shifting and aligning sky and object
# images before data cube reconstruction


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
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import MaxNLocator
from numpy import poly1d
from sys import stdout
from matplotlib import rc
from photutils import CircularAperture
from photutils import EllipticalAperture
from photutils import aperture_photometry

# add the class file to the PYTHONPATH
sys.path.append('/disk2/turner/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/Class')

# add the functions folder to the PYTHONPATH
sys.path.append('/disk2/turner/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/functions')

import flatfield_cube as f_f
import psf_blurring as psf
import twod_gaussian as g2d
import rotate_pa as rt_pa
import aperture_growth as ap_growth
import make_table
import arctangent_1d as arc_mod
import oned_gaussian as one_d_g
import search_for_closest_sky as sky_search
import compute_smear_from_helen as dur_smear
import pa_calc

from cubeClass import cubeOps
from galPhysClass import galPhys
from vel_field_class import vel_field

def disk_function_fixed_inc_fixed(theta,
                                  xcen,
                                  ycen,
                                  inc,
                                  xpos,
                                  ypos):
    """
    Def: Function to calculate disk velocity given input values.
    Note that all angles must be given in radians
    """
    # unpack the parameters

    pa, rt, vasym = theta

    # look at the difference between central pixel and pixel
    # being modelled

    diff_x = (xcen - xpos) * 1.0

    diff_y = (ycen - ypos) * 1.0

    # print diff_x, diff_y

    # calculate the pixel angle

    if diff_y == 0 and diff_x != 0:

        pixel_angle = np.arctan(np.sign(diff_x)*np.inf)

        # print 'This is the pixel angle %s' % pixel_angle

    elif diff_y == 0 and diff_x == 0:

        # print 'In the middle'

        pixel_angle = 0.0

    else:

        # print 'computing pixel angle'

        pixel_angle = np.arctan(diff_x / diff_y)

        # print 'pixel angle %s' % (pixel_angle * 180 / np.pi)

    # work out phi which is the overall angle between
    # the spaxel being modelled and the central spaxel/position angle
    # this involves summing with a rotation angle which depends on
    # the spaxel quadrant

    if diff_x >= 0 and diff_y >= 0 and not(diff_x == 0 and diff_y == 0):

        # print 'top left'
        # we're in the upper left quadrant, want rot to be 270

        rot = 3 * np.pi / 2

    elif diff_x >= 0 and diff_y < 0:

        # print 'top right'

        # we're in the upper right quandrant, want rot to be 90

        rot = np.pi / 2

    elif diff_x < 0 and diff_y < 0:

        # print 'lower right'

        # we're in the lower right quadrant

        rot = np.pi / 2

    elif diff_x < 0 and diff_y >= 0:

        # print 'lower left'

        # we're in the lower left quadrant

        rot = 3 * np.pi / 2

    elif diff_x == 0 and diff_y == 0:

        # print 'middle'

        # we're in the middle

        rot = pa

    phi = pixel_angle - pa + rot

#    print 'differences: %s %s' % (diff_x, diff_y)
#    print 'pixel angle %s' % (pixel_angle * 180 / np.pi)
#    print 'position angle %s' % (pa * 180 / np.pi)
#    print 'rotation angle %s' % (rot * 180 / np.pi)
#    print 'overall angle %s' % (phi * 180 / np.pi)
#    print 'cosine of angle %s' % (np.cos(phi))

    r = np.sqrt(diff_x*diff_x + diff_y*diff_y)

    vel = np.cos(phi) * np.sin(inc) * (2 / np.pi) * vasym * np.arctan(r / rt)

    # print vel, xpix, ypix

    return vel

def grid_factor(res_factor,
                xpix,
                ypix):

    """
    Def: return an empty grid with 10 times spatial resolution of
    the velocity data
    """

    # create a 1D arrays of length dim_x * dim_y containing the
    # spaxel coordinates

    xbin = np.arange(0, xpix * res_factor, 1)

    ybin = np.arange(0, ypix * res_factor, 1)

    ybin, xbin = np.meshgrid(ybin, xbin)

    xbin = np.ravel(xbin)

    ybin = np.ravel(ybin)

    return np.array(xbin) * 1.0, np.array(ybin) * 1.0

def compute_model_grid_for_chi_squared(xpix,
                                       ypix,
                                       theta,
                                       inc,
                                       redshift,
                                       wave_array,
                                       xcen,
                                       ycen,
                                       seeing,
                                       sersic_n,
                                       sigma,
                                       pix_scale,
                                       psf_factor,
                                       sersic_factor,
                                       m_factor,
                                       light_profile,
                                       smear=True):

    """
    Def:
    Use the grid function to construct a basis for the model.
    Then apply the disk function to each spaxel in the basis
    reshape back to 2d array and plot the model velocity
    """

    xbin, ybin = grid_factor(m_factor,
                             xpix,
                             ypix)

    # setup list to house the velocity measurements

    vel_array = []

    # need to increase rt by the model factor, m_factor.

    pa, rt, v = theta

    rt = rt * m_factor

    # and reconstruct theta

    theta = [pa, rt, v]

    # compute the model at each spaxel location

    for xpos, ypos in zip(xbin, ybin):

        # run the disk function

        vel_array.append(disk_function_fixed_inc_fixed(theta,
                                                       xcen * m_factor,
                                                       ycen * m_factor,
                                                       inc,
                                                       xpos,
                                                       ypos))

    # create numpy array from the vel_array list

    vel_array = np.array(vel_array)

    # reshape back to the chosen grid dimensions

    vel_2d = vel_array.reshape((xpix * m_factor,
                                ypix * m_factor))

    if float(m_factor) != 1.0:

        vel_2d = psf.bin_by_factor(vel_2d,
                                   m_factor)

    pa = theta[0]

    rt = theta[1]

    if smear:

        vel_2d_blurred, sigma_2d = psf.cube_blur(vel_2d,
                                                 redshift,
                                                 wave_array,
                                                 xcen,
                                                 ycen,
                                                 seeing,
                                                 pix_scale,
                                                 psf_factor,
                                                 sersic_factor,
                                                 pa,
                                                 inc,
                                                 rt,
                                                 light_profile,
                                                 sigma,
                                                 sersic_n)


        return [vel_2d, vel_2d_blurred, sigma_2d]

    return vel_2d

def extract_in_apertures_fixed_inc_fixed(param_file,
                                         inc,
                                         redshift,
                                         wave_array,
                                         xcen,
                                         ycen,
                                         r_aper,
                                         d_aper,
                                         seeing,
                                         sersic_n,
                                         sigma,
                                         pix_scale,
                                         psf_factor,
                                         sersic_factor,
                                         m_factor,
                                         light_profile,
                                         smear=True):

    """
    Def: Extract the velocity field along the kinematic axis returned by the
    model fitting in both the data and the model for comparison. The model
    will show a perfect arctangent function.

    Input:
            theta - array of best fitting model parameter values
            model_data - best fit model computed from the compute_model_grid
            vel_data - array containing the actual velocity data
            r_aper - aperture size in pixels to use for each aperture
            d_aper - distance spacing between apertures
    Output:
            1D arrays containing the extracted model and real velocity fields
            along the kinematic major axis
    """

    # assign the best fit parameters to variables from the theta array

    # load in the relavant parameter file

    table = ascii.read(param_file)

    # define the parameters here

    gal_name = table['Gal_name'].data[0]

    # define the observed data values
    # the observed velocities
    vel_field_name = gal_name[:-5] + '_vel_field.fits'

    vel_data = fits.open(vel_field_name)[0].data

    xpix = vel_data.shape[0]

    ypix = vel_data.shape[1]

    # the errors on those velocities
    vel_error_field_name = gal_name[:-5] + '_error_field.fits'

    vel_error_data = fits.open(vel_error_field_name)[0].data

    # the observed sigma
    sig_field_name = gal_name[:-5] + '_sig_field.fits'

    sig_data = fits.open(sig_field_name)[0].data

    # the errors on sigma
    sig_error_field_name = gal_name[:-5] + '_sig_error_field.fits'

    sig_error_data = fits.open(sig_error_field_name)[0].data

    # the errors on sigma
    sig_sky_field_name = gal_name[:-5] + '_sig_sky_field.fits'

    sig_sky_data = fits.open(sig_sky_field_name)[0].data

    # and now the rest of the derived parameters for this galaxy
    xcen = table['xcen'].data[0]

    ycen = table['ycen'].data[0]

    inc = table['inc'].data[0]

    pa = table['position_angle'].data[0]

    rt = table['Rt'].data[0]

    vmax = table['Vmax'].data[0]

    theta = [pa, rt, vmax]

    # compute the model grid with the specified parameters

    model_vel = compute_model_grid_for_chi_squared(xpix,
                                                   ypix,
                                                   theta,
                                                   inc,
                                                   redshift,
                                                   wave_array,
                                                   xcen,
                                                   ycen,
                                                   seeing,
                                                   sersic_n,
                                                   sigma,
                                                   pix_scale,
                                                   psf_factor,
                                                   sersic_factor, 
                                                   m_factor,
                                                   light_profile,
                                                   smear)

    mod_vel, mod_vel_blurred, sig_blurred = model_vel

    # calculate the intrinsic sigma 2d field
    sig_int = np.sqrt((sig_data - sig_blurred)**2 - sig_sky_data**2)

    # and the full sigma model
    sig_full_model = np.sqrt((sig_blurred + sigma)**2 + sig_sky_data**2)

    # the residuals
    vel_res = vel_data - mod_vel_blurred

    sig_res = sig_int - sig_full_model

    # use the external rot_pa class to extract the 
    # velocity values and x values along the different pa's
    # have to do this for the different model values and the
    # different data values

    # modelled velocity values
    one_d_mod_vel_intrinsic, one_d_model_x = rt_pa.extract(d_aper,
                                                           r_aper,
                                                           pa,
                                                           mod_vel,
                                                           xcen,
                                                           ycen,
                                                           pix_scale)

    # modelled velocity values
    one_d_mod_vel_blurred, one_d_model_x = rt_pa.extract(d_aper,
                                                         r_aper,
                                                         pa,
                                                         mod_vel_blurred,
                                                         xcen,
                                                         ycen,
                                                         pix_scale)

    # data velocity values
    one_d_data_vel, one_d_data_x = rt_pa.extract(d_aper,
                                             r_aper,
                                             pa,
                                             vel_data,
                                             xcen,
                                             ycen,
                                             pix_scale)

    # velocity residuals values
    one_d_vel_res, one_d_data_x = rt_pa.extract(d_aper,
                                                r_aper,
                                                pa,
                                                vel_res,
                                                xcen,
                                                ycen,
                                                pix_scale)

    # data velocity error values
    one_d_data_vel_errors, one_d_data_x = rt_pa.extract(d_aper,
                                                        r_aper,
                                                        pa,
                                                        vel_error_data,
                                                        xcen,
                                                        ycen,
                                                        pix_scale)

    # data sigma values
    one_d_data_sig, one_d_data_x = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa,
                                                 sig_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

    # intrinsic sigma values
    one_d_sig_int, one_d_data_x = rt_pa.extract(d_aper,
                                                r_aper,
                                                pa,
                                                sig_int,
                                                xcen,
                                                ycen,
                                                pix_scale)

    # full sigma model
    one_d_sig_model_full, one_d_data_x = rt_pa.extract(d_aper,
                                                       r_aper,
                                                       pa,
                                                       sig_full_model,
                                                       xcen,
                                                       ycen,
                                                       pix_scale)    

    # sigma residuals
    one_d_sig_res, one_d_data_x = rt_pa.extract(d_aper,
                                                r_aper,
                                                pa,
                                                sig_res,
                                                xcen,
                                                ycen,
                                                pix_scale)

    # data sigma error values
    one_d_data_sig_errors, x = rt_pa.extract(d_aper,
                                             r_aper,
                                             pa,
                                             sig_error_data,
                                             xcen,
                                             ycen,
                                             pix_scale)

    # plotting the model and extracted quantities

    min_ind = 0
    max_ind = 0

    try:

        while np.isnan(one_d_data_vel[min_ind]):

            min_ind += 1

    except IndexError:

        min_ind = 0

    try:

        while np.isnan(one_d_data_vel[::-1][max_ind]):

            max_ind += 1

        max_ind = max_ind + 1

    except IndexError:

        max_ind = 0

    # construct dictionary of these velocity values and 
    # the final distance at which the data is extracted from centre

    extract_d = {'model_intrinsic': [one_d_mod_vel_intrinsic[min_ind] / np.sin(inc),
                                     one_d_mod_vel_intrinsic[-max_ind] / np.sin(inc)],
                 'model_blurred': [one_d_mod_vel_blurred[min_ind] / np.sin(inc),
                                   one_d_mod_vel_blurred[-max_ind] / np.sin(inc)],
                 'vel_data': [one_d_data_vel[min_ind] / np.sin(inc),
                              one_d_data_vel[-max_ind] / np.sin(inc)],
                 'distance': [one_d_model_x[min_ind],
                              one_d_model_x[-max_ind]],
                 'vel_error': [one_d_data_vel_errors[min_ind],
                               one_d_data_vel_errors[-max_ind]],
                 'vel_max': [np.nanmax(abs(one_d_data_vel / np.sin(inc)))],
                 'inclination' : inc}

    # return the data used in plotting for use elsewhere

    return {'one_d_mod_vel_intrinsic': one_d_mod_vel_intrinsic,
            'one_d_mod_vel_blurred' : one_d_mod_vel_blurred,
            'one_d_data_vel' : one_d_data_vel,
            'one_d_vel_res': one_d_vel_res,
            'one_d_data_vel_errors': one_d_data_vel_errors,
            'one_d_data_sig' : one_d_data_sig,
            'one_d_sig_int' : one_d_sig_int,
            'one_d_sig_full_model' : one_d_sig_model_full,
            'one_d_sig_res' : one_d_sig_res,
            'one_d_data_sig_errors' : one_d_data_sig_errors,
            'one_d_data_x' : one_d_data_x,
            'one_d_model_x' : one_d_model_x,
            'vel_res' : vel_res,
            'intrinsic_sig' : sig_int,
            'sig_full_model' : sig_full_model,
            'sig_res' : sig_res}, extract_d

def extract_in_apertures_mcmc_version(gal_name,
                                      theta,
                                      inc,
                                      redshift,
                                      wave_array,
                                      xcen,
                                      ycen,
                                      r_aper,
                                      d_aper,
                                      seeing,
                                      sersic_n,
                                      sigma,
                                      pix_scale,
                                      psf_factor,
                                      sersic_factor,
                                      m_factor,
                                      light_profile,
                                      smear=True):

    """
    Def: Extract the velocity field along the kinematic axis returned by the
    model fitting in both the data and the model for comparison. The model
    will show a perfect arctangent function.

    Input:
            theta - array of best fitting model parameter values
            model_data - best fit model computed from the compute_model_grid
            vel_data - array containing the actual velocity data
            r_aper - aperture size in pixels to use for each aperture
            d_aper - distance spacing between apertures
    Output:
            1D arrays containing the extracted model and real velocity fields
            along the kinematic major axis
    """


    # define the observed data values
    # the observed velocities
    vel_field_name = gal_name[:-5] + '_vel_field.fits'

    vel_data = fits.open(vel_field_name)[0].data

    xpix = vel_data.shape[0]

    ypix = vel_data.shape[1]

    # the errors on those velocities
    vel_error_field_name = gal_name[:-5] + '_error_field.fits'

    vel_error_data = fits.open(vel_error_field_name)[0].data

    # the observed sigma
    sig_field_name = gal_name[:-5] + '_sig_field.fits'

    sig_data = fits.open(sig_field_name)[0].data

    # the errors on sigma
    sig_error_field_name = gal_name[:-5] + '_sig_error_field.fits'

    sig_error_data = fits.open(sig_error_field_name)[0].data

    # the errors on sigma
    sig_sky_field_name = gal_name[:-5] + '_sig_sky_field.fits'

    sig_sky_data = fits.open(sig_sky_field_name)[0].data

    # compute the model grid with the specified parameters

    # define the pa
    pa = theta[0]

    model_vel = compute_model_grid_for_chi_squared(xpix,
                                                   ypix,
                                                   theta,
                                                   inc,
                                                   redshift,
                                                   wave_array,
                                                   xcen,
                                                   ycen,
                                                   seeing,
                                                   sersic_n,
                                                   sigma,
                                                   pix_scale,
                                                   psf_factor,
                                                   sersic_factor, 
                                                   m_factor,
                                                   light_profile,
                                                   smear)

    mod_vel, mod_vel_blurred, sig_blurred = model_vel

    # calculate the intrinsic sigma 2d field
    sig_int_squared = (sig_data - sig_blurred)**2 - sig_sky_data**2

    # some of these values could be less than 0
    for i in range(sig_int_squared.shape[0]):
        for j in range(sig_int_squared.shape[1]):
            if sig_int_squared[i,j] < 0:
                sig_int_squared[i,j] = 0

    sig_int = np.sqrt(sig_int_squared)

    # and the full sigma model
    sig_full_model = np.sqrt((sig_blurred + sigma)**2 + sig_sky_data**2)

    # the residuals
    vel_res = vel_data - mod_vel_blurred

    sig_res = sig_int - sig_full_model

    # use the external rot_pa class to extract the 
    # velocity values and x values along the different pa's
    # have to do this for the different model values and the
    # different data values

    # modelled velocity values
    one_d_mod_vel_intrinsic, one_d_model_x = rt_pa.extract(d_aper,
                                                           r_aper,
                                                           pa,
                                                           mod_vel,
                                                           xcen,
                                                           ycen,
                                                           pix_scale)

    # modelled velocity values
    one_d_mod_vel_blurred, one_d_model_x = rt_pa.extract(d_aper,
                                                         r_aper,
                                                         pa,
                                                         mod_vel_blurred,
                                                         xcen,
                                                         ycen,
                                                         pix_scale)

    # data velocity values
    one_d_data_vel, one_d_data_x = rt_pa.extract(d_aper,
                                             r_aper,
                                             pa,
                                             vel_data,
                                             xcen,
                                             ycen,
                                             pix_scale)

    # velocity residuals values
    one_d_vel_res, one_d_data_x = rt_pa.extract(d_aper,
                                                r_aper,
                                                pa,
                                                vel_res,
                                                xcen,
                                                ycen,
                                                pix_scale)

    # data velocity error values
    one_d_data_vel_errors, one_d_data_x = rt_pa.extract(d_aper,
                                                        r_aper,
                                                        pa,
                                                        vel_error_data,
                                                        xcen,
                                                        ycen,
                                                        pix_scale)

    # data sigma values
    one_d_data_sig, one_d_data_x = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa,
                                                 sig_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

    # intrinsic sigma values
    one_d_sig_int, one_d_data_x = rt_pa.extract(d_aper,
                                                r_aper,
                                                pa,
                                                sig_int,
                                                xcen,
                                                ycen,
                                                pix_scale)

    # full sigma model
    one_d_sig_model_full, one_d_data_x = rt_pa.extract(d_aper,
                                                       r_aper,
                                                       pa,
                                                       sig_full_model,
                                                       xcen,
                                                       ycen,
                                                       pix_scale)    

    # sigma residuals
    one_d_sig_res, one_d_data_x = rt_pa.extract(d_aper,
                                                r_aper,
                                                pa,
                                                sig_res,
                                                xcen,
                                                ycen,
                                                pix_scale)

    # data sigma error values
    one_d_data_sig_errors, x = rt_pa.extract(d_aper,
                                             r_aper,
                                             pa,
                                             sig_error_data,
                                             xcen,
                                             ycen,
                                             pix_scale)

    # plotting the model and extracted quantities

    min_ind = 0
    max_ind = 0

    try:

        while np.isnan(one_d_data_vel[min_ind]):

            min_ind += 1

    except IndexError:

        min_ind = 0

    try:

        while np.isnan(one_d_data_vel[::-1][max_ind]):

            max_ind += 1

        max_ind = max_ind + 1

    except IndexError:

        max_ind = 0

    # construct dictionary of these velocity values and 
    # the final distance at which the data is extracted from centre

    extract_d = {'model_intrinsic': [one_d_mod_vel_intrinsic[min_ind] / np.sin(inc),
                                     one_d_mod_vel_intrinsic[-max_ind] / np.sin(inc)],
                 'model_blurred': [one_d_mod_vel_blurred[min_ind] / np.sin(inc),
                                   one_d_mod_vel_blurred[-max_ind] / np.sin(inc)],
                 'vel_data': [one_d_data_vel[min_ind] / np.sin(inc),
                              one_d_data_vel[-max_ind] / np.sin(inc)],
                 'distance': [one_d_model_x[min_ind],
                              one_d_model_x[-max_ind]],
                 'vel_error': [one_d_data_vel_errors[min_ind],
                               one_d_data_vel_errors[-max_ind]],
                 'vel_max': [np.nanmax(abs(one_d_data_vel / np.sin(inc)))],
                 'inclination' : inc}

    # return the data used in plotting for use elsewhere

    return {'one_d_mod_vel_intrinsic': one_d_mod_vel_intrinsic,
            'one_d_mod_vel_blurred' : one_d_mod_vel_blurred,
            'one_d_data_vel' : one_d_data_vel,
            'one_d_vel_res': one_d_vel_res,
            'one_d_data_vel_errors': one_d_data_vel_errors,
            'one_d_data_sig' : one_d_data_sig,
            'one_d_sig_int' : one_d_sig_int,
            'one_d_sig_full_model' : one_d_sig_model_full,
            'one_d_sig_res' : one_d_sig_res,
            'one_d_data_sig_errors' : one_d_data_sig_errors,
            'one_d_data_x' : one_d_data_x,
            'one_d_model_x' : one_d_model_x,
            'vel_res' : vel_res,
            'intrinsic_sig' : sig_int,
            'sig_full_model' : sig_full_model,
            'sig_res' : sig_res}, extract_d

def make_all_plots_fixed_inc_fixed(inc,
                                   redshift,
                                   wave_array,
                                   xcen,
                                   ycen,
                                   infile,
                                   r_aper,
                                   d_aper,
                                   seeing,
                                   sersic_n,
                                   sigma,
                                   pix_scale,
                                   psf_factor,
                                   sersic_factor,
                                   m_factor,
                                   light_profile,
                                   galaxy_boundaries_file,
                                   smear=True):

    """
    Def: Take all of the data from the stott velocity fields,
    mcmc modelling and hst imaging and return a grid of plots
    summarising the results.

    Input:
            in_file - file path and name of object

    Output:
            grid of plots
    """

    # Get the conversion between arcseconds and kpc at this redshift

    from astropy.cosmology import WMAP9 as cosmo

    scale = cosmo.kpc_proper_per_arcmin(redshift).value / 60.0

    # open the various files and run the methods to get the data
    # for plotting

    gal_name = infile[len(infile) -
                      infile[::-1].find("/"):]

    table = ascii.read('%s_chi_squared_params.txt' % infile[:-5])

    # and now the rest of the derived parameters for this galaxy
    xcen = table['xcen'].data[0]

    ycen = table['ycen'].data[0]

    inc = table['inc'].data[0]

    pa = table['position_angle'].data[0]

    rt = table['Rt'].data[0]

    va = table['Vmax'].data[0]

    theta = [pa, rt, va]

    hst_stamp_name = infile[:-5] + '_galfit.fits'

    table_hst = fits.open(hst_stamp_name)

    # do the initial numerical fitting to find the 
    # half light radius and other quantities

    table = ascii.read(galaxy_boundaries_file)

    xl = table['xl'][gal_num]
    xh = table['xh'][gal_num]
    yl = table['yl'][gal_num]
    yh = table['yh'][gal_num]

    half_light_dict = ap_growth.find_aperture_parameters(hst_stamp_name,
                                                         xl,
                                                         xh,
                                                         yl,
                                                         yh)

    # assign the parameters from the dictionary
    # where prefix num refers to the fact that
    # this has been done numerically

    num_cut_data = half_light_dict['cut_data']
    num_fit_data = half_light_dict['fit_data']
    num_axis_array = half_light_dict['a_array']
    num_sum_array = half_light_dict['sum_array']
    num_axis_ratio = half_light_dict['axis_ratio']
    num_r_e = half_light_dict['r_e_pixels']
    num_r_9 = half_light_dict['r_9_pixels']
    num_pa = half_light_dict['pa'] + np.pi / 2.0

    scaled_axis_array = 0.06 * scale * num_axis_array
    scaled_num_r_e = 0.06 * scale * num_r_e
    scaled_num_r_9 = 0.06 * scale * num_r_9

    galfit_mod = table_hst[2].data

    galfit_res = table_hst[3].data

    # Get thhe galfit axis ratio

    axis_r_str = table_hst[2].header['1_AR']

    axis_r = axis_r_str[:len(axis_r_str) -
                    axis_r_str[::-1].find("+") - 2]

    if axis_r[0] == '[':

        axis_r = axis_r[1:]

    # If the parameter has not been well determined by galfit
    # need to account for the asterisks

    if axis_r[0] == '*':

        axis_r = axis_r[1:-1]

    axis_r = float(axis_r)

    # Get the galfit scale radius

    r_e_str = table_hst[2].header['1_RE']

    r_e = r_e_str[:len(r_e_str) -
                    r_e_str[::-1].find("+") - 2]

    if r_e[0] == '[':

        r_e = r_e[1:]

    if r_e[0] == '*':

        r_e = r_e[1:-1]


    # since r_e has been measured using HST which has a pixel scale
    # of 0.06, need to multiply by the ratio of this to the actual
    # pixel_scale for KMOS which is 0.6. This is working in
    # arcseconds - should have realised this a long time ago

    r_e = float(r_e) * 0.6

    # This r_e is in pixels, multiply 
    # by 0.1 and then by scale to put into KPC

    r_e = pix_scale * r_e * scale

    r_d = r_e / 1.67835

    r_d_22 = 2.2 * r_d

    r_d_30 = 3.4 * r_d

    # Converting back to arcseconds

    r_e_arc = r_e / scale

    r_d_arc = r_e_arc / 1.67835

    r_d_arc_22 = 2.2 * r_d_arc

    r_d_arc_30 = 3.4 * r_d_arc

    hst_pa_str = table_hst[2].header['1_PA']

    hst_pa = hst_pa_str[:len(hst_pa_str) -
                              hst_pa_str[::-1].find("+") - 2]

    if hst_pa[0] == '[':

        hst_pa = hst_pa[1:]

    if hst_pa[0] == '*':

        hst_pa = hst_pa[1:-1]

    hst_pa = float(hst_pa)

    # convert between degrees and radians

    if hst_pa < 0:

        hst_pa = hst_pa + 180

    hst_pa = (hst_pa * np.pi) / 180

    data_hst = table_hst[1].data

    flux_name = infile[:-5] + '_flux_field.fits'

    table_flux = fits.open(flux_name)

    data_flux = table_flux[0].data

    vel_field_name = infile[:-5] + '_vel_field.fits'

    table_vel = fits.open(vel_field_name)

    data_vel = table_vel[0].data

    table_error = fits.open('%s_error_field.fits' % infile[:-5])

    error_vel = table_error[0].data

    vel = vel_field(vel_field_name,
                    xcen,
                    ycen)

    xpix = data_vel.shape[0]

    ypix = data_vel.shape[1]

    data_model = compute_model_grid_for_chi_squared(xpix,
                                                    ypix,
                                                    theta,
                                                    inc,
                                                    redshift,
                                                    wave_array,
                                                    xcen,
                                                    ycen,
                                                    seeing,
                                                    sersic_n,
                                                    sigma,
                                                    pix_scale,
                                                    psf_factor,
                                                    sersic_factor,
                                                    m_factor,
                                                    light_profile,
                                                    smear)

    mod_vel, mod_vel_blurred, sig_blurred = data_model

    # truncate this to the data size

    mask_array = np.empty(shape=(xpix, ypix))

    for i in range(0, xpix):

        for j in range(0, ypix):

            if np.isnan(data_vel[i][j]):

                mask_array[i][j] = np.nan

            else:

                mask_array[i][j] = 1.0



    table_sig = fits.open('%s_sig_field.fits' % infile[:-5])

    data_sig = table_sig[0].data

    table_sig_error = fits.open('%s_sig_error_field.fits' % infile[:-5])

    data_sig_error = table_sig_error[0].data

    idv_param_file = infile[:-5] + '_chi_squared_params.txt'

    one_d_plots, extract_values = extract_in_apertures_fixed_inc_fixed(idv_param_file,
                                                                       inc,
                                                                       redshift,
                                                                       wave_array,
                                                                       xcen,
                                                                       ycen,
                                                                       r_aper,
                                                                       d_aper,
                                                                       seeing,
                                                                       sersic_n,
                                                                       sigma,
                                                                       pix_scale,
                                                                       psf_factor,
                                                                       sersic_factor,
                                                                       m_factor,
                                                                       light_profile,
                                                                       smear)

    one_d_mod_vel_intrinsic = one_d_plots['one_d_mod_vel_intrinsic']
    one_d_mod_vel_blurred = one_d_plots['one_d_mod_vel_blurred']
    one_d_data_vel = one_d_plots['one_d_data_vel']
    one_d_vel_res = one_d_plots['one_d_vel_res']
    one_d_data_vel_errors = one_d_plots['one_d_data_vel_errors']
    one_d_data_sig = one_d_plots['one_d_data_sig']
    one_d_sig_int = one_d_plots['one_d_sig_int']
    one_d_sig_model_full = one_d_plots['one_d_sig_full_model']
    one_d_sig_res = one_d_plots['one_d_sig_res']
    one_d_data_sig_errors = one_d_plots['one_d_data_sig_errors']
    one_d_data_x = one_d_plots['one_d_data_x']
    one_d_model_x = one_d_plots['one_d_model_x']
    vel_res = one_d_plots['vel_res']
    sig_int = one_d_plots['intrinsic_sig']
    sig_full_model = one_d_plots['sig_full_model']
    sig_res = one_d_plots['sig_res']


    # take product of model and mask_array to return new data

    mod_vel_masked = mod_vel * mask_array
    mod_vel_blurred_masked = mod_vel_blurred * mask_array
    vel_res_masked = vel_res * mask_array
    sig_blurred_masked = sig_blurred * mask_array
    sig_int_masked = sig_int * mask_array
    sig_full_model_masked = sig_int * mask_array
    sig_res_masked = sig_res * mask_array

    # set the imshow plotting limmits
    vel_min, vel_max = np.nanpercentile(mod_vel_blurred_masked,
                                        [5.0, 95.0])

    sig_min, sig_max = np.nanpercentile(sig_int,
                                        [5.0, 95.0])

    d_for_mask = copy(data_vel)

    # get the continuum images and narrow band OIII
    cont_dict = f_f.flatfield(infile,
                              d_for_mask,
                              redshift)

    cont1 = cont_dict['cont1']
    cont2 = cont_dict['cont2']
    o_nband = cont_dict['OIII']

    # CUT DOWN ALL OF THE DATA WE HAVE TO GET RID OF SPAXELS 
    # ON THE OUTSKIRTS - 3 SPAXELS IN KMOS AND 5 in HST

    # smooth the continuum image

    b_cont2 = psf.blur_by_psf(cont2,
                              0.3,
                              pix_scale,
                              psf_factor)

    # for gaussian fitting only want to use the pixels
    # which have been accepted in the stott fitting

    g_mask = np.empty(shape=(data_vel.shape[0],
                             data_vel.shape[1]))

    print 'This is the mask shape: %s %s' % (g_mask.shape[0], g_mask.shape[1])

    for i in range(data_vel.shape[0]):

        for j in range(data_vel.shape[1]):

            if np.isnan(data_vel[i, j]):

                g_mask[i, j] = np.nan

            else:

                g_mask[i, j] = 1.0

    fit_b_cont2 = g_mask * b_cont2

    # attempt to fit the continuum with a gaussian

    fit_cont, fit_params = g2d.fit_gaussian(fit_b_cont2)

    # and assign the center coordinates

    fit_cont_x = fit_params[3]
    fit_cont_y = fit_params[2]

    print 'These are the fit center coordinates: %s %s' % (fit_cont_x, fit_cont_y) 

    # Also get estimate of the center using peak pixel within
    # that masking region

    cont_peak_coords = np.unravel_index(np.nanargmax(fit_b_cont2),
                                        fit_b_cont2.shape)

    cont_peak_x = cont_peak_coords[0]
    cont_peak_y = cont_peak_coords[1]

    print 'These are the continuum peak: %s %s' % (cont_peak_x, cont_peak_y) 

    fit_o_nband = g_mask * o_nband

    oiii_peak_coords = np.unravel_index(np.nanargmax(fit_o_nband),
                                        fit_o_nband.shape)

    oiii_peak_x = oiii_peak_coords[0]
    oiii_peak_y = oiii_peak_coords[1]

    print 'This is the OIII peak: %s %s' % (oiii_peak_x, oiii_peak_y)

    hst_fit, hst_fit_params = g2d.fit_gaussian(galfit_mod)

    # 1D spectrum finding - note this will be unweighted
    # and not as good as a weighted version which doesn't have
    # as large a contribution from the outside spaxels
    # using the spaxels in the mask

    obj_cube = cubeOps(infile)
    one_d_spectrum = []

    #  OIII wavelength
    central_l = (1 + redshift) * 0.500824

    o_peak = np.argmin(abs(central_l - wave_array))

    for i in range(data_vel.shape[0]):

        for j in range(data_vel.shape[1]):

            if not(np.isnan(g_mask[i, j])):

                one_d_spectrum.append(obj_cube.data[:, i, j])

    # sum for final spectrum
    one_d_spectrum = np.nansum(one_d_spectrum, axis=0)

    # Now have all information to define an astropy elliptical
    # aperture using the galfit parameters

    from photutils import EllipticalAperture

    theta = hst_pa + np.pi / 2.0
    major_axis = r_e_arc * 10
    minor_axis = major_axis * axis_r
    galfit_x = hst_fit_params[3]
    galfit_y = hst_fit_params[2]
    positions = [(galfit_y, galfit_x)]

    print 'This is the axis ratio: %s' % axis_r

    apertures = EllipticalAperture(positions,
                                   major_axis,
                                   minor_axis,
                                   theta)

    disk_apertures = EllipticalAperture(positions,
                                        1.8*major_axis,
                                        1.8*minor_axis,
                                        theta)

    # Now compute alternate PA from rotating the slit until
    # it maximises the velocity difference

    best_pa, pa_array, stat_array = rt_pa.rot_pa(d_aper,
                                                 r_aper,
                                                 data_vel, 
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

    # Also use the rt_pa method to extract the 1D spectrum
    # and errors along the HST and BEST pa's
    # these will be plotted along with the dynamical PA plots
    print 'THESE ARE THE PAS'
    print 'HST: %s' % hst_pa
    print 'ROT: %s' % best_pa
    print 'DYN: %s' % pa

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.imshow(data_vel)
    plt.close('all')

    # extract the velocity data in each case of position angle

    hst_pa_vel, hst_pa_x = rt_pa.extract(d_aper,
                                         r_aper,
                                         hst_pa,
                                         data_vel, 
                                         xcen,
                                         ycen,
                                         pix_scale)

    hst_pa_error, hst_pa_x = rt_pa.extract(d_aper,
                                           r_aper,
                                           hst_pa,
                                           error_vel, 
                                           xcen,
                                           ycen,
                                           pix_scale)

    best_pa_vel, best_pa_x = rt_pa.extract(d_aper,
                                         r_aper,
                                         best_pa,
                                         data_vel, 
                                         xcen,
                                         ycen,
                                         pix_scale)

    best_pa_error, best_pa_x = rt_pa.extract(d_aper,
                                           r_aper,
                                           best_pa,
                                           error_vel, 
                                           xcen,
                                           ycen,
                                           pix_scale)

    # extract the sigma data in each case of position angle

    hst_pa_sig, hst_pa_sig_x = rt_pa.extract(d_aper,
                                             r_aper,
                                             hst_pa,
                                             data_sig, 
                                             xcen,
                                             ycen,
                                             pix_scale)

    hst_pa_sig_error, hst_pa_sig_x = rt_pa.extract(d_aper,
                                                   r_aper,
                                                   hst_pa,
                                                   data_sig_error, 
                                                   xcen,
                                                   ycen,
                                                   pix_scale)

    best_pa_sig, best_pa_sig_x = rt_pa.extract(d_aper,
                                               r_aper,
                                               best_pa,
                                               data_sig, 
                                               xcen,
                                               ycen,
                                               pix_scale)

    best_pa_sig_error, best_pa_sig_x = rt_pa.extract(d_aper,
                                                     r_aper,
                                                     best_pa,
                                                     data_sig_error, 
                                                     xcen,
                                                     ycen,
                                                     pix_scale)

    dyn_pa_sig, dyn_pa_sig_x = rt_pa.extract(d_aper,
                                               r_aper,
                                               pa,
                                               data_sig, 
                                               xcen,
                                               ycen,
                                               pix_scale)

    dyn_pa_sig_error, dyn_pa_sig_x = rt_pa.extract(d_aper,
                                                     r_aper,
                                                     pa,
                                                     data_sig_error, 
                                                     xcen,
                                                     ycen,
                                                     pix_scale)

    # now want to take the average of the first and last sigma
    # values extracted along each of the position angles

    # HST POSITION ANGLE
    hst_i, hst_j = rt_pa.find_first_valid_entry(hst_pa_sig)

    hst_mean_sigma = np.nanmean([hst_pa_sig[hst_i],hst_pa_sig[hst_j]])

    # DYN POSITION ANGLE
    dyn_i, dyn_j = rt_pa.find_first_valid_entry(dyn_pa_sig)

    dyn_mean_sigma = np.nanmean([dyn_pa_sig[dyn_i],dyn_pa_sig[dyn_j]])

    # HST POSITION ANGLE
    best_i, best_j = rt_pa.find_first_valid_entry(best_pa_sig)

    best_mean_sigma = np.nanmean([best_pa_sig[best_i],best_pa_sig[best_j]])

    # calculate the boundaries from which to draw a line
    # through the images relating to the position angles

    x_inc_hst = 100 * np.abs(np.cos(hst_pa))
    y_inc_hst = 100 * np.abs(np.sin(hst_pa))

    # Find the boundaries for plotting the PAs
    # Use the continuum center in order to do this

    if 0 < hst_pa < np.pi / 2.0 or np.pi < hst_pa < 3 * np.pi / 2.0:

        # in the top right and bottom left areas
        # so adding to x goes with subtracting from y

        x_h_low = xcen + x_inc_hst
        x_h_high = xcen - x_inc_hst
        y_h_low = ycen - y_inc_hst
        y_h_high = ycen + y_inc_hst

    else:

        x_h_low = xcen - x_inc_hst
        x_h_high = xcen + x_inc_hst
        y_h_low = ycen - y_inc_hst
        y_h_high = ycen + y_inc_hst

    # calculate the boundaries from which to draw a line
    # through the images relating to the position angles

    x_inc = 100 * np.abs(np.cos(pa))
    y_inc = 100 * np.abs(np.sin(pa))

    # find boundaries by imposing the same conditions as
    # in the extract apertures for calculating the angle
    # i.e. relying on the invariance of two segments

    if 0 < pa < np.pi / 2.0 or np.pi < pa < 3 * np.pi / 2.0:

        # in the top right and bottom left areas
        # so adding to x goes with subtracting from y

        x_low = xcen + x_inc
        x_high = xcen - x_inc
        y_low = ycen - y_inc
        y_high = ycen + y_inc

    else:

        x_low = xcen - x_inc
        x_high = xcen + x_inc
        y_low = ycen - y_inc
        y_high = ycen + y_inc

    x_inc_best = 100 * np.abs(np.cos(best_pa))
    y_inc_best = 100 * np.abs(np.sin(best_pa))

    # find boundaries by imposing the same conditions as
    # in the extract apertures for calculating the angle
    # i.e. relying on the invariance of two segments

    if 0 < best_pa < np.pi / 2.0 or np.pi < best_pa < 3 * np.pi / 2.0:

        # in the top right and bottom left areas
        # so adding to x goes with subtracting from y

        x_low_best = xcen + x_inc_best
        x_high_best = xcen - x_inc_best
        y_low_best = ycen - y_inc_best
        y_high_best = ycen + y_inc_best

    else:

        x_low_best = xcen - x_inc_best
        x_high_best = xcen + x_inc_best
        y_low_best = ycen - y_inc_best
        y_high_best = ycen + y_inc_best

    x_inc_num = 100 * np.abs(np.cos(num_pa))
    y_inc_num = 100 * np.abs(np.sin(num_pa))

    if 0 < num_pa < np.pi / 2.0 or np.pi < num_pa < 3 * np.pi / 2.0:

        # in the top right and bottom left areas
        # so adding to x goes with subtracting from y

        x_low_num = xcen + x_inc_num
        x_high_num = xcen - x_inc_num
        y_low_num = ycen - y_inc_num
        y_high_num = ycen + y_inc_num

    else:

        x_low_num = xcen - x_inc_num
        x_high_num = xcen + x_inc_num
        y_low_num = ycen - y_inc_num
        y_high_num = ycen + y_inc_num

    # draw in the PA'S

    fig, ax = plt.subplots(4, 5, figsize=(24, 16))

    # flux plot
    ax[1][0].plot([y_h_low, y_h_high], [x_h_low, x_h_high],
               ls='--',
               color='aquamarine',
               label='hst\_pa')
    ax[1][0].plot([y_low, y_high], [x_low, x_high],
               ls='--',
               color='lightcoral',
               lw=2,
               label='dyn\_pa')
    ax[1][0].plot([y_low_best, y_high_best], [x_low_best, x_high_best],
               ls='--',
               color='darkorange',
               lw=2,
               label='rot\_pa')
    ax[1][0].plot([y_low_num, y_high_num], [x_low_num, x_high_num],
               ls='--',
               color='wheat',
               lw=2,
               label='num\_pa')
    l = ax[1][0].legend(loc='best',
                        frameon=False,
                        prop={'size':10})
    for text in l.get_texts():
        text.set_color("white")
    # velocity plot
    ax[1][1].plot([y_h_low, y_h_high], [x_h_low, x_h_high],
               ls='--',
               color='aquamarine',
               label='hst\_pa')
    ax[1][1].plot([y_low, y_high], [x_low, x_high],
               ls='--',
               color='lightcoral',
               lw=2,
               label='dyn\_pa')
    ax[1][1].plot([y_low_best, y_high_best], [x_low_best, x_high_best],
               ls='--',
               color='darkorange',
               lw=2,
               label='rot\_pa')
    l = ax[1][1].legend(loc='best',
                        frameon=False,
                        prop={'size':10})
    for text in l.get_texts():
        text.set_color("white")
    ax[1][2].plot([y_h_low, y_h_high], [x_h_low, x_h_high],
               ls='--',
               color='aquamarine',
               label='hst\_pa')
    ax[1][2].plot([y_low, y_high], [x_low, x_high],
               ls='--',
               color='lightcoral',
               lw=2,
               label='dyn\_pa')
    ax[1][2].plot([y_low_best, y_high_best], [x_low_best, x_high_best],
               ls='--',
               color='darkorange',
               lw=2,
               label='rot\_pa')
    l = ax[1][2].legend(loc='best',
                        frameon=False,
                        prop={'size':10})
    for text in l.get_texts():
        text.set_color("white")
    ax[1][3].plot([y_h_low, y_h_high], [x_h_low, x_h_high],
               ls='--',
               color='aquamarine',
               label='hst\_pa')
    ax[1][3].plot([y_low, y_high], [x_low, x_high],
               ls='--',
               color='lightcoral',
               lw=2,
               label='dyn\_pa')
    ax[1][3].plot([y_low_best, y_high_best], [x_low_best, x_high_best],
               ls='--',
               color='darkorange',
               lw=2,
               label='rot\_pa')
    l = ax[1][3].legend(loc='best',
                        frameon=False,
                        prop={'size':10})
    for text in l.get_texts():
        text.set_color("white")
    ax[1][4].plot([y_h_low, y_h_high], [x_h_low, x_h_high],
               ls='--',
               color='aquamarine',
               label='hst\_pa')
    ax[1][4].plot([y_low, y_high], [x_low, x_high],
               ls='--',
               color='lightcoral',
               lw=2,
               label='dyn\_pa')
    ax[1][4].plot([y_low_best, y_high_best], [x_low_best, x_high_best],
               ls='--',
               color='darkorange',
               lw=2,
               label='rot\_pa')
    l = ax[1][4].legend(loc='best',
                        frameon=False,
                        prop={'size':10})
    for text in l.get_texts():
        text.set_color("white")
    ax[2][4].plot([y_h_low, y_h_high], [x_h_low, x_h_high],
               ls='--',
               color='aquamarine',
               label='hst\_pa')
    ax[2][4].plot([y_low, y_high], [x_low, x_high],
               ls='--',
               color='lightcoral',
               lw=2,
               label='dyn\_pa')
    ax[2][4].plot([y_low_best, y_high_best], [x_low_best, x_high_best],
               ls='--',
               color='darkorange',
               lw=2,
               label='rot\_pa')
    l = ax[2][4].legend(loc='best',
                        frameon=False,
                        prop={'size':10})
    for text in l.get_texts():
        text.set_color("white")

    # mask background of velocity data to black

    # print data_hst.shape
    # print data_vel.shape

    m_data_flux = np.ma.array(data_flux,
                             mask=np.isnan(data_flux))
    m_data_hst = np.ma.array(data_hst,
                             mask=np.isnan(data_hst))
    m_data_vel = np.ma.array(data_vel,
                             mask=np.isnan(data_vel))
    m_data_vel_res = np.ma.array(vel_res_masked,
                             mask=np.isnan(vel_res_masked))
    m_data_mod_intrinsic = np.ma.array(mod_vel_masked,
                                       mask=np.isnan(mod_vel_masked))
    m_data_mod_blurred = np.ma.array(mod_vel_blurred_masked,
                                     mask=np.isnan(mod_vel_blurred_masked))
    m_data_sig = np.ma.array(sig_int_masked,
                             mask=np.isnan(sig_int_masked))

    cmap = plt.cm.bone_r
    cmap.set_bad('black', 1.)

    # HST

    im = ax[0][0].imshow(data_hst,
                         cmap=cmap,
                         vmax=10,
                         vmin=0)

    # HST - blurred

    blurred_hst = psf.blur_by_psf(data_hst,
                                  0.46,
                                  pix_scale,
                                  psf_factor)

    im = ax[3][3].imshow(blurred_hst,
                         cmap=cmap,
                         vmax=8,
                         vmin=0)

    ax[3][3].set_title('blurred HST')

    y_full_hst, x_full_hst = np.indices(galfit_mod.shape)

#        ax[0][0].contour(x_full_hst,
#                         y_full_hst,
#                         hst_fit,
#                         4,
#                         ls='solid',
#                         colors='b')

    apertures.plot(ax[0][0], color='green')
    disk_apertures.plot(ax[0][0], color='red')

    ax[0][0].text(4,7, 'z = %.2f' % redshift, color='black', fontsize=16)
    ax[0][0].text(4,14, 'pa = %.2f' % hst_pa, color='black', fontsize=16)
    ax[0][0].text(data_hst.shape[0] - 25,
                  data_hst.shape[1] - 6,
                  'F160\_W', color='black', fontsize=16)

    ax[0][0].tick_params(axis='x',
                      labelbottom='off')

    ax[0][0].tick_params(axis='y',
                      labelleft='off')


    ax[0][0].set_title('HST imaging')

    # GALFIT MODEL

    ax[0][1].text(4, 7,
                  r'$R_{e} = %.2f$Kpc' % r_e,
                  fontsize=16,
                  color='black')

    ax[0][1].text(4, 14,
                  r'$\frac{b}{a} = %.2f$' % axis_r,
                  fontsize=16,
                  color='black')

    im = ax[0][1].imshow(galfit_mod,
                      cmap=cmap,
                      vmax=10,
                      vmin=0)

    apertures.plot(ax[0][1], color='green')
    disk_apertures.plot(ax[0][1], color='red')

    ax[0][1].tick_params(axis='x',
                      labelbottom='off')

    ax[0][1].tick_params(axis='y',
                      labelleft='off')

    ax[0][1].set_title('GALFIT mod')

    # GALFIT RESIDUALS

    im = ax[0][2].imshow(galfit_res,
                      cmap=cmap,
                      vmax=10,
                      vmin=0)

    ax[0][2].tick_params(axis='x',
                      labelbottom='off')

    ax[0][2].tick_params(axis='y',
                      labelleft='off')

    ax[0][2].set_title('GALFIT res')

    cmap = plt.cm.hot
    cmap.set_bad('black', 1.)

    # FIRST CONTINUUM

    ax[0][3].scatter(cont_peak_y,
                     cont_peak_x,
                     marker='x',
                     s=100,
                     color='blue')
    ax[0][3].scatter(fit_cont_y,
                     fit_cont_x,
                     marker='*',
                     s=100,
                     color='green')
    ax[0][3].scatter(oiii_peak_y,
                     oiii_peak_x,
                     marker='+',
                     s=100,
                     color='purple')

    im = ax[0][3].imshow(cont1,
                      cmap=cmap,
                      vmax=0.4,
                      vmin=0.0)

    ax[0][3].tick_params(axis='x',
                      labelbottom='off')

    ax[0][3].tick_params(axis='y',
                      labelleft='off')


    ax[0][3].set_title('Cont1')

    # FLATFIELDED CONTINUUM

    ax[0][4].scatter(cont_peak_y,
                     cont_peak_x,
                     marker='x',
                     s=100,
                     color='blue')

    ax[0][4].scatter(fit_cont_y,
                     fit_cont_x,
                     marker='*',
                     s=100,
                     color='green')

    ax[0][4].scatter(oiii_peak_y,
                     oiii_peak_x,
                     marker='+',
                     s=100,
                     color='purple')


    im = ax[0][4].imshow(b_cont2,
                      cmap=cmap,
                      vmax=0.1,
                      vmin=-0.4)

    y_full, x_full = np.indices(b_cont2.shape)
    ax[0][4].contour(x_full,
                     y_full,
                     fit_cont,
                     4,
                     ls='solid',
                     colors='black')

    ax[0][4].tick_params(axis='x',
                      labelbottom='off')

    ax[0][4].tick_params(axis='y',
                      labelleft='off')


    ax[0][4].set_title('Cont2')

    # OIII NARROWBAND
    print 'OIII PEAK: %s %s' % (oiii_peak_x, oiii_peak_y)
    ax[1][0].scatter(oiii_peak_y,
                     oiii_peak_x,
                     marker='+',
                     s=100,
                     color='purple')

    ax[1][0].scatter(cont_peak_y,
                     cont_peak_x,
                     marker='x',
                     s=100,
                     color='blue')
    ax[1][0].scatter(fit_cont_y,
                     fit_cont_x,
                     marker='*',
                     s=100,
                     color='green')

    im = ax[1][0].imshow(o_nband,
                      cmap=cmap,
                      vmax=3,
                      vmin=-0.3)

    ax[1][0].tick_params(axis='x',
                      labelbottom='off')

    ax[1][0].tick_params(axis='y',
                      labelleft='off')


    ax[1][0].set_title('OIII')

    cmap = plt.cm.jet
    cmap.set_bad('black', 1.)

    # OIII FLUX

    ax[1][1].scatter(oiii_peak_y,
                     oiii_peak_x,
                     marker='+',
                     s=100,
                     color='purple')

    ax[1][1].scatter(cont_peak_y,
                     cont_peak_x,
                     marker='x',
                     s=100,
                     color='blue')
    ax[1][1].scatter(fit_cont_y,
                     fit_cont_x,
                     marker='*',
                     s=100,
                     color='green')

    im = ax[1][1].imshow(m_data_flux,
                      interpolation='nearest',
                      cmap=cmap)


    ax[1][1].tick_params(axis='x',
                      labelbottom='off')

    ax[1][1].tick_params(axis='y',
                      labelleft='off')


    # set the title
    ax[1][1].set_title('[OIII] Flux')

    divider = make_axes_locatable(ax[1][1])
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)

    # OIII VELOCITY

    ax[1][2].scatter(oiii_peak_y,
                     oiii_peak_x,
                     marker='+',
                     s=100,
                     color='purple')

    ax[1][2].scatter(cont_peak_y,
                     cont_peak_x,
                     marker='x',
                     s=100,
                     color='blue')
    ax[1][2].scatter(fit_cont_y,
                     fit_cont_x,
                     marker='*',
                     s=100,
                     color='green')

    im = ax[1][2].imshow(m_data_vel,
                      vmin=vel_min,
                      vmax=vel_max,
                      interpolation='nearest',
                      cmap=cmap)


    ax[1][2].tick_params(axis='x',
                      labelbottom='off')

    ax[1][2].tick_params(axis='y',
                      labelleft='off')


    # set the title
    ax[1][2].set_title('Velocity from data')

    divider = make_axes_locatable(ax[1][2])
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)

    ax[1][3].scatter(oiii_peak_y,
                     oiii_peak_x,
                     marker='+',
                     s=100,
                     color='purple')

    ax[1][3].scatter(cont_peak_y,
                     cont_peak_x,
                     marker='x',
                     s=100,
                     color='blue')
    ax[1][3].scatter(fit_cont_y,
                     fit_cont_x,
                     marker='*',
                     s=100,
                     color='green')

    im = ax[1][3].imshow(m_data_mod_blurred,
                      vmin=vel_min,
                      vmax=vel_max,
                      interpolation='nearest',
                      cmap=cmap)

    ax[1][3].tick_params(axis='x',
                      labelbottom='off')

    ax[1][3].tick_params(axis='y',
                      labelleft='off')

    # set the title
    ax[1][3].set_title('Velocity from model')

    divider = make_axes_locatable(ax[1][3])
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)

    # OIII DIspersion

    ax[1][4].scatter(oiii_peak_y,
                     oiii_peak_x,
                     marker='+',
                     s=100,
                     color='purple')

    ax[1][4].scatter(cont_peak_y,
                     cont_peak_x,
                     marker='x',
                     s=100,
                     color='blue')
    ax[1][4].scatter(fit_cont_y,
                     fit_cont_x,
                     marker='*',
                     s=100,
                     color='green')

    im = ax[1][4].imshow(m_data_sig,
                      vmin=sig_min,
                      vmax=sig_max,
                      interpolation='nearest',
                      cmap=cmap)


    ax[1][4].tick_params(axis='x',
                      labelbottom='off')

    ax[1][4].tick_params(axis='y',
                      labelleft='off')


    # set the title
    ax[1][4].set_title('Velocity Dispersion Data')

    divider = make_axes_locatable(ax[1][4])
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)

    # OIII DIspersion

    ax[2][4].scatter(oiii_peak_y,
                     oiii_peak_x,
                     marker='+',
                     s=100,
                     color='purple')

    ax[2][4].scatter(cont_peak_y,
                     cont_peak_x,
                     marker='x',
                     s=100,
                     color='blue')
    ax[2][4].scatter(fit_cont_y,
                     fit_cont_x,
                     marker='*',
                     s=100,
                     color='green')

    im = ax[2][4].imshow(m_data_vel_res,
                      vmin=vel_min,
                      vmax=vel_max,
                      interpolation='nearest',
                      cmap=cmap)


    ax[2][4].tick_params(axis='x',
                      labelbottom='off')

    ax[2][4].tick_params(axis='y',
                      labelleft='off')


    # set the title
    ax[2][4].set_title('Data - Model Vel')

    divider = make_axes_locatable(ax[2][4])
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)

    # 1D VELOCITY PLOT

    # at this point evaluate as well some 1D models
    # fit the extracted data along the different position
    # angles in 1D - will take seconds and provides a
    # chi-squared and different fit for each of the pas

    dyn_pa_fit = arc_mod.model_fit(one_d_data_vel,
                                   one_d_model_x,
                                   1. / one_d_data_vel_errors,
                                   va,
                                   rt)

    print 'DYN CHI: %s' % dyn_pa_fit.chisqr

    best_pa_fit = arc_mod.model_fit(best_pa_vel,
                                    best_pa_x,
                                    1. / best_pa_error,
                                    va,
                                    rt)

    print 'ROT CHI: %s' % best_pa_fit.chisqr

    hst_pa_fit = arc_mod.model_fit(hst_pa_vel,
                                   hst_pa_x,
                                   1. / hst_pa_error,
                                   va,
                                   rt)

    print 'HST CHI: %s' % hst_pa_fit.chisqr

    ax[2][0].set_ylabel(r'V$_{c}$[kms$^{-1}$]',
                      fontsize=10,
                      fontweight='bold')
    ax[2][0].set_xlabel(r'r [arcsec]',
                      fontsize=10,
                      fontweight='bold')
    # tick parameters 
    ax[2][0].tick_params(axis='both',
                       which='major',
                       labelsize=8,
                       length=6,
                       width=2)
    ax[2][0].tick_params(axis='both',
                       which='minor',
                       labelsize=8,
                       length=3,
                       width=1)

    ax[2][0].plot(one_d_model_x,
                  one_d_mod_vel_intrinsic,
                  color='red',
                  label='int\_model')

    ax[2][0].scatter(one_d_model_x,
                    one_d_mod_vel_intrinsic,
                    marker='o',
                    color='red')

    ax[2][0].plot(one_d_model_x,
                  one_d_mod_vel_blurred,
                  color='blue',
                  label='blurred\_model')

    ax[2][0].scatter(one_d_model_x,
                    one_d_mod_vel_blurred,
                    marker='o',
                  color='blue')

    ax[2][0].plot(one_d_model_x,
                  one_d_vel_res,
                  color='purple',
                  label='residuals')

    ax[2][0].scatter(one_d_model_x,
                    one_d_vel_res,
                    marker='^',
                  color='purple')

    ax[2][0].errorbar(one_d_model_x,
                   one_d_data_vel,
                   yerr=one_d_data_vel_errors,
                   fmt='+',
                   color='black',
                   label='data')

    ax[2][0].set_xlim(-1.5, 1.5)

    # ax[2][0].legend(prop={'size':5}, loc=1)

    ax[2][0].set_title('Model and observed Velocity')

    # ax[2][0].set_ylabel('velocity (kms$^{-1}$)')

    ax[2][0].set_xlabel('arcsec')

    ax[2][0].axhline(0, color='silver', ls='-.')
    ax[2][0].axvline(0, color='silver', ls='-.')
    ax[2][0].axhline(va, color='silver', ls='--')
    ax[2][0].axhline(-1.*va, color='silver', ls='--')

    # Also add in vertical lines for where the kinematics 
    # should be extracted

    ax[2][0].plot([r_e_arc, r_e_arc], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[2][0].plot([r_d_arc, r_d_arc], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[2][0].plot([r_d_arc_22, r_d_arc_22], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[2][0].plot([r_d_arc_30, r_d_arc_30], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[2][0].plot([-1*r_e_arc, -1*r_e_arc], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[2][0].plot([-r_d_arc, -r_d_arc], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[2][0].plot([-r_d_arc_22, -r_d_arc_22], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[2][0].plot([-r_d_arc_30, -r_d_arc_30], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)


    ax[2][0].minorticks_on()
    ax[2][0].set_xlabel('arcsec')
    leg = ax[2][0].legend(loc='upper left',fancybox=True, prop={'size':8})
    leg.get_frame().set_alpha(0.5)

    ax[2][1].set_ylabel(r'$\sigma$[kms$^{-1}$]',
                      fontsize=10,
                      fontweight='bold')
    ax[2][1].set_xlabel(r'r [arcsec]',
                      fontsize=10,
                      fontweight='bold')
    # tick parameters 
    ax[2][1].tick_params(axis='both',
                       which='major',
                       labelsize=8,
                       length=6,
                       width=2)
    ax[2][1].tick_params(axis='both',
                       which='minor',
                       labelsize=8,
                       length=3,
                       width=1)

    ax[2][1].minorticks_on()
    # 1D DISPERSION PLOT

    ax[2][1].errorbar(one_d_model_x,
                   one_d_data_sig,
                   yerr=one_d_data_sig_errors,
                   fmt='o',
                   color='red',
                   label='obs\_sig')

    ax[2][1].errorbar(one_d_model_x,
                   one_d_sig_int,
                   yerr=one_d_data_sig_errors,
                   fmt='o',
                   color='blue',
                   label='int\_sig')

    ax[2][1].plot(one_d_model_x,
                  one_d_sig_model_full,
                  color='orange',
                  label='sig\_model')

    ax[2][1].scatter(one_d_model_x,
                    one_d_sig_model_full,
                    marker='o',
                  color='orange')

    ax[2][1].plot(one_d_model_x,
                  one_d_sig_res,
                  color='purple')

    ax[2][1].scatter(one_d_model_x,
                    one_d_sig_res,
                    marker='o',
                  color='purple',
                  label='sig\_residuals')   

    ax[2][1].axvline(0, color='silver', ls='-.')
    ax[2][1].axvline(r_e_arc, color='maroon', ls='--', lw=2)
    ax[2][1].axvline(-1*r_e_arc, color='maroon', ls='--', lw=2)
    ax[2][1].axvline(r_d_arc, color='maroon', ls='--', lw=2)
    ax[2][1].axvline(-r_d_arc, color='maroon', ls='--', lw=2)
    ax[2][1].axvline(r_d_arc_22, color='maroon', ls='--', lw=2)
    ax[2][1].axvline(-r_d_arc_22, color='maroon', ls='--', lw=2)
    ax[2][1].axvline(r_d_arc_30, color='maroon', ls='--', lw=2)
    ax[2][1].axvline(-r_d_arc_30, color='maroon', ls='--', lw=2)
    ax[2][1].set_xlim(-1.5, 1.5)

    ax[2][1].set_title('Velocity Dispersion')

    # ax[2][1].set_ylabel('velocity (kms$^{-1}$)')

    ax[2][1].set_xlabel('arcsec')
    leg = ax[2][1].legend(loc='upper left',fancybox=True, prop={'size':8})
    leg.get_frame().set_alpha(0.5)


    # also want to fit a gaussian to the integrated spectrum to
    # determine emission line width. Surely the integrated sigma
    # is not a good measure of the turbulence as this will be higher
    # with higher velocity gradient?

    g_out, g_best, g_covar = one_d_g.ped_gauss_fit(obj_cube.wave_array[o_peak-50:o_peak+50],
                                            one_d_spectrum[o_peak-50:o_peak+50])

    gauss_spectrum = g_out.eval(x=obj_cube.wave_array[o_peak-50:o_peak+50])

    sigma_int = g_best['sigma']

    # also measure an error weighted sigma

    indices = ~np.isnan(data_sig)

    sigma_o = np.median(data_sig[indices])
                         

    indices = ~np.isnan(sig_int)

    sigma_o_i = np.median(sig_int[indices])
                           

    c = 2.99792458E5


    ax[2][2].plot(obj_cube.wave_array[o_peak-50:o_peak+50],
                  one_d_spectrum[o_peak-50:o_peak+50],
                  color='black')

    ax[2][2].plot(obj_cube.wave_array[o_peak-50:o_peak+50],
                  gauss_spectrum,
                  color='red')

    ax[2][2].axvline(central_l, color='red', ls='--')
    ax[2][2].axvline(obj_cube.wave_array[o_peak-5], color='red', ls='--')
    ax[2][2].axvline(obj_cube.wave_array[o_peak+5], color='red', ls='--')

    ax[2][2].set_title('Integrated Spectrum')

    ax[2][3].plot(pa_array,
                  stat_array,
                  color='black')

    ax[2][3].axvline(best_pa, color='darkorange', ls='--')

    if pa > np.pi:

        ax[2][3].axvline(pa - np.pi, color='lightcoral', ls='--')

    else:

        ax[2][3].axvline(pa, color='lightcoral', ls='--')

    if hst_pa > np.pi:

        ax[2][3].axvline(hst_pa - np.pi, color='aquamarine', ls='--')

    else:

        ax[2][3].axvline(hst_pa, color='aquamarine', ls='--')

    ax[2][3].set_title('PA Rotation')

    # plot the numerical fitting stuff
    # want to include on here in text what the
    # axis ratio and the PA are

    im = ax[3][0].imshow(num_cut_data,
                         vmax=5,
                         vmin=0)

    ax[3][0].text(1,2,
                  r'$\frac{b}{a} = %.2f$' % num_axis_ratio,
                  color='white',
                  fontsize=16)

    ax[3][0].text(1,6,
                  r'$pa = %.2f$' % num_pa,
                  color='white',
                  fontsize=16)

    y_full, x_full = np.indices(num_cut_data.shape)
    ax[3][0].contour(x_full,
                     y_full,
                     num_fit_data,
                     4,
                     ls='solid',
                     colors='black')

    # now plot the curve of growth parameters

    ax[3][1].plot(scaled_axis_array,
                  num_sum_array,
                  color='blue')
    ax[3][1].axvline(scaled_num_r_e, color='black',ls='--')
    ax[3][1].axvline(scaled_num_r_9, color='black',ls='--')
    ax[3][1].text(10, 50,
                  r'$R_{e} = %.2f$Kpc' % scaled_num_r_e,
                  color='black',
                  fontsize=16)
    ax[3][1].text(10, 500,
                  r'$R_{9} = %.2f$Kpc' % scaled_num_r_9,
                  color='black',
                  fontsize=16)

    ax[3][2].plot(one_d_model_x,
                  dyn_pa_fit.eval(r=one_d_model_x),
                  color='blue')

    ax[3][2].errorbar(one_d_model_x,
                   one_d_data_vel,
                   yerr=one_d_data_vel_errors,
                   fmt='o',
                   color='blue',
                   label='dyn\_pa')

    ax[3][2].plot(best_pa_x,
                  best_pa_fit.eval(r=best_pa_x),
                  color='darkorange')

    ax[3][2].errorbar(best_pa_x,
                      best_pa_vel,
                      yerr=best_pa_error,
                      fmt='o',
                      color='darkorange',
                      label='rot\_pa')

    ax[3][2].plot(hst_pa_x,
                  hst_pa_fit.eval(r=hst_pa_x),
                  color='aquamarine')

    ax[3][2].errorbar(hst_pa_x,
                      hst_pa_vel,
                      yerr=hst_pa_error,
                      fmt='o',
                      color='aquamarine',
                      label='hst\_pa')

    ax[3][2].set_title('Model and Real Velocity')

    # ax[3][2].set_ylabel('velocity (kms$^{-1}$)')

    ax[3][2].set_xlabel('arcsec')

    ax[3][2].axhline(0, color='silver', ls='-.')
    ax[3][2].axvline(0, color='silver', ls='-.')
    ax[3][2].axhline(va, color='silver', ls='--')
    ax[3][2].axhline(-1.*va, color='silver', ls='--')

    # Also add in vertical lines for where the kinematics 
    # should be extracted

    ax[3][2].plot([r_e_arc, r_e_arc], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].plot([r_d_arc, r_d_arc], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].plot([r_d_arc_22, r_d_arc_22], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].plot([r_d_arc_30, r_d_arc_30], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].plot([-1*r_e_arc, -1*r_e_arc], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].plot([-r_d_arc, -r_d_arc], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].plot([-r_d_arc_22, -r_d_arc_22], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].plot([-r_d_arc_30, -r_d_arc_30], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].set_xlim(-1.5, 1.5)
    leg = ax[3][2].legend(loc='upper left',fancybox=True, prop={'size':8})
    leg.get_frame().set_alpha(0.5)

    plt.suptitle('%s' % gal_name)

    # and the 1D plot showing the aperture growth


    fig.tight_layout()

    # some calculations for the final table

    # extracting the maximum velocity from the data
    data_velocity_value = (abs(np.nanmax(one_d_data_vel)) + \
                            abs(np.nanmin(one_d_data_vel))) / 2.0

    # and also want the associated velocity error
    minimum_vel_error = one_d_data_vel_errors[np.nanargmin(one_d_data_vel)]
    maximum_vel_error = one_d_data_vel_errors[np.nanargmax(one_d_data_vel)]


    # and combine in quadrature
    data_velocity_error = 0.5 * np.sqrt(minimum_vel_error**2 + maximum_vel_error**2)

    # sigma maps error
    # in quadrature take the last few values
    # in the actual data
    low_sigma_index, high_sigma_index = rt_pa.find_first_valid_entry(one_d_data_sig) 
    data_sigma_error = 0.5 * np.sqrt(one_d_data_sig_errors[low_sigma_index]**2 + one_d_data_sig_errors[high_sigma_index]**2)
    mean_sigma_error = np.nanmedian(one_d_data_sig_errors)

    # numerical value of sigma at the edges
    data_sigma_value = 0.5 * (one_d_data_sig[low_sigma_index] + one_d_data_sig[high_sigma_index])

    # sigma maps error
    # in quadrature take the last few values
    # in the actual data
    low_sigma_index_int, high_sigma_index_int = rt_pa.find_first_valid_entry(one_d_sig_int) 
    data_sigma_error_int = 0.5 * np.sqrt(one_d_data_sig_errors[low_sigma_index_int]**2 + one_d_data_sig_errors[high_sigma_index_int]**2)

    # numerical value of sigma at the edges
    data_sigma_value_int = 0.5 * (one_d_sig_int[low_sigma_index_int] + one_d_sig_int[high_sigma_index_int])


    b_data_velocity_value = (abs(np.nanmax(best_pa_vel)) + \
                              abs(np.nanmin(best_pa_vel))) / 2.0

    # and for the rotated position angle errors
    min_v_error_rpa = best_pa_error[np.nanargmin(best_pa_vel)]
    max_v_error_rpa = best_pa_error[np.nanargmax(best_pa_vel)]

    # and combine in quadrature
    rt_pa_observed_velocity_error = 0.5 * np.sqrt(min_v_error_rpa**2 + min_v_error_rpa**2)

    h_data_velocity_value = (abs(np.nanmax(hst_pa_vel)) + \
                              abs(np.nanmin(hst_pa_vel))) / 2.0

    max_data_velocity_value = np.nanmax(abs(one_d_data_vel))

    b_max_data_velocity_value = np.nanmax(abs(best_pa_vel))

    h_max_data_velocity_value = np.nanmax(abs(hst_pa_vel))

    # extract from both the 1d and 2d models at the special radii
    # defined as the 90 percent light and 1.8r_e and also 
    # find the radius at which the data extends to

    arc_num_r_9 = scaled_num_r_9 / scale

    # get the velocity indices

    extended_r = np.arange(-10, 10, 0.01)

    ex_r_22_idx = np.argmin(abs(r_d_arc_22 - extended_r))

    ex_3Rd_idx = np.argmin(abs(r_d_arc_30 - extended_r))

    ex_r_9_idx = np.argmin(abs(arc_num_r_9 - extended_r))

    one_d_model_x_r_22_idx = np.argmin(abs(r_d_arc_22 - one_d_model_x))

    one_d_model_x_3Rd_idx = np.argmin(abs(r_d_arc_30 - one_d_model_x))

    one_d_model_x_r_9_idx = np.argmin(abs(arc_num_r_9 - one_d_model_x))

    # find the associated velocity values from data

    d_extrapolation = dyn_pa_fit.eval(r=extended_r)

    b_extrapolation = best_pa_fit.eval(r=extended_r)

    h_extrapolation = hst_pa_fit.eval(r=extended_r)

    # need to know the constants in the fitting to subtract from
    # the inferred velocity values

    dyn_constant = dyn_pa_fit.best_values['const']

    rot_constant = best_pa_fit.best_values['const']

    hst_constant = hst_pa_fit.best_values['const']

    # and find the extrapolation values, sans constants

    dyn_v22 = d_extrapolation[ex_r_22_idx] - dyn_constant

    dyn_v3Rd = d_extrapolation[ex_3Rd_idx] - dyn_constant

    dyn_v9 = d_extrapolation[ex_r_9_idx] - dyn_constant

    b_v22 = b_extrapolation[ex_r_22_idx] - rot_constant

    b_v3Rd = b_extrapolation[ex_3Rd_idx] - rot_constant

    b_v9 = b_extrapolation[ex_r_9_idx] - rot_constant

    h_v22 = h_extrapolation[ex_r_22_idx] - hst_constant

    h_v9 = h_extrapolation[ex_r_22_idx] - hst_constant

    v_2d_r22 = one_d_mod_vel_intrinsic[one_d_model_x_r_22_idx]

    v_2d_3Rd = one_d_mod_vel_intrinsic[one_d_model_x_3Rd_idx]

    v_2d_r9 = one_d_mod_vel_intrinsic[one_d_model_x_r_9_idx]

    v_smeared_2d_r22 = one_d_mod_vel_blurred[one_d_model_x_r_22_idx]

    v_smeared_2d_3Rd = one_d_mod_vel_blurred[one_d_model_x_3Rd_idx]

    v_smeared_2d_r9 = one_d_mod_vel_blurred[one_d_model_x_r_9_idx]

    # also want to figure out the radius of the last velocity
    # point in the dyn, hst, rot extraction regimes

    s, e = rt_pa.find_first_valid_entry(one_d_data_vel)

    dyn_pa_extent = scale * np.nanmax([one_d_model_x[s], one_d_model_x[e]])

    s, e = rt_pa.find_first_valid_entry(best_pa_x)

    rot_pa_extent = scale * np.nanmax([best_pa_x[s], best_pa_x[e]])

    s, e = rt_pa.find_first_valid_entry(hst_pa_x)

    hst_pa_extent = scale * np.nanmax([hst_pa_x[s], hst_pa_x[e]])

    # assume for now that q = 0.15
    q = 0.2

    inclination_galfit = np.arccos(np.sqrt((axis_r**2 - q**2)/(1 - q**2)))

    inclination_num = np.arccos(np.sqrt((num_axis_ratio**2 - q**2)/(1 - q**2)))

    # comparing with Durham beam smearing values
    rd_psf = r_e_arc * (2.0 / seeing)

    # the velocity
    if abs(v_2d_3Rd / np.sin(inclination_galfit)) > 0 and abs(v_2d_3Rd / np.sin(inclination_galfit)) < 50:

        trigger = 1

    elif abs(v_2d_3Rd / np.sin(inclination_galfit)) > 50 and abs(v_2d_3Rd / np.sin(inclination_galfit)) < 100:

        trigger = 2

    elif abs(v_2d_3Rd / np.sin(inclination_galfit)) > 50 and abs(v_2d_3Rd / np.sin(inclination_galfit)) < 100:

        trigger = 3

    else:

        trigger = 4

    dur_vel_val = dur_smear.compute_velocity_smear_from_ratio(rd_psf, abs(v_smeared_2d_r22 / np.sin(inclination_galfit)))
    dur_vel_val_3 = dur_smear.compute_velocity_smear_from_ratio_3(rd_psf, abs(v_smeared_2d_3Rd / np.sin(inclination_galfit)))
    dur_mean_sig = dur_smear.compute_mean_sigma_smear_from_ratio(rd_psf, sigma_o , trigger)
    dur_outer_sig = dur_smear.compute_outer_sigma_smear_from_ratio(rd_psf, data_sigma_value , trigger)

    mdyn_22 = np.log10(((r_d_22 * 3.089E19 * (abs(v_2d_r22 / np.sin(inclination_galfit)) * 1000)**2) / 1.3267E20))
    mdyn_30 = np.log10(((r_d_30 * 3.089E19 * (abs(v_2d_3Rd / np.sin(inclination_galfit)) * 1000)**2) / 1.3267E20))

    data_values = [gal_name[26:-5],
                   abs(data_velocity_value / np.sin(inclination_galfit)),
                   data_velocity_error,
                   abs(b_data_velocity_value / np.sin(inclination_galfit)),
                   rt_pa_observed_velocity_error,
                   sigma_o,
                   mean_sigma_error,
                   sigma_o_i,
                   data_sigma_value,
                   data_sigma_error,
                   data_sigma_value_int,
                   data_sigma_error_int,
                   abs(dyn_v22 / np.sin(inclination_galfit)),
                   data_velocity_error,
                   abs(dyn_v3Rd / np.sin(inclination_galfit)),
                   data_velocity_error,
                   abs(b_v22 / np.sin(inclination_galfit)),
                   rt_pa_observed_velocity_error,
                   abs(b_v3Rd / np.sin(inclination_galfit)),
                   rt_pa_observed_velocity_error,
                   abs(v_smeared_2d_r22 / np.sin(inclination_galfit)),
                   data_velocity_error,
                   abs(v_smeared_2d_3Rd / np.sin(inclination_galfit)),
                   data_velocity_error,
                   abs(v_2d_r22 / np.sin(inclination_galfit)),
                   data_velocity_error,
                   abs(v_2d_3Rd / np.sin(inclination_galfit)),
                   data_velocity_error,
                   mdyn_22,
                   mdyn_30,
                   v_2d_3Rd / sigma_o_i,
                   dyn_pa_extent,
                   axis_r,
                   inclination_galfit,
                   hst_pa,
                   pa,
                   best_pa,
                   r_e_arc,
                   r_d_arc,
                   r_d_arc_22,
                   r_d_arc_30,
                   r_e,
                   r_d,
                   r_d_22,
                   r_d_30,
                   rd_psf,
                   dur_vel_val,
                   dur_vel_val_3,
                   dur_mean_sig,
                   dur_outer_sig]

    print 'CONSTANTS: %s %s %s' % (dyn_constant, rot_constant, hst_constant)
    print 'OBSERVED_VELOCITY_DYNAMIC_PA: %s' % abs(data_velocity_value / np.sin(inclination_galfit))
    print 'OBSERVED_VEL_ERROR_DYNAMIC_PA: %s' % data_velocity_error
    print 'OBSERVED_VELOCITY_ROTATED_PA: %s' % abs(b_data_velocity_value / np.sin(inclination_galfit))
    print 'OBSERVED_VELOCITY_ROTATED_PA_ERROR: %s' % rt_pa_observed_velocity_error
    print 'THESE ARE THE SIGMAS'
    print 'ROT SIGMA: %s' % best_mean_sigma
    print 'DYN SIGMA: %s' % dyn_mean_sigma
    print 'MEAN_OBSERVED_SIGMA: %s' % sigma_o
    print 'MEAN INTRINSIC SIGMA: %s' % sigma_o_i
    print 'MEAN SIGMA ERROR: %s' % mean_sigma_error
    print 'OBSERVED_SIGMA_DYNAMIC_EDGES: %s' % data_sigma_value
    print 'OBSERVED_SIGMA_ERROR: %s' % data_sigma_error
    print 'INTRINSIC_SIGMA_DYNAMIC_EDGES: %s' % data_sigma_value_int
    print 'OBSERVED_SIGMA_ERROR: %s' % data_sigma_error_int
    print '1D_ALONG_DYN_PA_1.8: %s' % abs(dyn_v22 / np.sin(inclination_galfit))
    print '1D_ALONG_DYN_PA_3Rd: %s' % abs(dyn_v3Rd / np.sin(inclination_galfit))
    print '1D_ALONG_ROTATED_PA_1.8: %s' % abs(b_v22 / np.sin(inclination_galfit))
    print '1D_ALONG_ROTATED_PA_3Rd: %s' % abs(b_v3Rd / np.sin(inclination_galfit))
    print '2D_ALONG_DYN_PA_1.8: %s' % abs(v_2d_r22 / np.sin(inclination_galfit))
    print '2D_ALONG_DYN_PA_3Rd: %s' % abs(v_2d_3Rd / np.sin(inclination_galfit))
    print 'AXIS RATIO: %s' % axis_r
    print 'GALFIT INCLINATION %s' % inclination_galfit
    print 'HST_PA: %s' % hst_pa
    if pa > np.pi:
        pa = pa - np.pi
    print 'DYN_PA: %s' % pa
    if best_pa > np.pi:
        best_pa = best_pa - np.pi
    print 'BEST_PA: %s' % best_pa
    print 'EFFECTIVE RADIUS: %s' % r_e_arc
    print 'EFFECTIVE RADIUS Kpcs %s' % r_e
    print 'Rd/RPSF: %s' % (r_e_arc * 6.72)

    plt.show()

    fig.savefig('%s_grid_chi_squared.png' % infile[:-5])


    return data_values

def multi_make_all_plots_fixed_inc_fixed(infile,
                                         r_aper,
                                         d_aper,
                                         seeing,
                                         sersic_n,
                                         sigma,
                                         pix_scale,
                                         psf_factor,
                                         sersic_factor,
                                         m_factor,
                                         smear=True):

    # create the table names

    column_names = ['Name',
                    'observed_vmax_dyn_pa',
                    'observed_vmax_dyn_pa_error',
                    'observed_vmax_rt_pa',
                    'observed_vmax_rt_pa_error',
                    'mean_observed_sigma',
                    'mean_sigma_error',
                    'mean_intrinsic_sigma',
                    'observed_sigma_edges',
                    'observed_sigma_edges_error',
                    'intrinsic_sigma_edges',
                    'intrinsic_sigma_edges_error',
                    '1d_dyn_pa_r_2.2',
                    '1d_dyn_pa_r_2.2_error',
                    '1d_dyn_pa_3Rd',
                    '1d_dyn_pa_3Rd_error',
                    '1d_rpa_r2.2',
                    '1d_rpa_r2.2_error',
                    '1d_rpa_3Rd',
                    '1d_rpa_3Rd_error',
                    '2d_beam_smeared_Vmax_r2.2',
                    '2d_beam_smeared_Vmax_r2.2_error',
                    '2d_beam_smeared_Vmax_3Rd',
                    '2d_beam_smeared_Vmax_3Rd_error',
                    '2d_intrinsic_Vmax_r2.2',
                    '2d_intrinsic_Vmax_r2.2_error',
                    '2d_intrinsic_Vmax_3Rd',
                    '2d_intrinsic_Vmax_3Rd_error',
                    'Mdyn_2.2',
                    'Mdyn_3.0',
                    'v_over_sigma',
                    'Last_data_radius',
                    'axis_ratio',
                    'inclination',
                    'HST_PA',
                    'DYN_PA',
                    'R_PA',
                    'R_e(arcsec)',
                    'R_d(arcsec)',
                    '2.2R_d(arcsec)',
                    '3R_d(arcsec)',
                    'R_e(Kpc)',
                    'R_d(Kpc)',
                    '2.2R_d(Kpc)',
                    '3R_d(Kpc)',
                    'Rd/Rpsf',
                    'Durham_Vel_corr_2Rd',
                    'Durham_Vel_corr_3Rd',
                    'Durham_Sig_mean_corr',
                    'Durham_Sig_outer_corr']


    save_dir = '/disk2/turner/disk1/turner/DATA/kmos_dynamics_paper_plots/'

    big_list = []

    # read in the table of cube names
    Table = ascii.read(infile)

    # assign variables to the different items in the infile
    for entry in Table:

        obj_name = entry[0]

        cube = cubeOps(obj_name)

        xpix = cube.data.shape[1]

        ypix = cube.data.shape[2]

        wave_array = cube.wave_array

        redshift = entry[1]

        xcen = entry[10]

        ycen = entry[11]

        inc = entry[12]

        r_e = entry[16]

        sersic_pa = entry[17]

        a_r = np.sqrt((np.cos(inc) * np.cos(inc)) * (1 - (0.2**2)) + 0.2 ** 2)

        sersic_field = psf.sersic_2d_astropy(dim_x=ypix,
                                             dim_y=xpix,
                                             rt=r_e,
                                             n=1.0,
                                             a_r=a_r,
                                             pa=sersic_pa,
                                             xcen=xcen,
                                             ycen=ycen,
                                             sersic_factor=sersic_factor)

        big_list.append(make_all_plots_fixed_inc_fixed(inc,
                                                       redshift,
                                                        wave_array,
                                                        xcen,
                                                        ycen,
                                                        obj_name,
                                                        r_aper,
                                                        d_aper,
                                                        seeing,
                                                        sersic_n,
                                                        sigma,
                                                        pix_scale,
                                                        psf_factor,
                                                        sersic_factor,
                                                        m_factor,
                                                        sersic_field,
                                                        smear))
    
    # create the table
    make_table.table_create(column_names,
                            big_list,
                            save_dir,
                            'goods_isolated_rotator_properties_test.cat')

def make_all_plots_mcmc_version(inc,
                                redshift,
                                wave_array,
                                xcen,
                                ycen,
                                infile,
                                r_aper,
                                d_aper,
                                seeing,
                                sersic_n,
                                sigma,
                                pix_scale,
                                psf_factor,
                                sersic_factor,
                                m_factor,
                                light_profile,
                                galaxy_boundaries_file,
                                gal_num,
                                hst_x_cen,
                                hst_y_cen,
                                inc_error=0.1,
                                sig_error=10,
                                smear=True):

    """
    Def: Take all of the data from the stott velocity fields,
    mcmc modelling and hst imaging and return a grid of plots
    summarising the results.

    Input:
            in_file - file path and name of object

    Output:
            grid of plots
    """

    # Get the conversion between arcseconds and kpc at this redshift

    from astropy.cosmology import WMAP9 as cosmo

    scale = cosmo.kpc_proper_per_arcmin(redshift).value / 60.0

    # open the various files and run the methods to get the data
    # for plotting

    gal_name = infile[len(infile) -
                      infile[::-1].find("/"):]

    # instead of reading in the chi_squared params, read in the
    # mcmc evaluations

    param_file = np.genfromtxt('%s_vel_field_params_fixed_inc_fixed.txt' % infile[:-5])

    theta_max = param_file[1][1:]

    pa_max = theta_max[0]

    rt_max = theta_max[1]

    va_max = theta_max[2]

    theta_50 = param_file[2][1:]

    pa_50 = theta_50[0]

    # also take this as the canonical pa

    pa = theta_50[0]

    rt_50 = theta_50[1]

    # also take this as the canonical rt

    rt = theta_50[1]

    va_50 = theta_50[2]

    # also take this as the canonical va

    va = theta_50[2]

    theta_16 = param_file[3][1:]

    pa_16 = theta_16[0]

    rt_16 = theta_16[1]

    va_16 = theta_16[2]

    theta_84 = param_file[4][1:]

    pa_84 = theta_84[0]

    rt_84 = theta_84[1]

    va_84 = theta_84[2]

    # also need to compute the maximum and minimum inclinations
    # or the 1 sigma limits on inclination. Assume a constant fractional
    # error in inclination estimation and compute upper and lower values

    inc_16 = inc + (inc_error * inc)
    inc_16_factor = 1. / (np.sin(inc_16) / np.sin(inc))

    inc_84 = inc - (inc_error * inc)
    inc_84_factor = 1. / (np.sin(inc_84) / np.sin(inc))

    # and now set the theta values
    # these will go into later 
    # evaluations of all physical 
    # properties. It will be exactly
    # the same as the case for the single
    # evaluations, just with 4 blocks of
    # code instead.

    theta_max = [pa_max, rt_max, va_max]

    theta_50 = [pa_50, rt_50, va_50]

    # swapping around the rt values,
    # and using the 50th percentile pa value

    theta_16 = [pa_50, rt_84, va_16 * inc_16_factor]

    theta_84 = [pa_50, rt_16, va_84 * inc_84_factor]

    hst_stamp_name = infile[:-5] + '_galfit.fits'

    table_hst = fits.open(hst_stamp_name)

    # do the initial numerical fitting to find the 
    # half light radius and other quantities

    table = ascii.read(galaxy_boundaries_file)

    xl = table['xl'][gal_num]
    xh = table['xh'][gal_num]
    yl = table['yl'][gal_num]
    yh = table['yh'][gal_num]

    half_light_dict = ap_growth.find_aperture_parameters(hst_stamp_name,
                                                         xl,
                                                         xh,
                                                         yl,
                                                         yh)

    # assign the parameters from the dictionary
    # where prefix num refers to the fact that
    # this has been done numerically

    num_cut_data = half_light_dict['cut_data']
    num_fit_data = half_light_dict['fit_data']
    num_axis_array = half_light_dict['a_array']
    num_sum_array = half_light_dict['sum_array']
    num_axis_ratio = half_light_dict['axis_ratio']
    num_r_e = half_light_dict['r_e_pixels']
    num_r_9 = half_light_dict['r_9_pixels']
    num_pa = half_light_dict['pa'] + np.pi / 2.0

    scaled_axis_array = 0.06 * scale * num_axis_array
    scaled_num_r_e = 0.06 * scale * num_r_e
    scaled_num_r_9 = 0.06 * scale * num_r_9

    galfit_mod = table_hst[2].data

    galfit_res = table_hst[3].data

    # Get thhe galfit axis ratio

    axis_r_str = table_hst[2].header['1_AR']

    axis_r = axis_r_str[:len(axis_r_str) -
                    axis_r_str[::-1].find("+") - 2]

    if axis_r[0] == '[':

        axis_r = axis_r[1:]

    # If the parameter has not been well determined by galfit
    # need to account for the asterisks

    if axis_r[0] == '*':

        axis_r = axis_r[1:-1]

    axis_r = float(axis_r)

    # Get the galfit scale radius

    r_e_str = table_hst[2].header['1_RE']

    r_e = r_e_str[:len(r_e_str) -
                    r_e_str[::-1].find("+") - 2]

    if r_e[0] == '[':

        r_e = r_e[1:]

    if r_e[0] == '*':

        r_e = r_e[1:-1]


    # since r_e has been measured using HST which has a pixel scale
    # of 0.06, need to multiply by the ratio of this to the actual
    # pixel_scale for KMOS which is 0.6. This is working in
    # arcseconds - should have realised this a long time ago

    r_e = float(r_e) * 0.6

    # This r_e is in pixels, multiply 
    # by 0.1 and then by scale to put into KPC

    r_e = pix_scale * r_e * scale

    r_d = r_e / 1.67835

    r_d_22 = 2.2 * r_d

    r_d_30 = 3.4 * r_d

    # Converting back to arcseconds

    r_e_arc = r_e / scale

    # to make extractions need to convolve with the KMOS seeing

    r_e_arc_conv = 1.1774*np.sqrt((r_e_arc/1.1774)**2 + (seeing/2.35482)**2)

    print 'RE ARC AND RE ARC CONV: %s %s' % (r_e_arc, r_e_arc_conv)

    r_d_arc = r_e_arc / 1.67835

    r_d_arc_conv = r_e_arc_conv / 1.67835

    r_d_arc_22 = 2.2 * r_d_arc

    r_d_arc_conv_22 = 2.2 * r_d_arc_conv

    r_d_arc_30 = 3.4 * r_d_arc

    r_d_arc_conv_30 = 3.4 * r_d_arc_conv

    # and also add the kiloparsec values

    r_d_conv = r_d_arc_conv * scale

    r_d_conv_22 = r_d_arc_conv_22 * scale

    r_d_conv_30 = r_d_arc_conv_30 * scale

    hst_pa_str = table_hst[2].header['1_PA']

    hst_pa = hst_pa_str[:len(hst_pa_str) -
                              hst_pa_str[::-1].find("+") - 2]

    if hst_pa[0] == '[':

        hst_pa = hst_pa[1:]

    if hst_pa[0] == '*':

        hst_pa = hst_pa[1:-1]

    hst_pa = float(hst_pa)

    # convert between degrees and radians

    if hst_pa < 0:

        hst_pa = hst_pa + 180

    hst_pa = (hst_pa * np.pi) / 180

    data_hst = table_hst[1].data

    flux_name = infile[:-5] + '_flux_field.fits'

    table_flux = fits.open(flux_name)

    data_flux = table_flux[0].data

    vel_field_name = infile[:-5] + '_vel_field.fits'

    table_vel = fits.open(vel_field_name)

    data_vel = table_vel[0].data

    table_error = fits.open('%s_error_field.fits' % infile[:-5])

    error_vel = table_error[0].data

    vel = vel_field(vel_field_name,
                    xcen,
                    ycen)

    xpix = data_vel.shape[0]

    ypix = data_vel.shape[1]

    data_model_max = compute_model_grid_for_chi_squared(xpix,
                                                    ypix,
                                                    theta_max,
                                                    inc,
                                                    redshift,
                                                    wave_array,
                                                    xcen,
                                                    ycen,
                                                    seeing,
                                                    sersic_n,
                                                    sigma,
                                                    pix_scale,
                                                    psf_factor,
                                                    sersic_factor,
                                                    m_factor,
                                                    light_profile,
                                                    smear)

    mod_vel_max, mod_vel_max_blurred, sig_max_blurred = data_model_max

    data_model_50 = compute_model_grid_for_chi_squared(xpix,
                                                    ypix,
                                                    theta_50,
                                                    inc,
                                                    redshift,
                                                    wave_array,
                                                    xcen,
                                                    ycen,
                                                    seeing,
                                                    sersic_n,
                                                    sigma,
                                                    pix_scale,
                                                    psf_factor,
                                                    sersic_factor,
                                                    m_factor,
                                                    light_profile,
                                                    smear)

    mod_vel_50, mod_vel_50_blurred, sig_50_blurred = data_model_50

    data_model_16 = compute_model_grid_for_chi_squared(xpix,
                                                    ypix,
                                                    theta_16,
                                                    inc,
                                                    redshift,
                                                    wave_array,
                                                    xcen,
                                                    ycen,
                                                    seeing,
                                                    sersic_n,
                                                    sigma,
                                                    pix_scale,
                                                    psf_factor,
                                                    sersic_factor,
                                                    m_factor,
                                                    light_profile,
                                                    smear)

    mod_vel_16, mod_vel_16_blurred, sig_16_blurred = data_model_16

    data_model_84 = compute_model_grid_for_chi_squared(xpix,
                                                    ypix,
                                                    theta_84,
                                                    inc,
                                                    redshift,
                                                    wave_array,
                                                    xcen,
                                                    ycen,
                                                    seeing,
                                                    sersic_n,
                                                    sigma,
                                                    pix_scale,
                                                    psf_factor,
                                                    sersic_factor,
                                                    m_factor,
                                                    light_profile,
                                                    smear)

    mod_vel_84, mod_vel_84_blurred, sig_84_blurred = data_model_84

    # truncate this to the data size

    mask_array = np.empty(shape=(xpix, ypix))

    for i in range(0, xpix):

        for j in range(0, ypix):

            if np.isnan(data_vel[i][j]):

                mask_array[i][j] = np.nan

            else:

                mask_array[i][j] = 1.0



    table_sig = fits.open('%s_sig_field.fits' % infile[:-5])

    data_sig = table_sig[0].data

    table_sig_error = fits.open('%s_sig_error_field.fits' % infile[:-5])

    data_sig_error = table_sig_error[0].data

    idv_param_file = infile[:-5] + '_chi_squared_params.txt'

    # This is where it gets complicated.
    # need to find all previous references to the mod_vel, mod_vel_blurred
    # and sig_blurred

    one_d_plots, extract_values = extract_in_apertures_mcmc_version(infile,
                                                                    theta_50,
                                                                    inc,
                                                                    redshift,
                                                                    wave_array,
                                                                    xcen,
                                                                    ycen,
                                                                    r_aper,
                                                                    d_aper,
                                                                    seeing,
                                                                    sersic_n,
                                                                    sigma,
                                                                    pix_scale,
                                                                    psf_factor,
                                                                    sersic_factor,
                                                                    m_factor,
                                                                    light_profile,
                                                                    smear)

    one_d_mod_vel_intrinsic = one_d_plots['one_d_mod_vel_intrinsic']
    one_d_mod_vel_blurred = one_d_plots['one_d_mod_vel_blurred']
    one_d_data_vel = one_d_plots['one_d_data_vel']
    one_d_vel_res = one_d_plots['one_d_vel_res']
    one_d_data_vel_errors = one_d_plots['one_d_data_vel_errors']
    one_d_data_sig = one_d_plots['one_d_data_sig']
    one_d_sig_int = one_d_plots['one_d_sig_int']
    one_d_sig_model_full = one_d_plots['one_d_sig_full_model']
    one_d_sig_res = one_d_plots['one_d_sig_res']
    one_d_data_sig_errors = one_d_plots['one_d_data_sig_errors']
    one_d_data_x = one_d_plots['one_d_data_x']
    one_d_model_x = one_d_plots['one_d_model_x']
    vel_res = one_d_plots['vel_res']
    sig_int = one_d_plots['intrinsic_sig']
    sig_full_model = one_d_plots['sig_full_model']
    sig_res = one_d_plots['sig_res']

    one_d_plots_max, extract_values_max = extract_in_apertures_mcmc_version(infile,
                                                                    theta_max,
                                                                    inc,
                                                                    redshift,
                                                                    wave_array,
                                                                    xcen,
                                                                    ycen,
                                                                    r_aper,
                                                                    d_aper,
                                                                    seeing,
                                                                    sersic_n,
                                                                    sigma,
                                                                    pix_scale,
                                                                    psf_factor,
                                                                    sersic_factor,
                                                                    m_factor,
                                                                    light_profile,
                                                                    smear)

    one_d_mod_vel_intrinsic_max = one_d_plots_max['one_d_mod_vel_intrinsic']
    one_d_mod_vel_blurred_max = one_d_plots_max['one_d_mod_vel_blurred']
    one_d_data_vel_max = one_d_plots_max['one_d_data_vel']
    one_d_vel_res_max = one_d_plots_max['one_d_vel_res']
    one_d_data_vel_errors_max = one_d_plots_max['one_d_data_vel_errors']
    one_d_data_sig_max = one_d_plots_max['one_d_data_sig']
    one_d_sig_int_max = one_d_plots_max['one_d_sig_int']
    one_d_sig_model_full_max = one_d_plots_max['one_d_sig_full_model']
    one_d_sig_res_max = one_d_plots_max['one_d_sig_res']
    one_d_data_sig_errors_max = one_d_plots_max['one_d_data_sig_errors']
    one_d_data_x_max = one_d_plots_max['one_d_data_x']
    one_d_model_x_max = one_d_plots_max['one_d_model_x']
    vel_res_max = one_d_plots_max['vel_res']
    sig_int_max = one_d_plots_max['intrinsic_sig']
    sig_full_model_max = one_d_plots_max['sig_full_model']
    sig_res_max = one_d_plots_max['sig_res']

    one_d_plots_16, extract_values_16 = extract_in_apertures_mcmc_version(infile,
                                                                    theta_16,
                                                                    inc,
                                                                    redshift,
                                                                    wave_array,
                                                                    xcen,
                                                                    ycen,
                                                                    r_aper,
                                                                    d_aper,
                                                                    seeing,
                                                                    sersic_n,
                                                                    sigma,
                                                                    pix_scale,
                                                                    psf_factor,
                                                                    sersic_factor,
                                                                    m_factor,
                                                                    light_profile,
                                                                    smear)

    one_d_mod_vel_intrinsic_16 = one_d_plots_16['one_d_mod_vel_intrinsic']
    one_d_mod_vel_blurred_16 = one_d_plots_16['one_d_mod_vel_blurred']
    one_d_data_vel_16 = one_d_plots_16['one_d_data_vel']
    one_d_vel_res_16 = one_d_plots_16['one_d_vel_res']
    one_d_data_vel_errors_16 = one_d_plots_16['one_d_data_vel_errors']
    one_d_data_sig_16 = one_d_plots_16['one_d_data_sig']
    one_d_sig_int_16 = one_d_plots_16['one_d_sig_int']
    one_d_sig_model_full_16 = one_d_plots_16['one_d_sig_full_model']
    one_d_sig_res_16 = one_d_plots_16['one_d_sig_res']
    one_d_data_sig_errors_16 = one_d_plots_16['one_d_data_sig_errors']
    one_d_data_x_16 = one_d_plots_16['one_d_data_x']
    one_d_model_x_16 = one_d_plots_16['one_d_model_x']
    vel_res_16 = one_d_plots_16['vel_res']
    sig_int_16 = one_d_plots_16['intrinsic_sig']
    sig_full_model_16 = one_d_plots_16['sig_full_model']
    sig_res_16 = one_d_plots_16['sig_res']

    one_d_plots_16_80, extract_values_16_80 = extract_in_apertures_mcmc_version(infile,
                                                                    theta_16,
                                                                    inc,
                                                                    redshift,
                                                                    wave_array,
                                                                    xcen,
                                                                    ycen,
                                                                    r_aper,
                                                                    d_aper,
                                                                    seeing,
                                                                    sersic_n,
                                                                    80,
                                                                    pix_scale,
                                                                    psf_factor,
                                                                    sersic_factor,
                                                                    m_factor,
                                                                    light_profile,
                                                                    smear)

    sig_int_16_80 = one_d_plots_16_80['intrinsic_sig']

    one_d_plots_16_40, extract_values_16_40 = extract_in_apertures_mcmc_version(infile,
                                                                    theta_16,
                                                                    inc,
                                                                    redshift,
                                                                    wave_array,
                                                                    xcen,
                                                                    ycen,
                                                                    r_aper,
                                                                    d_aper,
                                                                    seeing,
                                                                    sersic_n,
                                                                    40,
                                                                    pix_scale,
                                                                    psf_factor,
                                                                    sersic_factor,
                                                                    m_factor,
                                                                    light_profile,
                                                                    smear)

    sig_int_16_40 = one_d_plots_16_40['intrinsic_sig']


    one_d_plots_84, extract_values_84 = extract_in_apertures_mcmc_version(infile,
                                                                          theta_84,
                                                                          inc,
                                                                          redshift,
                                                                          wave_array,
                                                                          xcen,
                                                                          ycen,
                                                                          r_aper,
                                                                          d_aper,
                                                                          seeing,
                                                                          sersic_n,
                                                                          sigma,
                                                                          pix_scale,
                                                                          psf_factor,
                                                                          sersic_factor,
                                                                          m_factor,
                                                                          light_profile,
                                                                          smear)

    one_d_mod_vel_intrinsic_84 = one_d_plots_84['one_d_mod_vel_intrinsic']
    one_d_mod_vel_blurred_84 = one_d_plots_84['one_d_mod_vel_blurred']
    one_d_data_vel_84 = one_d_plots_84['one_d_data_vel']
    one_d_vel_res_84 = one_d_plots_84['one_d_vel_res']
    one_d_data_vel_errors_84 = one_d_plots_84['one_d_data_vel_errors']
    one_d_data_sig_84 = one_d_plots_84['one_d_data_sig']
    one_d_sig_int_84 = one_d_plots_84['one_d_sig_int']
    one_d_sig_model_full_84 = one_d_plots_84['one_d_sig_full_model']
    one_d_sig_res_84 = one_d_plots_84['one_d_sig_res']
    one_d_data_sig_errors_84 = one_d_plots_84['one_d_data_sig_errors']
    one_d_data_x_84 = one_d_plots_84['one_d_data_x']
    one_d_model_x_84 = one_d_plots_84['one_d_model_x']
    vel_res_84 = one_d_plots_84['vel_res']
    sig_int_84 = one_d_plots_84['intrinsic_sig']
    sig_full_model_84 = one_d_plots_84['sig_full_model']
    sig_res_84 = one_d_plots_84['sig_res']

    one_d_plots_84_80, extract_values_84_80 = extract_in_apertures_mcmc_version(infile,
                                                                    theta_84,
                                                                    inc,
                                                                    redshift,
                                                                    wave_array,
                                                                    xcen,
                                                                    ycen,
                                                                    r_aper,
                                                                    d_aper,
                                                                    seeing,
                                                                    sersic_n,
                                                                    80,
                                                                    pix_scale,
                                                                    psf_factor,
                                                                    sersic_factor,
                                                                    m_factor,
                                                                    light_profile,
                                                                    smear)

    sig_int_84_80 = one_d_plots_84_80['intrinsic_sig']

    one_d_plots_84_40, extract_values_84_40 = extract_in_apertures_mcmc_version(infile,
                                                                    theta_84,
                                                                    inc,
                                                                    redshift,
                                                                    wave_array,
                                                                    xcen,
                                                                    ycen,
                                                                    r_aper,
                                                                    d_aper,
                                                                    seeing,
                                                                    sersic_n,
                                                                    40,
                                                                    pix_scale,
                                                                    psf_factor,
                                                                    sersic_factor,
                                                                    m_factor,
                                                                    light_profile,
                                                                    smear)

    sig_int_84_40 = one_d_plots_84_40['intrinsic_sig']


    # take product of model and mask_array to return new data

    mod_vel_masked = mod_vel_50 * mask_array
    mod_vel_blurred_masked = mod_vel_50_blurred * mask_array
    vel_res_masked = vel_res * mask_array
    sig_blurred_masked = sig_50_blurred * mask_array
    sig_int_masked = sig_int * mask_array
    sig_full_model_masked = sig_int * mask_array
    sig_res_masked = sig_res * mask_array

    # set the imshow plotting limits
    vel_min, vel_max = np.nanpercentile(mod_vel_blurred_masked,
                                        [5.0, 95.0])

    dat_vel_min, dat_vel_max = np.nanpercentile(data_vel,
                                        [5.0, 95.0])

    sig_min, sig_max = np.nanpercentile(sig_int,
                                        [5.0, 95.0])

    d_for_mask = copy(data_vel)

    # get the continuum images and narrow band OIII
    cont_dict = f_f.flatfield(infile,
                              d_for_mask,
                              redshift)

    cont1 = cont_dict['cont1']
    cont2 = cont_dict['cont2']
    o_nband = cont_dict['OIII']

    # CUT DOWN ALL OF THE DATA WE HAVE TO GET RID OF SPAXELS 
    # ON THE OUTSKIRTS - 3 SPAXELS IN KMOS AND 5 in HST

    # smooth the continuum image

    b_cont2 = psf.blur_by_psf(cont2,
                              0.3,
                              pix_scale,
                              psf_factor)

    # for gaussian fitting only want to use the pixels
    # which have been accepted in the stott fitting

    g_mask = np.empty(shape=(data_vel.shape[0],
                             data_vel.shape[1]))

    print 'This is the mask shape: %s %s' % (g_mask.shape[0], g_mask.shape[1])

    for i in range(data_vel.shape[0]):

        for j in range(data_vel.shape[1]):

            if np.isnan(data_vel[i, j]):

                g_mask[i, j] = np.nan

            else:

                g_mask[i, j] = 1.0

    fit_b_cont2 = g_mask * b_cont2

    # attempt to fit the continuum with a gaussian

    fit_cont, fit_params = g2d.fit_gaussian(fit_b_cont2)

    # and assign the center coordinates

    fit_cont_x = fit_params[3]
    fit_cont_y = fit_params[2]

    print 'These are the fit center coordinates: %s %s' % (fit_cont_x, fit_cont_y) 

    # Also get estimate of the center using peak pixel within
    # that masking region

    cont_peak_coords = np.unravel_index(np.nanargmax(fit_b_cont2),
                                        fit_b_cont2.shape)

    cont_peak_x = cont_peak_coords[0]
    cont_peak_y = cont_peak_coords[1]

    print 'These are the continuum peak: %s %s' % (cont_peak_x, cont_peak_y) 

    fit_o_nband = g_mask * o_nband

    oiii_peak_coords = np.unravel_index(np.nanargmax(fit_o_nband),
                                        fit_o_nband.shape)

    oiii_peak_x = oiii_peak_coords[0]
    oiii_peak_y = oiii_peak_coords[1]

    print 'This is the OIII peak: %s %s' % (oiii_peak_x, oiii_peak_y)

    hst_fit, hst_fit_params = g2d.fit_gaussian(galfit_mod)

    # 1D spectrum finding - note this will be unweighted
    # and not as good as a weighted version which doesn't have
    # as large a contribution from the outside spaxels
    # using the spaxels in the mask

    obj_cube = cubeOps(infile)
    one_d_spectrum = []

    #  OIII wavelength
    central_l = (1 + redshift) * 0.500824

    o_peak = np.argmin(abs(central_l - wave_array))

    for i in range(data_vel.shape[0]):

        for j in range(data_vel.shape[1]):

            if not(np.isnan(g_mask[i, j])):

                one_d_spectrum.append(obj_cube.data[:, i, j])

    # sum for final spectrum
    one_d_spectrum = np.nansum(one_d_spectrum, axis=0)

    # Now have all information to define an astropy elliptical
    # aperture using the galfit parameters

    from photutils import EllipticalAperture

    theta = hst_pa + np.pi / 2.0
    major_axis = r_e_arc * 10
    minor_axis = major_axis * axis_r
    galfit_x = hst_fit_params[3]
    galfit_y = hst_fit_params[2]
    positions = [(galfit_y, galfit_x)]

    print 'This is the axis ratio: %s' % axis_r

    apertures = EllipticalAperture(positions,
                                   major_axis,
                                   minor_axis,
                                   theta)

    disk_apertures = EllipticalAperture(positions,
                                        1.8*major_axis,
                                        1.8*minor_axis,
                                        theta)

    # Now compute alternate PA from rotating the slit until
    # it maximises the velocity difference

    best_pa, pa_array, stat_array = rt_pa.rot_pa(d_aper,
                                                 r_aper,
                                                 data_vel, 
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

    # Also use the rt_pa method to extract the 1D spectrum
    # and errors along the HST and BEST pa's
    # these will be plotted along with the dynamical PA plots
    print 'THESE ARE THE PAS'
    print 'HST: %s' % hst_pa
    print 'ROT: %s' % best_pa
    print 'DYN: %s' % pa

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.imshow(data_vel)
    plt.close('all')

    # extract the velocity data in each case of position angle

    hst_pa_vel, hst_pa_x = rt_pa.extract(d_aper,
                                         r_aper,
                                         hst_pa,
                                         data_vel, 
                                         xcen,
                                         ycen,
                                         pix_scale)

    hst_pa_error, hst_pa_x = rt_pa.extract(d_aper,
                                           r_aper,
                                           hst_pa,
                                           error_vel, 
                                           xcen,
                                           ycen,
                                           pix_scale)

    best_pa_vel, best_pa_x = rt_pa.extract(d_aper,
                                         r_aper,
                                         best_pa,
                                         data_vel, 
                                         xcen,
                                         ycen,
                                         pix_scale)

    best_pa_error, best_pa_x = rt_pa.extract(d_aper,
                                           r_aper,
                                           best_pa,
                                           error_vel, 
                                           xcen,
                                           ycen,
                                           pix_scale)

    # extract the sigma data in each case of position angle

    hst_pa_sig, hst_pa_sig_x = rt_pa.extract(d_aper,
                                             r_aper,
                                             hst_pa,
                                             data_sig, 
                                             xcen,
                                             ycen,
                                             pix_scale)

    hst_pa_sig_error, hst_pa_sig_x = rt_pa.extract(d_aper,
                                                   r_aper,
                                                   hst_pa,
                                                   data_sig_error, 
                                                   xcen,
                                                   ycen,
                                                   pix_scale)

    best_pa_sig, best_pa_sig_x = rt_pa.extract(d_aper,
                                               r_aper,
                                               best_pa,
                                               data_sig, 
                                               xcen,
                                               ycen,
                                               pix_scale)

    best_pa_sig_error, best_pa_sig_x = rt_pa.extract(d_aper,
                                                     r_aper,
                                                     best_pa,
                                                     data_sig_error, 
                                                     xcen,
                                                     ycen,
                                                     pix_scale)

    dyn_pa_sig, dyn_pa_sig_x = rt_pa.extract(d_aper,
                                               r_aper,
                                               pa,
                                               data_sig, 
                                               xcen,
                                               ycen,
                                               pix_scale)

    dyn_pa_sig_error, dyn_pa_sig_x = rt_pa.extract(d_aper,
                                                     r_aper,
                                                     pa,
                                                     data_sig_error, 
                                                     xcen,
                                                     ycen,
                                                     pix_scale)

    # now want to take the average of the first and last sigma
    # values extracted along each of the position angles

    # HST POSITION ANGLE
    hst_i, hst_j = rt_pa.find_first_valid_entry(hst_pa_sig)

    hst_mean_sigma = np.nanmean([hst_pa_sig[hst_i],hst_pa_sig[hst_j]])

    # DYN POSITION ANGLE
    dyn_i, dyn_j = rt_pa.find_first_valid_entry(dyn_pa_sig)

    dyn_mean_sigma = np.nanmean([dyn_pa_sig[dyn_i],dyn_pa_sig[dyn_j]])

    # HST POSITION ANGLE
    best_i, best_j = rt_pa.find_first_valid_entry(best_pa_sig)

    best_mean_sigma = np.nanmean([best_pa_sig[best_i],best_pa_sig[best_j]])

    # calculate the boundaries from which to draw a line
    # through the images relating to the position angles

    x_inc_hst = 100 * np.abs(np.cos(hst_pa))
    y_inc_hst = 100 * np.abs(np.sin(hst_pa))

    # Find the boundaries for plotting the PAs
    # Use the continuum center in order to do this

    if 0 < hst_pa < np.pi / 2.0 or np.pi < hst_pa < 3 * np.pi / 2.0:

        # in the top right and bottom left areas
        # so adding to x goes with subtracting from y

        x_h_low = xcen + x_inc_hst
        x_h_high = xcen - x_inc_hst
        y_h_low = ycen - y_inc_hst
        y_h_high = ycen + y_inc_hst

    else:

        x_h_low = xcen - x_inc_hst
        x_h_high = xcen + x_inc_hst
        y_h_low = ycen - y_inc_hst
        y_h_high = ycen + y_inc_hst

    # calculate the boundaries from which to draw a line
    # through the images relating to the position angles

    x_inc = 100 * np.abs(np.cos(pa))
    y_inc = 100 * np.abs(np.sin(pa))

    # find boundaries by imposing the same conditions as
    # in the extract apertures for calculating the angle
    # i.e. relying on the invariance of two segments

    if 0 < pa < np.pi / 2.0 or np.pi < pa < 3 * np.pi / 2.0:

        # in the top right and bottom left areas
        # so adding to x goes with subtracting from y

        x_low = xcen + x_inc
        x_high = xcen - x_inc
        y_low = ycen - y_inc
        y_high = ycen + y_inc

    else:

        x_low = xcen - x_inc
        x_high = xcen + x_inc
        y_low = ycen - y_inc
        y_high = ycen + y_inc

    x_inc_best = 100 * np.abs(np.cos(best_pa))
    y_inc_best = 100 * np.abs(np.sin(best_pa))

    # find boundaries by imposing the same conditions as
    # in the extract apertures for calculating the angle
    # i.e. relying on the invariance of two segments

    if 0 < best_pa < np.pi / 2.0 or np.pi < best_pa < 3 * np.pi / 2.0:

        # in the top right and bottom left areas
        # so adding to x goes with subtracting from y

        x_low_best = xcen + x_inc_best
        x_high_best = xcen - x_inc_best
        y_low_best = ycen - y_inc_best
        y_high_best = ycen + y_inc_best

    else:

        x_low_best = xcen - x_inc_best
        x_high_best = xcen + x_inc_best
        y_low_best = ycen - y_inc_best
        y_high_best = ycen + y_inc_best

    x_inc_num = 100 * np.abs(np.cos(num_pa))
    y_inc_num = 100 * np.abs(np.sin(num_pa))

    if 0 < num_pa < np.pi / 2.0 or np.pi < num_pa < 3 * np.pi / 2.0:

        # in the top right and bottom left areas
        # so adding to x goes with subtracting from y

        x_low_num = xcen + x_inc_num
        x_high_num = xcen - x_inc_num
        y_low_num = ycen - y_inc_num
        y_high_num = ycen + y_inc_num

    else:

        x_low_num = xcen - x_inc_num
        x_high_num = xcen + x_inc_num
        y_low_num = ycen - y_inc_num
        y_high_num = ycen + y_inc_num

    # draw in the PA'S

    fig, ax = plt.subplots(4, 5, figsize=(24, 16))

    # flux plot
    ax[1][0].plot([y_h_low, y_h_high], [x_h_low, x_h_high],
               ls='--',
               color='aquamarine',
               label='hst\_pa')
    ax[1][0].plot([y_low, y_high], [x_low, x_high],
               ls='--',
               color='lightcoral',
               lw=2,
               label='dyn\_pa')
    ax[1][0].plot([y_low_best, y_high_best], [x_low_best, x_high_best],
               ls='--',
               color='darkorange',
               lw=2,
               label='rot\_pa')
    ax[1][0].plot([y_low_num, y_high_num], [x_low_num, x_high_num],
               ls='--',
               color='wheat',
               lw=2,
               label='num\_pa')
    l = ax[1][0].legend(loc='best',
                        frameon=False,
                        prop={'size':10})
    for text in l.get_texts():
        text.set_color("white")
    # velocity plot
    ax[1][1].plot([y_h_low, y_h_high], [x_h_low, x_h_high],
               ls='--',
               color='aquamarine',
               label='hst\_pa')
    ax[1][1].plot([y_low, y_high], [x_low, x_high],
               ls='--',
               color='lightcoral',
               lw=2,
               label='dyn\_pa')
    ax[1][1].plot([y_low_best, y_high_best], [x_low_best, x_high_best],
               ls='--',
               color='darkorange',
               lw=2,
               label='rot\_pa')
    l = ax[1][1].legend(loc='best',
                        frameon=False,
                        prop={'size':10})
    for text in l.get_texts():
        text.set_color("white")
    ax[1][2].plot([y_h_low, y_h_high], [x_h_low, x_h_high],
               ls='--',
               color='aquamarine',
               label='hst\_pa')
    ax[1][2].plot([y_low, y_high], [x_low, x_high],
               ls='--',
               color='lightcoral',
               lw=2,
               label='dyn\_pa')
    ax[1][2].plot([y_low_best, y_high_best], [x_low_best, x_high_best],
               ls='--',
               color='darkorange',
               lw=2,
               label='rot\_pa')
    l = ax[1][2].legend(loc='best',
                        frameon=False,
                        prop={'size':10})
    for text in l.get_texts():
        text.set_color("white")
    ax[1][3].plot([y_h_low, y_h_high], [x_h_low, x_h_high],
               ls='--',
               color='aquamarine',
               label='hst\_pa')
    ax[1][3].plot([y_low, y_high], [x_low, x_high],
               ls='--',
               color='lightcoral',
               lw=2,
               label='dyn\_pa')
    ax[1][3].plot([y_low_best, y_high_best], [x_low_best, x_high_best],
               ls='--',
               color='darkorange',
               lw=2,
               label='rot\_pa')
    l = ax[1][3].legend(loc='best',
                        frameon=False,
                        prop={'size':10})
    for text in l.get_texts():
        text.set_color("white")
    ax[1][4].plot([y_h_low, y_h_high], [x_h_low, x_h_high],
               ls='--',
               color='aquamarine',
               label='hst\_pa')
    ax[1][4].plot([y_low, y_high], [x_low, x_high],
               ls='--',
               color='lightcoral',
               lw=2,
               label='dyn\_pa')
    ax[1][4].plot([y_low_best, y_high_best], [x_low_best, x_high_best],
               ls='--',
               color='darkorange',
               lw=2,
               label='rot\_pa')
    l = ax[1][4].legend(loc='best',
                        frameon=False,
                        prop={'size':10})
    for text in l.get_texts():
        text.set_color("white")
    ax[2][4].plot([y_h_low, y_h_high], [x_h_low, x_h_high],
               ls='--',
               color='aquamarine',
               label='hst\_pa')
    ax[2][4].plot([y_low, y_high], [x_low, x_high],
               ls='--',
               color='lightcoral',
               lw=2,
               label='dyn\_pa')
    ax[2][4].plot([y_low_best, y_high_best], [x_low_best, x_high_best],
               ls='--',
               color='darkorange',
               lw=2,
               label='rot\_pa')
    l = ax[2][4].legend(loc='best',
                        frameon=False,
                        prop={'size':10})
    for text in l.get_texts():
        text.set_color("white")

    # mask background of velocity data to black

    # print data_hst.shape
    # print data_vel.shape

    m_data_flux = np.ma.array(data_flux,
                             mask=np.isnan(data_flux))
    m_data_hst = np.ma.array(data_hst,
                             mask=np.isnan(data_hst))
    m_data_vel = np.ma.array(data_vel,
                             mask=np.isnan(data_vel))
    m_data_vel_res = np.ma.array(vel_res_masked,
                             mask=np.isnan(vel_res_masked))
    m_data_mod_intrinsic = np.ma.array(mod_vel_masked,
                                       mask=np.isnan(mod_vel_masked))
    m_data_mod_blurred = np.ma.array(mod_vel_blurred_masked,
                                     mask=np.isnan(mod_vel_blurred_masked))
    m_data_sig = np.ma.array(sig_int_masked,
                             mask=np.isnan(sig_int_masked))

    cmap = plt.cm.bone_r
    cmap.set_bad('black', 1.)

    # HST

    im = ax[0][0].imshow(data_hst,
                         cmap=cmap,
                         vmax=10,
                         vmin=0)

    # HST - blurred

    blurred_hst = psf.blur_by_psf(data_hst,
                                  0.46,
                                  pix_scale,
                                  psf_factor)

    im = ax[3][3].imshow(blurred_hst,
                         cmap=cmap,
                         vmax=8,
                         vmin=0)

    ax[3][3].set_title('blurred HST')

    y_full_hst, x_full_hst = np.indices(galfit_mod.shape)

#        ax[0][0].contour(x_full_hst,
#                         y_full_hst,
#                         hst_fit,
#                         4,
#                         ls='solid',
#                         colors='b')

    apertures.plot(ax[0][0], color='green')
    disk_apertures.plot(ax[0][0], color='red')

    ax[0][0].text(4,7, 'z = %.2f' % redshift, color='black', fontsize=16)
    ax[0][0].text(4,14, 'pa = %.2f' % hst_pa, color='black', fontsize=16)
    ax[0][0].text(data_hst.shape[0] - 25,
                  data_hst.shape[1] - 6,
                  'F160_W', color='black', fontsize=16)

    ax[0][0].tick_params(axis='x',
                      labelbottom='off')

    ax[0][0].tick_params(axis='y',
                      labelleft='off')


    ax[0][0].set_title('HST imaging')

    # GALFIT MODEL

    ax[0][1].text(4, 7,
                  r'$R_{e} = %.2f$Kpc' % r_e,
                  fontsize=16,
                  color='black')

    ax[0][1].text(4, 14,
                  r'$\frac{b}{a} = %.2f$' % axis_r,
                  fontsize=16,
                  color='black')

    im = ax[0][1].imshow(galfit_mod,
                      cmap=cmap,
                      vmax=10,
                      vmin=0)

    apertures.plot(ax[0][1], color='green')
    disk_apertures.plot(ax[0][1], color='red')

    ax[0][1].tick_params(axis='x',
                      labelbottom='off')

    ax[0][1].tick_params(axis='y',
                      labelleft='off')

    ax[0][1].set_title('GALFIT mod')

    # GALFIT RESIDUALS

    im = ax[0][2].imshow(galfit_res,
                      cmap=cmap,
                      vmax=10,
                      vmin=0)

    ax[0][2].tick_params(axis='x',
                      labelbottom='off')

    ax[0][2].tick_params(axis='y',
                      labelleft='off')

    ax[0][2].set_title('GALFIT res')

    cmap = plt.cm.hot
    cmap.set_bad('black', 1.)

    # FIRST CONTINUUM

    ax[0][3].scatter(cont_peak_y,
                     cont_peak_x,
                     marker='x',
                     s=100,
                     color='blue')
    ax[0][3].scatter(fit_cont_y,
                     fit_cont_x,
                     marker='*',
                     s=100,
                     color='green')
    ax[0][3].scatter(oiii_peak_y,
                     oiii_peak_x,
                     marker='+',
                     s=100,
                     color='purple')

    im = ax[0][3].imshow(cont1,
                      cmap=cmap,
                      vmax=0.4,
                      vmin=0.0)

    ax[0][3].tick_params(axis='x',
                      labelbottom='off')

    ax[0][3].tick_params(axis='y',
                      labelleft='off')


    ax[0][3].set_title('Cont1')

    # FLATFIELDED CONTINUUM

    ax[0][4].scatter(cont_peak_y,
                     cont_peak_x,
                     marker='x',
                     s=100,
                     color='blue')

    ax[0][4].scatter(fit_cont_y,
                     fit_cont_x,
                     marker='*',
                     s=100,
                     color='green')

    ax[0][4].scatter(oiii_peak_y,
                     oiii_peak_x,
                     marker='+',
                     s=100,
                     color='purple')


    im = ax[0][4].imshow(b_cont2,
                      cmap=cmap,
                      vmax=0.1,
                      vmin=-0.4)

    y_full, x_full = np.indices(b_cont2.shape)
    ax[0][4].contour(x_full,
                     y_full,
                     fit_cont,
                     4,
                     ls='solid',
                     colors='black')

    ax[0][4].tick_params(axis='x',
                      labelbottom='off')

    ax[0][4].tick_params(axis='y',
                      labelleft='off')


    ax[0][4].set_title('Cont2')

    # OIII NARROWBAND
    print 'OIII PEAK: %s %s' % (oiii_peak_x, oiii_peak_y)
    ax[1][0].scatter(oiii_peak_y,
                     oiii_peak_x,
                     marker='+',
                     s=100,
                     color='purple')

    ax[1][0].scatter(cont_peak_y,
                     cont_peak_x,
                     marker='x',
                     s=100,
                     color='blue')
    ax[1][0].scatter(fit_cont_y,
                     fit_cont_x,
                     marker='*',
                     s=100,
                     color='green')

    im = ax[1][0].imshow(o_nband,
                      cmap=cmap,
                      vmax=3,
                      vmin=-0.3)

    ax[1][0].tick_params(axis='x',
                      labelbottom='off')

    ax[1][0].tick_params(axis='y',
                      labelleft='off')


    ax[1][0].set_title('OIII')

    cmap = plt.cm.jet
    cmap.set_bad('black', 1.)

    # OIII FLUX

    ax[1][1].scatter(oiii_peak_y,
                     oiii_peak_x,
                     marker='+',
                     s=100,
                     color='purple')

    ax[1][1].scatter(cont_peak_y,
                     cont_peak_x,
                     marker='x',
                     s=100,
                     color='blue')
    ax[1][1].scatter(fit_cont_y,
                     fit_cont_x,
                     marker='*',
                     s=100,
                     color='green')

    im = ax[1][1].imshow(m_data_flux,
                      interpolation='nearest',
                      cmap=cmap)


    ax[1][1].tick_params(axis='x',
                      labelbottom='off')

    ax[1][1].tick_params(axis='y',
                      labelleft='off')


    # set the title
    ax[1][1].set_title('[OIII] Flux')

    divider = make_axes_locatable(ax[1][1])
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)

    # OIII VELOCITY

    ax[1][2].scatter(oiii_peak_y,
                     oiii_peak_x,
                     marker='+',
                     s=100,
                     color='purple')

    ax[1][2].scatter(cont_peak_y,
                     cont_peak_x,
                     marker='x',
                     s=100,
                     color='blue')
    ax[1][2].scatter(fit_cont_y,
                     fit_cont_x,
                     marker='*',
                     s=100,
                     color='green')

    im = ax[1][2].imshow(m_data_vel,
                      vmin=vel_min,
                      vmax=vel_max,
                      interpolation='nearest',
                      cmap=cmap)


    ax[1][2].tick_params(axis='x',
                      labelbottom='off')

    ax[1][2].tick_params(axis='y',
                      labelleft='off')


    # set the title
    ax[1][2].set_title('Velocity from data')

    divider = make_axes_locatable(ax[1][2])
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)

    ax[1][3].scatter(oiii_peak_y,
                     oiii_peak_x,
                     marker='+',
                     s=100,
                     color='purple')

    ax[1][3].scatter(cont_peak_y,
                     cont_peak_x,
                     marker='x',
                     s=100,
                     color='blue')
    ax[1][3].scatter(fit_cont_y,
                     fit_cont_x,
                     marker='*',
                     s=100,
                     color='green')

    im = ax[1][3].imshow(m_data_mod_blurred,
                      vmin=vel_min,
                      vmax=vel_max,
                      interpolation='nearest',
                      cmap=cmap)

    ax[1][3].tick_params(axis='x',
                      labelbottom='off')

    ax[1][3].tick_params(axis='y',
                      labelleft='off')

    # set the title
    ax[1][3].set_title('Velocity from model')

    divider = make_axes_locatable(ax[1][3])
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)

    # OIII DIspersion

    ax[1][4].scatter(oiii_peak_y,
                     oiii_peak_x,
                     marker='+',
                     s=100,
                     color='purple')

    ax[1][4].scatter(cont_peak_y,
                     cont_peak_x,
                     marker='x',
                     s=100,
                     color='blue')
    ax[1][4].scatter(fit_cont_y,
                     fit_cont_x,
                     marker='*',
                     s=100,
                     color='green')

    im = ax[1][4].imshow(m_data_sig,
                      vmin=sig_min,
                      vmax=sig_max,
                      interpolation='nearest',
                      cmap=cmap)


    ax[1][4].tick_params(axis='x',
                      labelbottom='off')

    ax[1][4].tick_params(axis='y',
                      labelleft='off')


    # set the title
    ax[1][4].set_title('Velocity Dispersion Data')

    divider = make_axes_locatable(ax[1][4])
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)

    # OIII DIspersion

    ax[2][4].scatter(oiii_peak_y,
                     oiii_peak_x,
                     marker='+',
                     s=100,
                     color='purple')

    ax[2][4].scatter(cont_peak_y,
                     cont_peak_x,
                     marker='x',
                     s=100,
                     color='blue')
    ax[2][4].scatter(fit_cont_y,
                     fit_cont_x,
                     marker='*',
                     s=100,
                     color='green')

    im = ax[2][4].imshow(m_data_vel_res,
                      vmin=vel_min,
                      vmax=vel_max,
                      interpolation='nearest',
                      cmap=cmap)


    ax[2][4].tick_params(axis='x',
                      labelbottom='off')

    ax[2][4].tick_params(axis='y',
                      labelleft='off')


    # set the title
    ax[2][4].set_title('Data - Model Vel')

    divider = make_axes_locatable(ax[2][4])
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)

    # 1D VELOCITY PLOT

    # at this point evaluate as well some 1D models
    # fit the extracted data along the different position
    # angles in 1D - will take seconds and provides a
    # chi-squared and different fit for each of the pas

    dyn_pa_fit = arc_mod.model_fit(one_d_data_vel,
                                   one_d_model_x,
                                   1. / one_d_data_vel_errors,
                                   va,
                                   rt)

    print 'DYN CHI: %s' % dyn_pa_fit.chisqr

    best_pa_fit = arc_mod.model_fit(best_pa_vel,
                                    best_pa_x,
                                    1. / best_pa_error,
                                    va,
                                    rt)

    print 'ROT CHI: %s' % best_pa_fit.chisqr

    hst_pa_fit = arc_mod.model_fit(hst_pa_vel,
                                   hst_pa_x,
                                   1. / hst_pa_error,
                                   va,
                                   rt)

    print 'HST CHI: %s' % hst_pa_fit.chisqr

    ax[2][0].set_ylabel(r'V$_{c}$[kms$^{-1}$]',
                      fontsize=10,
                      fontweight='bold')
    ax[2][0].set_xlabel(r'r [arcsec]',
                      fontsize=10,
                      fontweight='bold')
    # tick parameters 
    ax[2][0].tick_params(axis='both',
                       which='major',
                       labelsize=8,
                       length=6,
                       width=2)
    ax[2][0].tick_params(axis='both',
                       which='minor',
                       labelsize=8,
                       length=3,
                       width=1)

    ax[2][0].plot(one_d_model_x,
                  one_d_mod_vel_intrinsic,
                  color='red',
                  label='int\_model',
                  lw=2)

#    ax[2][0].scatter(one_d_model_x,
#                    one_d_mod_vel_intrinsic,
#                    marker='o',
#                    color='red')

    ax[2][0].plot(one_d_model_x,
                  one_d_mod_vel_blurred,
                  color='blue',
                  label='blurred\_model',
                  lw=2)

#    ax[2][0].scatter(one_d_model_x,
#                    one_d_mod_vel_blurred,
#                    marker='o',
#                   color='blue')

    ax[2][0].plot(one_d_model_x,
                  one_d_vel_res,
                  color='purple',
                  label='residuals',
                  lw=2)

    ax[2][0].scatter(one_d_model_x,
                     one_d_vel_res,
                     marker='^',
                     color='purple',
                     s=25)

    (_, caps, _) =  ax[2][0].errorbar(one_d_model_x,
                                      one_d_data_vel,
                                      yerr=one_d_data_vel_errors,
                                      fmt='+',
                                      color='black',
                                      label='data',
                                      capsize=3,
                                      elinewidth=2)

    for cap in caps:
        cap.set_markeredgewidth(2)

    # new addition showing the error region

    ax[2][0].fill_between(one_d_model_x,
                          one_d_mod_vel_intrinsic_16,
                          one_d_mod_vel_intrinsic_84,
                          facecolor='indianred',
                          alpha=0.5)
    ax[2][0].fill_between(one_d_model_x,
                          one_d_mod_vel_blurred_16,
                          one_d_mod_vel_blurred_84,
                          facecolor='cornflowerblue',
                          alpha=0.5)

    ax[2][0].set_xlim(-1.5, 1.5)

    # ax[2][0].legend(prop={'size':5}, loc=1)

    ax[2][0].set_title('Model and observed Velocity')

    # ax[2][0].set_ylabel('velocity (kms$^{-1}$)')

    ax[2][0].set_xlabel('arcsec')

    ax[2][0].axhline(0, color='silver', ls='-.')
    ax[2][0].axvline(0, color='silver', ls='-.')
    ax[2][0].axhline(va, color='silver', ls='--')
    ax[2][0].axhline(-1.*va, color='silver', ls='--')

    # Also add in vertical lines for where the kinematics 
    # should be extracted

    ax[2][0].plot([r_e_arc_conv, r_e_arc_conv], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[2][0].plot([r_d_arc_conv, r_d_arc_conv], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[2][0].plot([r_d_arc_conv_22, r_d_arc_conv_22], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[2][0].plot([r_d_arc_conv_30, r_d_arc_conv_30], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[2][0].plot([-1*r_e_arc_conv, -1*r_e_arc_conv], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[2][0].plot([-r_d_arc_conv, -r_d_arc_conv], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[2][0].plot([-r_d_arc_conv_22, -r_d_arc_conv_22], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[2][0].plot([-r_d_arc_conv_30, -r_d_arc_conv_30], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)


    ax[2][0].minorticks_on()
    ax[2][0].set_xlabel('arcsec')
    leg = ax[2][0].legend(loc='upper left',fancybox=True, prop={'size':8})
    leg.get_frame().set_alpha(0.5)

    ax[2][1].set_ylabel(r'$\sigma$[kms$^{-1}$]',
                      fontsize=10,
                      fontweight='bold')
    ax[2][1].set_xlabel(r'r [arcsec]',
                      fontsize=10,
                      fontweight='bold')
    # tick parameters 
    ax[2][1].tick_params(axis='both',
                       which='major',
                       labelsize=8,
                       length=6,
                       width=2)
    ax[2][1].tick_params(axis='both',
                       which='minor',
                       labelsize=8,
                       length=3,
                       width=1)

    ax[2][1].minorticks_on()
    # 1D DISPERSION PLOT

    ax[2][1].errorbar(one_d_model_x,
                   one_d_data_sig,
                   yerr=one_d_data_sig_errors,
                   fmt='o',
                   color='red',
                   label='obs\_sig')

    ax[2][1].errorbar(one_d_model_x,
                   one_d_sig_int,
                   yerr=one_d_data_sig_errors,
                   fmt='o',
                   color='blue',
                   label='int\_sig')

    ax[2][1].plot(one_d_model_x,
                  one_d_sig_model_full,
                  color='orange',
                  label='sig\_model')

    ax[2][1].scatter(one_d_model_x,
                    one_d_sig_model_full,
                    marker='o',
                  color='orange')

    ax[2][1].plot(one_d_model_x,
                  one_d_sig_res,
                  color='purple')

    ax[2][1].scatter(one_d_model_x,
                    one_d_sig_res,
                    marker='o',
                  color='purple',
                  label='sig\_residuals')   

    ax[2][1].axvline(0, color='silver', ls='-.')
    ax[2][1].axvline(r_e_arc_conv, color='maroon', ls='--', lw=2)
    ax[2][1].axvline(-1*r_e_arc_conv, color='maroon', ls='--', lw=2)
    ax[2][1].axvline(r_d_arc_conv, color='maroon', ls='--', lw=2)
    ax[2][1].axvline(-r_d_arc_conv, color='maroon', ls='--', lw=2)
    ax[2][1].axvline(r_d_arc_conv_22, color='maroon', ls='--', lw=2)
    ax[2][1].axvline(-r_d_arc_conv_22, color='maroon', ls='--', lw=2)
    ax[2][1].axvline(r_d_arc_conv_30, color='maroon', ls='--', lw=2)
    ax[2][1].axvline(-r_d_arc_conv_30, color='maroon', ls='--', lw=2)
    ax[2][1].set_xlim(-1.5, 1.5)

    ax[2][1].set_title('Velocity Dispersion')

    # ax[2][1].set_ylabel('velocity (kms$^{-1}$)')

    ax[2][1].set_xlabel('arcsec')
    leg = ax[2][1].legend(loc='upper left',fancybox=True, prop={'size':8})
    leg.get_frame().set_alpha(0.5)


    # also want to fit a gaussian to the integrated spectrum to
    # determine emission line width. Surely the integrated sigma
    # is not a good measure of the turbulence as this will be higher
    # with higher velocity gradient?

    g_out, g_best, g_covar = one_d_g.ped_gauss_fit(obj_cube.wave_array[o_peak-50:o_peak+50],
                                            one_d_spectrum[o_peak-50:o_peak+50])

    gauss_spectrum = g_out.eval(x=obj_cube.wave_array[o_peak-50:o_peak+50])

    sigma_int = g_best['sigma']

    # also measure an error weighted sigma

    indices = ~np.isnan(data_sig)

    sigma_o = np.median(data_sig[indices])
                         

    indices = ~np.isnan(sig_int)

    sigma_o_i = np.median(sig_int[indices])
                           

    indices = ~np.isnan(sig_int_84)

    sigma_o_i_84 = np.median(sig_int_84[indices])
                           

    indices = ~np.isnan(sig_int_16)

    sigma_o_i_16 = np.median(sig_int_16[indices])
                           

    indices = ~np.isnan(sig_int_16_80)

    sigma_o_i_16_80 = np.median(sig_int_16_80[indices])
                           

    indices = ~np.isnan(sig_int_84_40)

    sigma_o_i_84_40 = np.median(sig_int_84_40[indices])
                           

    c = 2.99792458E5


    ax[2][2].plot(obj_cube.wave_array[o_peak-50:o_peak+50],
                  one_d_spectrum[o_peak-50:o_peak+50],
                  color='black')

    ax[2][2].plot(obj_cube.wave_array[o_peak-50:o_peak+50],
                  gauss_spectrum,
                  color='red')

    ax[2][2].axvline(central_l, color='red', ls='--')
    ax[2][2].axvline(obj_cube.wave_array[o_peak-5], color='red', ls='--')
    ax[2][2].axvline(obj_cube.wave_array[o_peak+5], color='red', ls='--')

    ax[2][2].set_title('Integrated Spectrum')

    ax[2][3].plot(pa_array,
                  stat_array,
                  color='black')

    ax[2][3].axvline(best_pa, color='darkorange', ls='--')

    if pa > np.pi:

        ax[2][3].axvline(pa - np.pi, color='lightcoral', ls='--')

    else:

        ax[2][3].axvline(pa, color='lightcoral', ls='--')

    if hst_pa > np.pi:

        ax[2][3].axvline(hst_pa - np.pi, color='aquamarine', ls='--')

    else:

        ax[2][3].axvline(hst_pa, color='aquamarine', ls='--')

    ax[2][3].set_title('PA Rotation')

    # plot the numerical fitting stuff
    # want to include on here in text what the
    # axis ratio and the PA are

    im = ax[3][0].imshow(num_cut_data,
                         vmax=5,
                         vmin=0)

    ax[3][0].text(1,2,
                  r'$\frac{b}{a} = %.2f$' % num_axis_ratio,
                  color='white',
                  fontsize=16)

    ax[3][0].text(1,6,
                  r'$pa = %.2f$' % num_pa,
                  color='white',
                  fontsize=16)

    y_full, x_full = np.indices(num_cut_data.shape)
    ax[3][0].contour(x_full,
                     y_full,
                     num_fit_data,
                     4,
                     ls='solid',
                     colors='black')

    # now plot the curve of growth parameters

    ax[3][1].plot(scaled_axis_array,
                  num_sum_array,
                  color='blue')
    ax[3][1].axvline(scaled_num_r_e, color='black',ls='--')
    ax[3][1].axvline(scaled_num_r_9, color='black',ls='--')
    ax[3][1].text(10, 50,
                  r'$R_{e} = %.2f$Kpc' % scaled_num_r_e,
                  color='black',
                  fontsize=16)
    ax[3][1].text(10, 500,
                  r'$R_{9} = %.2f$Kpc' % scaled_num_r_9,
                  color='black',
                  fontsize=16)

    ax[3][2].plot(one_d_model_x,
                  dyn_pa_fit.eval(r=one_d_model_x),
                  color='blue')

    ax[3][2].errorbar(one_d_model_x,
                   one_d_data_vel,
                   yerr=one_d_data_vel_errors,
                   fmt='o',
                   color='blue',
                   label='dyn\_pa')

    ax[3][2].plot(best_pa_x,
                  best_pa_fit.eval(r=best_pa_x),
                  color='darkorange')

    ax[3][2].errorbar(best_pa_x,
                      best_pa_vel,
                      yerr=best_pa_error,
                      fmt='o',
                      color='darkorange',
                      label='rot\_pa')

    ax[3][2].plot(hst_pa_x,
                  hst_pa_fit.eval(r=hst_pa_x),
                  color='aquamarine')

    ax[3][2].errorbar(hst_pa_x,
                      hst_pa_vel,
                      yerr=hst_pa_error,
                      fmt='o',
                      color='aquamarine',
                      label='hst\_pa')

    ax[3][2].set_title('Model and Real Velocity')

    # ax[3][2].set_ylabel('velocity (kms$^{-1}$)')

    ax[3][2].set_xlabel('arcsec')

    ax[3][2].axhline(0, color='silver', ls='-.')
    ax[3][2].axvline(0, color='silver', ls='-.')
    ax[3][2].axhline(va, color='silver', ls='--')
    ax[3][2].axhline(-1.*va, color='silver', ls='--')

    # Also add in vertical lines for where the kinematics 
    # should be extracted

    ax[3][2].plot([r_e_arc_conv, r_e_arc_conv], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].plot([r_d_arc_conv, r_d_arc_conv], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].plot([r_d_arc_conv_22, r_d_arc_conv_22], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].plot([r_d_arc_conv_30, r_d_arc_conv_30], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].plot([-1*r_e_arc_conv, -1*r_e_arc_conv], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].plot([-r_d_arc_conv, -r_d_arc_conv], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].plot([-r_d_arc_conv_22, -r_d_arc_conv_22], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].plot([-r_d_arc_conv_30, -r_d_arc_conv_30], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].set_xlim(-1.5, 1.5)
    leg = ax[3][2].legend(loc='upper left',fancybox=True, prop={'size':8})
    leg.get_frame().set_alpha(0.5)

    #plt.suptitle('%s' % gal_name)

    # and the 1D plot showing the aperture growth

    fig.tight_layout()

    #plt.show()

    fig.savefig('%s_grid_mcmc_params.png' % infile[:-5])
    plt.close('all')

    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('font', weight='bold')
    rc('text', usetex=True)
    rc('axes', linewidth=2)
    plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

    # Now create the grids for the paper
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(14, 5))

    cmap = plt.cm.Greys_r
    cmap.set_bad('black', 1.)

    def rof(number):
        """Round a number to the closest half integer.
        >>> round_of_rating(1.3)
        1.5
        >>> round_of_rating(2.6)
        2.5
        >>> round_of_rating(3.0)
        3.0
        >>> round_of_rating(4.1)
        4.0"""

        return round(number * 2) / 2

    # we want to centre nicely the HST image and the velocity and flux fields
    # by using indices that will give a 1 arcsecond box centred on the galaxy

    # HST TYPE DATA INDICES
    x_upper_hst = np.round(hst_x_cen + 15.0)
    if x_upper_hst >= data_hst.shape[0]:
        x_upper_hst = data_hst.shape[0] - 1
    x_lower_hst = np.round(hst_x_cen - 15.0)
    if x_lower_hst < 0:
        x_lower_hst = 0
    y_upper_hst = np.round(hst_y_cen + 15.0)
    if y_upper_hst >= data_hst.shape[1]:
        y_upper_hst = data_hst.shape[1] - 1
    y_lower_hst = np.round(hst_y_cen - 15.0)
    if y_lower_hst < 0:
        y_lower_hst = 0

    # KMOS TYPE DATA INDICES
    x_upper_kmos = np.round(xcen + 12.0)
    if x_upper_kmos >= m_data_flux.shape[0]:
        x_upper_kmos = m_data_flux.shape[0] - 1
    x_lower_kmos = np.round(xcen - 12.0)
    if x_lower_kmos < 0:
        x_lower_kmos = 0
    y_upper_kmos = np.round(ycen + 12.0)
    if y_upper_kmos >= m_data_flux.shape[1]:
        y_upper_kmos = m_data_flux.shape[1] - 1
    y_lower_kmos = np.round(ycen - 12.0)
    if y_lower_kmos < 0:
        y_lower_kmos = 0

    print x_upper_hst, x_lower_hst, y_upper_hst, y_lower_hst
    print x_upper_kmos, x_lower_kmos, y_upper_kmos, y_lower_kmos

    # defining the plotting data

    hst_plot_data = data_hst[x_lower_hst:x_upper_hst, y_lower_hst:y_upper_hst][::-1]
    galfit_plot_data = galfit_mod[x_lower_hst:x_upper_hst, y_lower_hst:y_upper_hst][::-1]
    galfit_res_plot_data = galfit_res[x_lower_hst:x_upper_hst, y_lower_hst:y_upper_hst][::-1]

    nband_plot_data = o_nband[x_lower_kmos:x_upper_kmos, y_lower_kmos:y_upper_kmos][::-1]
    flux_plot_data = m_data_flux[x_lower_kmos:x_upper_kmos, y_lower_kmos:y_upper_kmos][::-1]
    vel_plot_data = m_data_vel[x_lower_kmos:x_upper_kmos, y_lower_kmos:y_upper_kmos][::-1]
    mod_vel_plot_data = m_data_mod_blurred[x_lower_kmos:x_upper_kmos, y_lower_kmos:y_upper_kmos][::-1]
    res_plot_data = m_data_vel_res[x_lower_kmos:x_upper_kmos, y_lower_kmos:y_upper_kmos][::-1]
    sig_plot_data = m_data_sig[x_lower_kmos:x_upper_kmos, y_lower_kmos:y_upper_kmos][::-1]

    # define the new hst centres (different formula due to where the counting begins)
    new_hst_xcen = (hst_plot_data.shape[0] - (hst_x_cen - x_lower_hst))
    new_hst_ycen = hst_y_cen - y_lower_hst - 1

    # define the new kmos centres
    new_xcen = (flux_plot_data.shape[0] - 1 - (xcen - x_lower_kmos))
    new_ycen = ycen - y_lower_kmos

    print 'SHAPES KMOS'
    print m_data_flux.shape, flux_plot_data.shape, x_upper_kmos, x_lower_kmos, y_upper_kmos, y_lower_kmos
    print 'SHAPES HST'
    print data_hst.shape, hst_plot_data.shape, x_upper_hst, x_lower_hst, y_upper_hst, y_lower_hst
    print 'CENTERS'
    print xcen, ycen, new_xcen, new_ycen, hst_x_cen, hst_y_cen, new_hst_xcen, new_hst_ycen

    # defining the subplot axes limits

    hst_x_7 = hst_plot_data.shape[0] - 1
    hst_x_6 = 5 * (hst_plot_data.shape[0] - 1) / 6.
    hst_x_5 = 4 * (hst_plot_data.shape[0] - 1) / 6.
    hst_x_4 = 3 * (hst_plot_data.shape[0] - 1) / 6.
    hst_x_3 = 2 * (hst_plot_data.shape[0] - 1) / 6.
    hst_x_2 = 1 * (hst_plot_data.shape[0] - 1) / 6.
    hst_x_1 = 0

    hst_y_7 = hst_plot_data.shape[1] - 1
    hst_y_6 = 5 * (hst_plot_data.shape[1] - 1) / 6.
    hst_y_5 = 4 * (hst_plot_data.shape[1] - 1) / 6.
    hst_y_4 = 3 * (hst_plot_data.shape[1] - 1) / 6.
    hst_y_3 = 2 * (hst_plot_data.shape[1] - 1) / 6.
    hst_y_2 = 1 * (hst_plot_data.shape[1] - 1) / 6.
    hst_y_1 = 0


    kmos_x_7 = flux_plot_data.shape[0] - 1
    kmos_x_6 = 5 * (flux_plot_data.shape[0] - 1) / 6.
    kmos_x_5 = 4 * (flux_plot_data.shape[0] - 1) / 6.
    kmos_x_4 = 3 * (flux_plot_data.shape[0] - 1) / 6.
    kmos_x_3 = 2 * (flux_plot_data.shape[0] - 1) / 6.
    kmos_x_2 = 1 * (flux_plot_data.shape[0] - 1) / 6.
    kmos_x_1 = 0

    kmos_y_7 = flux_plot_data.shape[1] - 1
    kmos_y_6 = 5 * (flux_plot_data.shape[1] - 1) / 6.
    kmos_y_5 = 4 * (flux_plot_data.shape[1] - 1) / 6.
    kmos_y_4 = 3 * (flux_plot_data.shape[1] - 1) / 6.
    kmos_y_3 = 2 * (flux_plot_data.shape[1] - 1) / 6.
    kmos_y_2 = 1 * (flux_plot_data.shape[1] - 1) / 6.
    kmos_y_1 = 0

    plt.sca(axes[0, 0])
    plt.xticks([hst_y_1,  hst_y_3, hst_y_4, hst_y_5,  hst_y_7], ['-1.5','-0.5', '0', '0.5', '1.5'], color='black')
    plt.yticks([hst_x_7,  hst_x_5, hst_x_4, hst_x_3,  hst_x_1], ['-1.5','-0.5', '0', '0.5', '1.5'], color='black')

    plt.sca(axes[0, 1])
    plt.xticks([hst_y_1,  hst_y_3, hst_y_4, hst_y_5,  hst_y_7], ['-1.5','-0.5', '0', '0.5', '1.5'], color='black')
    plt.yticks([hst_x_7,  hst_x_5, hst_x_4, hst_x_3,  hst_x_1], ['-1.5','-0.5', '0', '0.5', '1.5'], color='black')

    plt.sca(axes[0, 2])
    plt.xticks([hst_y_1,  hst_y_3, hst_y_4, hst_y_5,  hst_y_7], ['-1.5','-0.5', '0', '0.5', '1.5'], color='black')
    plt.yticks([hst_x_7,  hst_x_5, hst_x_4, hst_x_3,  hst_x_1], ['-1.5','-0.5', '0', '0.5', '1.5'], color='black')

    plt.sca(axes[0, 3])
    plt.xticks([kmos_y_1,  kmos_y_3, kmos_y_4, kmos_y_5,  kmos_y_7], ['-1.5','-0.5', '0', '0.5', '1.5'], color='black')
    plt.yticks([kmos_x_7,  kmos_x_5, kmos_x_4, kmos_x_3,     kmos_x_1], ['-1.5','-0.5', '0', '0.5', '1.5'], color='black')

    plt.sca(axes[0, 4])
    plt.xticks([kmos_y_1,  kmos_y_3, kmos_y_4, kmos_y_5,  kmos_y_7], ['-1.5','-0.5', '0', '0.5', '1.5'], color='black')
    plt.yticks([kmos_x_7,  kmos_x_5, kmos_x_4, kmos_x_3,     kmos_x_1], ['-1.5','-0.5', '0', '0.5', '1.5'], color='black')

    plt.sca(axes[1, 0])
    plt.xticks([kmos_y_1,  kmos_y_3, kmos_y_4, kmos_y_5,  kmos_y_7], ['-1.5','-0.5', '0', '0.5', '1.5'], color='black')
    plt.yticks([kmos_x_7,  kmos_x_5, kmos_x_4, kmos_x_3,     kmos_x_1], ['-1.5','-0.5', '0', '0.5', '1.5'], color='black')

    plt.sca(axes[1, 1])
    plt.xticks([kmos_y_1,  kmos_y_3, kmos_y_4, kmos_y_5,  kmos_y_7], ['-1.5','-0.5', '0', '0.5', '1.5'], color='black')
    plt.yticks([kmos_x_7,  kmos_x_5, kmos_x_4, kmos_x_3,     kmos_x_1], ['-1.5','-0.5', '0', '0.5', '1.5'], color='black')

    plt.sca(axes[1, 2])
    plt.xticks([kmos_y_1,  kmos_y_3, kmos_y_4, kmos_y_5,  kmos_y_7], ['-1.5','-0.5', '0', '0.5', '1.5'], color='black')
    plt.yticks([kmos_x_7,  kmos_x_5, kmos_x_4, kmos_x_3,     kmos_x_1], ['-1.5','-0.5', '0', '0.5', '1.5'], color='black')

    # also here want to plot the position angles onto the appropriate plots
    # first check that pas are less than pi
    hst_copy_pa = copy(hst_pa)
    dyn_copy_pa = copy(pa)

    if hst_copy_pa > np.pi:
        hst_copy_pa = hst_copy_pa - np.pi
    if dyn_copy_pa > np.pi:
        dyn_copy_pa = dyn_copy_pa - np.pi

    # now need to apply correction to account for the fact we are
    # flipping in the y direction

    hst_with_hst_limits = pa_calc.pa_limits(hst_copy_pa, new_hst_xcen, new_hst_ycen)
    hst_with_hst_x_low = hst_with_hst_limits[0][0]
    hst_with_hst_x_high = hst_with_hst_limits[0][1]

    hst_with_hst_y_low = hst_with_hst_limits[1][0]
    hst_with_hst_y_high = hst_with_hst_limits[1][1]

    hst_with_kmos_limits = pa_calc.pa_limits(hst_copy_pa, new_xcen, new_ycen)
    hst_with_kmos_x_low = hst_with_kmos_limits[0][0]
    hst_with_kmos_x_high = hst_with_kmos_limits[0][1]

    hst_with_kmos_y_low = hst_with_kmos_limits[1][0]
    hst_with_kmos_y_high = hst_with_kmos_limits[1][1]

    kmos_with_kmos_limits = pa_calc.pa_limits(dyn_copy_pa, new_xcen, new_ycen)
    kmos_with_kmos_x_low = kmos_with_kmos_limits[0][0]
    kmos_with_kmos_x_high = kmos_with_kmos_limits[0][1]

    kmos_with_kmos_y_low = kmos_with_kmos_limits[1][0]
    kmos_with_kmos_y_high = kmos_with_kmos_limits[1][1]

    # and plot the PAs onto the diagrams

    axes[0][0].plot([hst_with_hst_y_low, hst_with_hst_y_high], [hst_with_hst_x_high, hst_with_hst_x_low],
                    ls='--',
                    color='orange',
                    lw=3)
    axes[0][1].plot([hst_with_hst_y_low, hst_with_hst_y_high], [hst_with_hst_x_high, hst_with_hst_x_low],
                    ls='--',
                    color='orange',
                    lw=3)
    axes[1][0].plot([hst_with_kmos_y_low, hst_with_kmos_y_high], [hst_with_kmos_x_high, hst_with_kmos_x_low],
                    ls='--',
                    color='orange',
                    lw=3)
    axes[1][1].plot([hst_with_kmos_y_low, hst_with_kmos_y_high], [hst_with_kmos_x_high, hst_with_kmos_x_low],
                    ls='--',
                    color='orange',
                    lw=3)
    axes[1][2].plot([hst_with_kmos_y_low, hst_with_kmos_y_high], [hst_with_kmos_x_high, hst_with_kmos_x_low],
                    ls='--',
                    color='orange',
                    lw=3)
    axes[0][4].plot([hst_with_kmos_y_low, hst_with_kmos_y_high], [hst_with_kmos_x_high, hst_with_kmos_x_low],
                    ls='--',
                    color='orange',
                    lw=3)
    axes[1][0].plot([kmos_with_kmos_y_low, kmos_with_kmos_y_high], [kmos_with_kmos_x_high, kmos_with_kmos_x_low],
                    ls='-',
                    color='black',
                    lw=3)
    axes[1][1].plot([kmos_with_kmos_y_low, kmos_with_kmos_y_high], [kmos_with_kmos_x_high, kmos_with_kmos_x_low],
                    ls='-',
                    color='black',
                    lw=3)
    axes[1][2].plot([kmos_with_kmos_y_low, kmos_with_kmos_y_high], [kmos_with_kmos_x_high, kmos_with_kmos_x_low],
                    ls='-',
                    color='black',
                    lw=3)
    axes[0][4].plot([kmos_with_kmos_y_low, kmos_with_kmos_y_high], [kmos_with_kmos_x_high, kmos_with_kmos_x_low],
                ls='-',
                color='black',
                lw=3)


    # HST PLOTTING

    # Galaxy name

    gal_name_short = gal_name[26:-5]

    hst_im = axes[0][0].imshow(hst_plot_data,
                               cmap=cmap,
                               vmax=8,
                               vmin=0)


    axes[0][0].text(2,hst_plot_data.shape[0] - 3,
                    r'\textbf{z = %.2f}' % redshift,
                    color='lightgray',
                    fontsize=16)

    gal_name_short = gal_name_short.replace('_','\_')

    axes[0][0].text(2,5,
                    r'\textbf{%s}' % gal_name_short,
                    color='lightgray',
                    fontsize=16)

    axes[0][0].set(adjustable='box-forced', aspect='equal')

    # tick parameters 
    axes[0][0].tick_params(axis='both',
                           which='major',
                           labelsize=10,
                           length=4,
                           width=4,
                           color='lightgray')
    [i.set_color('lightgray') for i in axes[0][0].spines.itervalues()]
    [i.set_linewidth(4.0) for i in axes[0][0].spines.itervalues()]
    axes[0][0].set_ylabel(r'\textbf{arcsec}',
                           fontsize=14,
                           fontweight='bold')



    # GALFIT

    galfit_im = axes[0][1].imshow(galfit_plot_data,
                               cmap=cmap,
                               vmax=8,
                               vmin=0)

    axes[0][1].set(adjustable='box-forced', aspect='equal')
    axes[0][1].text(2,hst_plot_data.shape[0] - 3,
                    r'\textbf{Galfit Model}',
                    color='lightgray',
                    fontsize=16)

    # tick parameters 
    axes[0][1].tick_params(axis='both',
                           which='major',
                           labelsize=10,
                           length=4,
                           width=4,
                           color='lightgray')
    [i.set_color('lightgray') for i in axes[0][1].spines.itervalues()]
    [i.set_linewidth(4.0) for i in axes[0][1].spines.itervalues()]


    # GALFIT RESIDUALS

    galfit_res_im = axes[0][2].imshow(galfit_res_plot_data,
                               cmap=cmap,
                               vmax=8,
                               vmin=0)

    axes[0][2].set(adjustable='box-forced', aspect='equal')
    axes[0][2].text(2,hst_plot_data.shape[0] - 3,
                    r'\textbf{Galfit Residual}',
                    color='lightgray',
                    fontsize=16)
    # tick parameters 
    axes[0][2].tick_params(axis='both',
                           which='major',
                           labelsize=10,
                           length=4,
                           width=4,
                           color='lightgray')
    [i.set_color('lightgray') for i in axes[0][2].spines.itervalues()]
    [i.set_linewidth(4.0) for i in axes[0][2].spines.itervalues()]

    # OIII FLUX

    cmap = plt.cm.jet
    cmap.set_bad('white', 1.)

    flux_im = axes[0][3].imshow(flux_plot_data,
                                interpolation='nearest',
                                cmap=cmap)

    axes[0][3].set(adjustable='box-forced', aspect='equal')
    axes[0][3].text(1,2, r'\textbf{[O~{\sc III}] Intensity}', color='black', fontsize=16)
    axes[0][3].tick_params(axis='both',
                           which='major',
                           labelsize=10,
                           length=4,
                           width=4)
    [i.set_linewidth(4.0) for i in axes[0][3].spines.itervalues()]

    axes[0][3].scatter(new_ycen,
                       new_xcen,
                       marker='+',
                       s=100,
                       color='black',
                       linewidths=3)

    # Plotting the beam size on here also
    from photutils import CircularAperture
    positions = [(flux_plot_data.shape[1] - 3), (flux_plot_data.shape[0] - 3)]
    apertures = CircularAperture(positions, r=2)

    apertures.plot(axes[0][3],lw=2)


    # OIII VEL

    vel_limits = np.floor((abs(vel_min) + vel_max) / 2.0)

    axes[1][0].imshow(vel_plot_data,
                      vmin=-vel_limits,
                      vmax=vel_limits,
                      cmap=cmap)

    vel_im = axes[1][0].imshow(vel_plot_data,
                      vmin=-vel_limits,
                      vmax=vel_limits,
                      interpolation='nearest',
                      cmap=cmap)

    vel_data_cbaxes = fig.add_axes([0.0675, 0.11, 0.1, 0.02]) 
    
    cb = plt.colorbar(vel_im,
                      cax = vel_data_cbaxes,
                      orientation='horizontal',
                      ticks=[-vel_limits, vel_limits])
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    for l in cb.ax.xaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(16)

    axes[1][0].set(adjustable='box-forced', aspect='equal')

    axes[1][0].set_ylabel(r'\textbf{arcsec}',
                      fontsize=14,
                      fontweight='bold')

    axes[1][0].text(4,2, r'\textbf{Vel. (km s$\boldsymbol{^{-1}}$)}', color='black', fontsize=16)
    axes[1][0].tick_params(axis='both',
                           which='major',
                           labelsize=10,
                           length=4,
                           width=4)
    [i.set_linewidth(4.0) for i in axes[1][0].spines.itervalues()]

    # scatter central positions
    axes[1][0].scatter(new_ycen,
                       new_xcen,
                       marker='+',
                       s=120,
                       color='black',
                       linewidths=3)

    # OIII VEL MOD

    axes[1][1].imshow(mod_vel_plot_data,
                      vmin=-vel_limits,
                      vmax=vel_limits,
                      cmap=cmap)

    vel_mod_im = axes[1][1].imshow(mod_vel_plot_data,
                      vmin=-vel_limits,
                      vmax=vel_limits,
                                   interpolation='nearest',
                                   cmap=cmap)

    vel_mod_cbaxes = fig.add_axes([0.2675, 0.11, 0.1, 0.02]) 
    
    cb = plt.colorbar(vel_mod_im,
                      cax = vel_mod_cbaxes,
                      orientation='horizontal',
                      ticks=[-vel_limits, vel_limits])
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    for l in cb.ax.xaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(16)


    axes[1][1].set(adjustable='box-forced', aspect='equal')

    axes[1][1].scatter(new_ycen,
                       new_xcen,
                       marker='+',
                       s=100,
                       color='black',
                       linewidths=3)
    axes[1][1].text(0,2, r'\textbf{Model Vel. (km s$\boldsymbol{^{-1}}$)}',
                    color='black',
                    fontsize=13)
    axes[1][1].tick_params(axis='both',
                           which='major',
                           labelsize=10,
                           length=4,
                           width=4)
    [i.set_linewidth(4.0) for i in axes[1][1].spines.itervalues()]

    # OIII VEL MOD RESIDUAL

    axes[1][2].imshow(res_plot_data,
                      vmin=-vel_limits,
                      vmax=vel_limits,
                      cmap=cmap)

    vel_res_im = axes[1][2].imshow(res_plot_data,
                      vmin=-vel_limits,
                      vmax=vel_limits,
                                   interpolation='nearest',
                                   cmap=cmap)

    vel_res_cbaxes = fig.add_axes([0.4675, 0.11, 0.1, 0.02]) 
    
    cb = plt.colorbar(vel_res_im,
                      cax = vel_res_cbaxes,
                      orientation='horizontal',
                      ticks=[-vel_limits, vel_limits])
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    for l in cb.ax.xaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(16)


    axes[1][2].set(adjustable='box-forced', aspect='equal')

    axes[1][2].scatter(new_ycen,
                       new_xcen,
                       marker='+',
                       s=100,
                       color='black',
                       linewidths=3)
    axes[1][2].text(0,2, r'\textbf{Model Res. (km s$\boldsymbol{^{-1}}$)}',
                    color='black',
                    fontsize=13)
    axes[1][2].tick_params(axis='both',
                           which='major',
                           labelsize=10,
                           length=4,
                           width=4)
    [i.set_linewidth(4.0) for i in axes[1][2].spines.itervalues()]


    # OIII SIGMA

    cmap = plt.cm.gnuplot
    cmap.set_bad('white', 1.)

    axes[0][4].imshow(sig_plot_data,
                      vmin=sig_min,
                      vmax=sig_max,
                      cmap=cmap)

    sig_im = axes[0][4].imshow(sig_plot_data,
                      vmin=sig_min,
                      vmax=sig_max,
                      interpolation='nearest',
                      cmap=cmap)

    sig_cbaxes = fig.add_axes([0.8675, 0.60, 0.1, 0.02]) 
    
    cb = plt.colorbar(sig_im,
                      cax = sig_cbaxes,
                      orientation='horizontal',
                      ticks=[np.ceil(sig_min), 0, np.floor(sig_max)])
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    for l in cb.ax.xaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(16)

    axes[0][4].set(adjustable='box-forced', aspect='equal')

    axes[0][4].scatter(new_ycen,
                       new_xcen,
                       marker='+',
                       s=100,
                       color='black',
                       linewidths=3)
    axes[0][4].text(5,2, r'\textbf{$\boldsymbol{\sigma_{int}}$ (km s$\boldsymbol{^{-1}}$)}',
                    color='black',
                    fontsize=16)
    axes[0][4].tick_params(axis='both',
                           which='major',
                           labelsize=10,
                           length=4,
                           width=4)
    [i.set_linewidth(4.0) for i in axes[0][4].spines.itervalues()]

    # OIII VEL 1-D


    axes[1][3].set_ylabel(r'\textbf{V$\boldsymbol{_{C}}$[kms$\boldsymbol{^{-1}}$]}',
                      fontsize=14,
                      fontweight='bold')
    # tick parameters 
    axes[1][3].tick_params(axis='both',
                       which='major',
                       labelsize=10,
                       length=6,
                       width=4)
    axes[1][3].tick_params(axis='both',
                       which='minor',
                       labelsize=10,
                       length=3,
                       width=1)

    axes[1][3].plot(one_d_model_x,
                  one_d_mod_vel_intrinsic,
                  color='red',
                  label='int\_model',
                  lw=2)
    axes[1][3].fill_between(one_d_model_x,
                          one_d_mod_vel_intrinsic_16,
                          one_d_mod_vel_intrinsic_84,
                          facecolor='indianred',
                          alpha=0.5)

#    axes[1][3].scatter(one_d_model_x,
#                    one_d_mod_vel_intrinsic,
#                    marker='o',
#                    color='red')

    axes[1][3].plot(one_d_model_x,
                  one_d_mod_vel_blurred,
                  color='blue',
                  label='blurred\_model',
                  lw=2)

#    axes[1][3].scatter(one_d_model_x,
#                    one_d_mod_vel_blurred,
#                    marker='o',
#                   color='blue')

    axes[1][3].plot(one_d_model_x,
                  one_d_vel_res,
                  color='purple',
                  label='residuals',
                  lw=2)

    axes[1][3].scatter(one_d_model_x,
                     one_d_vel_res,
                     marker='^',
                     color='purple',
                     s=25)

    (_, caps, _) =  axes[1][3].errorbar(one_d_model_x,
                                      one_d_data_vel,
                                      yerr=one_d_data_vel_errors,
                                      fmt='+',
                                      color='black',
                                      label='data',
                                      capsize=2,
                                      elinewidth=1,
                                      alpha=0.7)

    for cap in caps:
        cap.set_markeredgewidth(2)

    # new addition showing the error region


    axes[1][3].fill_between(one_d_model_x,
                          one_d_mod_vel_blurred_16,
                          one_d_mod_vel_blurred_84,
                          facecolor='cornflowerblue',
                          alpha=0.5)

    axes[1][3].set_xlim(-1.2, 1.2)


    axes[1][3].axhline(0, color='silver', ls='-.')
    axes[1][3].axvline(0, color='silver', ls='-.')
    axes[1][3].axhline(va, color='silver', ls='--')
    axes[1][3].axhline(-1.*va, color='silver', ls='--')

    # Also add in vertical lines for where the kinematics 
    # should be extracted

    axes[1][3].plot([r_d_arc_conv_30, r_d_arc_conv_30], [-1*va, va],
                  color='blue',
                  ls='--',
                  lw=2)



    axes[1][3].plot([-r_d_arc_conv_30, -r_d_arc_conv_30], [-1*va, va],
                  color='blue',
                  ls='--',
                  lw=2)

    axes[1][3].plot([r_d_arc_30, r_d_arc_30], [-1*va, va],
                  color='red',
                  ls='--',
                  lw=2)
    


    axes[1][3].plot([-r_d_arc_30, -r_d_arc_30], [-1*va, va],
                  color='red',
                  ls='--',
                  lw=2)

    [i.set_linewidth(4.0) for i in axes[1][3].spines.itervalues()]
    [i.set_linewidth(4.0) for i in axes[1][4].spines.itervalues()]

    axes[1][3].minorticks_on()

    #leg = axes[1][3].legend(loc='upper left',fancybox=True, prop={'size':6})
    #leg.get_frame().set_alpha(0.5)

    # OIII SIG 1-D



    # axes[1][4].set(adjustable='box-forced', aspect='equal')

    axes[1][4].set_ylabel(r'\textbf{$\boldsymbol{\sigma_{int}}$[kms$\boldsymbol{^{-1}}$]}',
                      fontsize=14,
                      fontweight='bold')

    # tick parameters 
    axes[1][4].tick_params(axis='both',
                       which='major',
                       labelsize=10,
                       length=6,
                       width=4)
    axes[1][4].tick_params(axis='both',
                       which='minor',
                       labelsize=10,
                       length=3,
                       width=1)

    axes[1][4].minorticks_on()
    # 1D DISPERSION PLOT

    axes[1][4].errorbar(one_d_model_x,
                   one_d_data_sig,
                   yerr=one_d_data_sig_errors,
                   fmt='o',
                   color='blue',
                   label='obs\_sig')

    axes[1][4].errorbar(one_d_model_x,
                   one_d_sig_int,
                   yerr=one_d_data_sig_errors,
                   fmt='o',
                   color='red',
                   label='int\_sig')

    # instead of plotting the turquoise
    # plot instead horizontal lines showing 
    # intrinsic and observed velocity dispersions

    axes[1][4].axhline(sigma_o, color='blue', ls='-.', lw=2)
    axes[1][4].axhline(sigma_o_i, color='red', ls='-.', lw=2)

#    axes[1][4].plot(one_d_model_x,
#                  one_d_sig_model_full - sigma,
#                  color='turquoise',
#                  label='sig\_model')
#    axes[1][4].scatter(one_d_model_x,
#                    one_d_sig_model_full - sigma,
#                    marker='o',
#                  color='turquoise')

#    axes[1][4].plot(one_d_model_x,
#                  one_d_sig_res,
#                  color='purple')
#    axes[1][4].scatter(one_d_model_x,
#                    one_d_sig_res,
#                    marker='o',
#                  color='purple',
#                  label='sig\_residuals')   

    axes[1][4].axvline(0, color='silver', ls='-.')
#    axes[1][4].axvline(r_d_arc_30, color='cornflowerblue', ls='--', lw=2)
#    axes[1][4].axvline(-r_d_arc_30, color='cornflowerblue', ls='--', lw=2)
#    axes[1][4].axvline(r_d_arc_conv_30, color='indianred', ls='--', lw=2)
#    axes[1][4].axvline(-r_d_arc_conv_30, color='indianred', ls='--', lw=2)
    axes[1][4].set_xlim(-1.2, 1.2)


    # axes[1][4].set_ylabel('velocity (kms$^{-1}$)')

    #leg = axes[1][4].legend(loc='upper left',fancybox=True, prop={'size':6})
    #leg.get_frame().set_alpha(0.5)
    fig.tight_layout()
    plt.show()
    fig.savefig('%s_grid_paper.png' % infile[:-5])
    plt.close('all')

    # some calculations for the final table

    # extracting the maximum velocity from the data
    data_velocity_value = (abs(np.nanmax(one_d_data_vel)) + \
                            abs(np.nanmin(one_d_data_vel))) / 2.0

    # and also want the associated velocity error
    minimum_vel_error = one_d_data_vel_errors[np.nanargmin(one_d_data_vel)]
    maximum_vel_error = one_d_data_vel_errors[np.nanargmax(one_d_data_vel)]


    # and combine in quadrature
    data_velocity_error = 0.5 * np.sqrt(minimum_vel_error**2 + maximum_vel_error**2)

    # sigma maps error
    # in quadrature take the last few values
    # in the actual data
    low_sigma_index, high_sigma_index = rt_pa.find_first_valid_entry(one_d_data_sig) 
    data_sigma_error = 0.5 * np.sqrt(one_d_data_sig_errors[low_sigma_index]**2 + one_d_data_sig_errors[high_sigma_index]**2)
    mean_sigma_error = np.nanmedian(one_d_data_sig_errors)

    # numerical value of sigma at the edges
    data_sigma_value = 0.5 * (one_d_data_sig[low_sigma_index] + one_d_data_sig[high_sigma_index])

    # sigma maps error
    # in quadrature take the last few values
    # in the actual data
    low_sigma_index_int, high_sigma_index_int = rt_pa.find_first_valid_entry(one_d_sig_int) 
    data_sigma_error_int = 0.5 * np.sqrt(one_d_data_sig_errors[low_sigma_index_int]**2 + one_d_data_sig_errors[high_sigma_index_int]**2)

    # numerical value of sigma at the edges
    data_sigma_value_int = 0.5 * (one_d_sig_int[low_sigma_index_int] + one_d_sig_int[high_sigma_index_int])


    b_data_velocity_value = (abs(np.nanmax(best_pa_vel)) + \
                              abs(np.nanmin(best_pa_vel))) / 2.0

    # and for the rotated position angle errors
    min_v_error_rpa = best_pa_error[np.nanargmin(best_pa_vel)]
    max_v_error_rpa = best_pa_error[np.nanargmax(best_pa_vel)]

    # and combine in quadrature
    rt_pa_observed_velocity_error = 0.5 * np.sqrt(min_v_error_rpa**2 + min_v_error_rpa**2)

    h_data_velocity_value = (abs(np.nanmax(hst_pa_vel)) + \
                              abs(np.nanmin(hst_pa_vel))) / 2.0

    max_data_velocity_value = np.nanmax(abs(one_d_data_vel))

    b_max_data_velocity_value = np.nanmax(abs(best_pa_vel))

    h_max_data_velocity_value = np.nanmax(abs(hst_pa_vel))

    # extract from both the 1d and 2d models at the special radii
    # defined as the 90 percent light and 1.8r_e and also 
    # find the radius at which the data extends to

    arc_num_r_9 = scaled_num_r_9 / scale

    # get the velocity indices

    extended_r = np.arange(-10, 10, 0.01)

    ex_r_22_idx = np.argmin(abs(r_d_arc_conv_22 - extended_r))

    ex_3Rd_idx = np.argmin(abs(r_d_arc_conv_30 - extended_r))

    ex_r_9_idx = np.argmin(abs(arc_num_r_9 - extended_r))

    one_d_model_x_r_22_idx = np.argmin(abs(r_d_arc_conv_22 - one_d_model_x))

    one_d_model_x_3Rd_idx = np.argmin(abs(r_d_arc_conv_30 - one_d_model_x))

    r_22_idx_unconvolved = np.argmin(abs(r_d_arc_22 - one_d_model_x))

    r_30_idx_unconvolved = np.argmin(abs(r_d_arc_30 - one_d_model_x))

    one_d_model_x_r_9_idx = np.argmin(abs(arc_num_r_9 - one_d_model_x))

    # find the associated velocity values from data

    d_extrapolation = dyn_pa_fit.eval(r=extended_r)

    b_extrapolation = best_pa_fit.eval(r=extended_r)

    h_extrapolation = hst_pa_fit.eval(r=extended_r)

    # need to know the constants in the fitting to subtract from
    # the inferred velocity values

    dyn_constant = dyn_pa_fit.best_values['const']

    rot_constant = best_pa_fit.best_values['const']

    hst_constant = hst_pa_fit.best_values['const']

    # and find the extrapolation values, sans constants

    dyn_v22 = d_extrapolation[ex_r_22_idx] - dyn_constant

    dyn_v3Rd = d_extrapolation[ex_3Rd_idx] - dyn_constant

    dyn_v9 = d_extrapolation[ex_r_9_idx] - dyn_constant

    b_v22 = b_extrapolation[ex_r_22_idx] - rot_constant

    b_v3Rd = b_extrapolation[ex_3Rd_idx] - rot_constant

    b_v9 = b_extrapolation[ex_r_9_idx] - rot_constant

    h_v22 = h_extrapolation[ex_r_22_idx] - hst_constant

    h_v9 = h_extrapolation[ex_r_22_idx] - hst_constant

    # THE INTRINSIC MODEL PARAMETERS KINEMATIC EXTRACTION

    # Working out v_2d_r22 and corresponding errors

    # 50th percentile intrinsic velocity
    v_2d_r22 = one_d_mod_vel_intrinsic[r_22_idx_unconvolved]

    # 84th percentile intrinsic velocity
    v_84_2d_r22 = one_d_mod_vel_intrinsic_84[r_22_idx_unconvolved]

    # 16th percentile intrinsic velocity
    v_16_2d_r22 = one_d_mod_vel_intrinsic_16[r_22_idx_unconvolved]

    # mean observation error along kinematic axis
    mean_obs_error = np.nanmean(one_d_data_vel_errors)

    # difference at 84th percentile
    v_22_84_diff = abs(v_84_2d_r22 - v_2d_r22)

    # difference at 16th percentile
    v_22_16_diff = abs(v_16_2d_r22 - v_2d_r22)

    # v22 upper error
    v_2d_r22_upper_error = np.sqrt(mean_obs_error**2 + v_22_84_diff**2)

    # v22 lower error
    v_2d_r22_lower_error = np.sqrt(mean_obs_error**2 + v_22_16_diff**2)

    print 'V22 UPPER AND LOWER ERROR: %s %s %s' % (v_2d_r22, v_2d_r22_upper_error, v_2d_r22_lower_error)

    # 50th percentile intrinsic velocity
    v_2d_3Rd = one_d_mod_vel_intrinsic[r_30_idx_unconvolved]

    # 84th percentile intrinsic velocity
    v_84_2d_3Rd = one_d_mod_vel_intrinsic_84[r_30_idx_unconvolved]

    # 16th percentile intrinsic velocity
    v_16_2d_3Rd = one_d_mod_vel_intrinsic_16[r_30_idx_unconvolved]

    # mean observation error along kinematic axis
    mean_obs_error = np.nanmean(one_d_data_vel_errors)

    # difference at 84th percentile
    v_3_84_diff = abs(v_84_2d_3Rd - v_2d_3Rd)

    # difference at 16th percentile
    v_3_16_diff = abs(v_16_2d_3Rd - v_2d_3Rd)

    # v22 upper error
    v_2d_3Rd_upper_error = np.sqrt(mean_obs_error**2 + v_3_84_diff**2)

    # v22 lower error
    v_2d_3Rd_lower_error = np.sqrt(mean_obs_error**2 + v_3_16_diff**2)

    print 'V3 UPPER AND LOWER ERROR: %s %s %s' % (v_2d_3Rd, v_2d_3Rd_upper_error, v_2d_3Rd_lower_error)

    # THE SMEARED MODEL PARAMETERS KINEMATIC EXTRACTION

    # 50th percentile intrinsic velocity
    v_2d_r22_blurred = one_d_mod_vel_blurred[one_d_model_x_r_22_idx]

    # 84th percentile intrinsic velocity
    v_84_2d_r22_blurred = one_d_mod_vel_blurred_84[one_d_model_x_r_22_idx]

    # 16th percentile intrinsic velocity
    v_16_2d_r22_blurred = one_d_mod_vel_blurred_16[one_d_model_x_r_22_idx]

    # mean observation error along kinematic axis
    mean_obs_error = np.nanmean(one_d_data_vel_errors)

    # difference at 84th percentile
    v_22_84_diff_blurred = abs(v_84_2d_r22_blurred - v_2d_r22_blurred)

    # difference at 16th percentile
    v_22_16_diff_blurred = abs(v_16_2d_r22_blurred - v_2d_r22_blurred)

    # v22 upper error
    v_2d_r22_upper_error_blurred = np.sqrt(mean_obs_error**2 + v_22_84_diff_blurred**2)

    # v22 lower error
    v_2d_r22_lower_error_blurred = np.sqrt(mean_obs_error**2 + v_22_16_diff_blurred**2)

    print 'V22 UPPER AND LOWER ERROR: %s %s %s' % (v_2d_r22_blurred, v_2d_r22_upper_error_blurred, v_2d_r22_lower_error_blurred)

    # 50th percentile intrinsic velocity
    v_2d_3Rd_blurred = one_d_mod_vel_blurred[one_d_model_x_3Rd_idx]

    # 84th percentile intrinsic velocity
    v_84_2d_3Rd_blurred = one_d_mod_vel_blurred_84[one_d_model_x_3Rd_idx]

    # 16th percentile intrinsic velocity
    v_16_2d_3Rd_blurred = one_d_mod_vel_blurred_16[one_d_model_x_3Rd_idx]

    # mean observation error along kinematic axis
    mean_obs_error = np.nanmean(one_d_data_vel_errors)

    # difference at 84th percentile
    v_3_84_diff_blurred = abs(v_84_2d_3Rd_blurred - v_2d_3Rd_blurred)

    # difference at 16th percentile
    v_3_16_diff_blurred = abs(v_16_2d_3Rd_blurred - v_2d_3Rd_blurred)

    # v22 upper error
    v_2d_3Rd_upper_error_blurred = np.sqrt(mean_obs_error**2 + v_3_84_diff_blurred**2)

    # v22 lower error
    v_2d_3Rd_lower_error_blurred = np.sqrt(mean_obs_error**2 + v_3_16_diff_blurred**2)

    print 'V3 UPPER AND LOWER ERROR: %s %s %s' % (v_2d_3Rd_blurred, v_2d_3Rd_upper_error_blurred, v_2d_3Rd_lower_error_blurred)

    v_2d_r9 = one_d_mod_vel_intrinsic[one_d_model_x_r_9_idx]

    v_smeared_2d_r22 = one_d_mod_vel_blurred[one_d_model_x_r_22_idx]

    v_smeared_2d_3Rd = one_d_mod_vel_blurred[one_d_model_x_3Rd_idx]

    v_smeared_2d_r9 = one_d_mod_vel_blurred[one_d_model_x_r_9_idx]

    # calculate the upper and lower mean sigma errors

    sigma_lower_diff = abs(sigma_o_i - sigma_o_i_84_40)

    sigma_upper_diff = abs(sigma_o_i - sigma_o_i_16_80)

    sigma_lower_error = np.sqrt(sigma_lower_diff**2 + mean_sigma_error**2)

    sigma_upper_error = np.sqrt(sigma_upper_diff**2 + mean_sigma_error**2)

    # also want to figure out the radius of the last velocity
    # point in the dyn, hst, rot extraction regimes

    s, e = rt_pa.find_first_valid_entry(one_d_data_vel)

    dyn_pa_extent = scale * np.nanmax([one_d_model_x[s], one_d_model_x[e]])

    s, e = rt_pa.find_first_valid_entry(best_pa_x)

    rot_pa_extent = scale * np.nanmax([best_pa_x[s], best_pa_x[e]])

    s, e = rt_pa.find_first_valid_entry(hst_pa_x)

    hst_pa_extent = scale * np.nanmax([hst_pa_x[s], hst_pa_x[e]])

    # assume for now that q = 0.15
    q = 0.2

    inclination_galfit = np.arccos(np.sqrt((axis_r**2 - q**2)/(1 - q**2)))

    inclination_num = np.arccos(np.sqrt((num_axis_ratio**2 - q**2)/(1 - q**2)))

    # comparing with Durham beam smearing values
    rd_psf = r_d_arc * (2.0 / seeing)

    # the velocity
    if v_smeared_2d_3Rd > 0 and v_smeared_2d_3Rd < 50:

        trigger = 1

    elif v_smeared_2d_3Rd > 50 and v_smeared_2d_3Rd < 100:

        trigger = 2

    elif v_smeared_2d_3Rd > 100 and v_smeared_2d_3Rd < 150:

        trigger = 3

    else:

        trigger = 4

    print 'TRIGGER: %s' % trigger

    dur_vel_val = dur_smear.compute_velocity_smear_from_ratio(rd_psf, v_smeared_2d_r22)
    dur_vel_val_3 = dur_smear.compute_velocity_smear_from_ratio_3(rd_psf, v_smeared_2d_3Rd)
    dur_mean_sig = dur_smear.compute_mean_sigma_smear_from_ratio(rd_psf, sigma_o , trigger)
    dur_outer_sig = dur_smear.compute_outer_sigma_smear_from_ratio(rd_psf, data_sigma_value , trigger)

    # Dynamical mass computation (still to filter through errors on that)
    mdyn_22 = np.log10(((r_d_22 * 3.089E19 * (abs(v_2d_r22 / np.sin(inclination_galfit)) * 1000)**2) / 1.3267E20))
    mdyn_30 = np.log10(((r_d_30 * 3.089E19 * (abs(v_2d_3Rd / np.sin(inclination_galfit)) * 1000)**2) / 1.3267E20))

    # V over sigma and error
    v_corred = v_2d_3Rd / np.sin(inclination_galfit)
    v_over_sigma = v_corred / sigma_o_i
    v_over_sigma_error_upper = v_over_sigma * np.sqrt((v_2d_3Rd_upper_error/v_corred)**2 + (sigma_lower_error/sigma_o_i)**2)
    v_over_sigma_error_lower = v_over_sigma * np.sqrt((v_2d_3Rd_lower_error/v_corred)**2 + (sigma_upper_error/sigma_o_i)**2)

    # S0.5 and error
    s0 = np.sqrt(0.5*((v_2d_3Rd / np.sin(inclination_galfit))**2) + sigma_o_i**2)
    s0_upper_error = np.sqrt(((v_2d_3Rd / np.sin(inclination_galfit)) * v_2d_3Rd_upper_error / 2.0)**2 + (sigma_o_i * sigma_upper_error)**2) / s0
    s0_lower_error = np.sqrt(((v_2d_3Rd / np.sin(inclination_galfit)) * v_2d_3Rd_lower_error / 2.0)**2 + (sigma_o_i * sigma_lower_error)**2) / s0
    log_s0 = np.log10(s0)
    log_s0_upper_error = 0.434 * (s0_upper_error/s0)
    log_s0_lower_error = 0.434 * (s0_lower_error/s0)

    # finally if the kinematic inclination angle is
    # greater than pi, subtract pi to make if comparable to HST

    if pa > np.pi:
        pa = pa - np.pi
        pa_16 = pa_16 - np.pi
        pa_84 = pa_84 - np.pi

    # errors on the position angle
    pa_upper_error = abs(pa_84 - pa)
    pa_lower_error = abs(pa_16 - pa)

    # and calculate the difference in position angles from HST to 
    # dynamic for each galaxy
    delta_pa = abs(pa - hst_pa)

    # if greater than 90 do a trick
    if delta_pa > np.pi / 2.:
        delta_pa = np.pi / 2. - (delta_pa - np.pi / 2.)

    print 'THIS IS THE DATA VELOCITY VALUE: %s %s' % (data_velocity_value, np.sin(inclination_galfit))

    data_values = [gal_name[26:-5],
                   int(abs(data_velocity_value)),
                   int(data_velocity_error),
                   abs(b_data_velocity_value / np.sin(inclination_galfit)),
                   rt_pa_observed_velocity_error,
                   int(sigma_o),
                   int(mean_sigma_error),
                   int(sigma_o_i),
                   int(sigma_upper_error),
                   int(sigma_lower_error),
                   data_sigma_value,
                   data_sigma_error,
                   data_sigma_value_int,
                   data_sigma_error_int,
                   abs(dyn_v22 / np.sin(inclination_galfit)),
                   data_velocity_error,
                   abs(dyn_v3Rd / np.sin(inclination_galfit)),
                   data_velocity_error,
                   abs(b_v22 / np.sin(inclination_galfit)),
                   rt_pa_observed_velocity_error,
                   abs(b_v3Rd / np.sin(inclination_galfit)),
                   rt_pa_observed_velocity_error,
                   abs(v_smeared_2d_r22 / np.sin(inclination_galfit)),
                   data_velocity_error,
                   abs(v_smeared_2d_3Rd / np.sin(inclination_galfit)),
                   data_velocity_error,
                   int(abs(v_2d_r22 / np.sin(inclination_galfit))),
                   int(v_2d_r22_upper_error),
                   int(v_2d_r22_lower_error),
                   int(abs(v_2d_3Rd / np.sin(inclination_galfit))),
                   int(v_2d_3Rd_upper_error),
                   int(v_2d_3Rd_lower_error),
                   int(abs(v_2d_r22_blurred / np.sin(inclination_galfit))),
                   int(v_2d_r22_upper_error_blurred),
                   int(v_2d_r22_lower_error_blurred),
                   int(abs(v_2d_3Rd_blurred / np.sin(inclination_galfit))),
                   int(v_2d_3Rd_upper_error_blurred),
                   int(v_2d_3Rd_lower_error_blurred),
                   np.round(mdyn_22,2),
                   np.round(mdyn_30, 2),
                   np.round(v_over_sigma,2),
                   np.round(v_over_sigma_error_upper, 2),
                   np.round(v_over_sigma_error_lower, 2),
                   dyn_pa_extent,
                   np.round(axis_r,2),
                   int((180 / np.pi) * inclination_galfit),
                   int((180 / np.pi) * hst_pa),
                   int((180 / np.pi) * pa),
                   int((180 / np.pi) * pa_upper_error),
                   int((180 / np.pi) * pa_lower_error),
                   int((180 / np.pi) * best_pa),
                   int((180 / np.pi) * delta_pa),
                   r_e_arc,
                   r_d_arc,
                   r_d_arc_22,
                   r_d_arc_30,
                   np.round(r_e,2),
                   np.round(r_d,2),
                   np.round(r_d_22,2),
                   np.round(r_d_30,2),
                   np.round(rd_psf,2),
                   int(dur_vel_val / np.sin(inclination_galfit)),
                   int(dur_vel_val_3 / np.sin(inclination_galfit)),
                   int(dur_mean_sig),
                   int(dur_outer_sig),
                   s0,
                   s0_upper_error,
                   s0_lower_error,
                   log_s0,
                   log_s0_upper_error,
                   log_s0_lower_error]

    print 'CONSTANTS: %s %s %s' % (dyn_constant, rot_constant, hst_constant)
    print 'OBSERVED_VELOCITY_DYNAMIC_PA: %s' % abs(data_velocity_value / np.sin(inclination_galfit))
    print 'OBSERVED_VEL_ERROR_DYNAMIC_PA: %s' % data_velocity_error
    print 'OBSERVED_VELOCITY_ROTATED_PA: %s' % abs(b_data_velocity_value / np.sin(inclination_galfit))
    print 'OBSERVED_VELOCITY_ROTATED_PA_ERROR: %s' % rt_pa_observed_velocity_error
    print 'THESE ARE THE SIGMAS'
    print 'ROT SIGMA: %s' % best_mean_sigma
    print 'DYN SIGMA: %s' % dyn_mean_sigma
    print 'MEAN_OBSERVED_SIGMA: %s' % sigma_o
    print 'MEAN INTRINSIC SIGMA: %s' % sigma_o_i
    print 'MEAN SIGMA UPPER ERROR: %s' % sigma_upper_error
    print 'MEAN SIGMA LOWER ERROR: %s' % sigma_lower_error
    print 'MEAN INTRINSIC SIGMA 84: %s' % sigma_o_i_84
    print 'MEAN INTRINSIC SIGMA 16: %s' % sigma_o_i_16
    print 'MEAN INTRINSIC SIGMA MAX: %s' % sigma_o_i_84_40
    print 'MEAN INTRINSIC SIGMA MIN: %s' % sigma_o_i_16_80
    print 'MEAN SIGMA ERROR: %s' % mean_sigma_error
    print 'OBSERVED_SIGMA_DYNAMIC_EDGES: %s' % data_sigma_value
    print 'OBSERVED_SIGMA_ERROR: %s' % data_sigma_error
    print 'INTRINSIC_SIGMA_DYNAMIC_EDGES: %s' % data_sigma_value_int
    print 'OBSERVED_SIGMA_ERROR: %s' % data_sigma_error_int
    print '1D_ALONG_DYN_PA_2.2: %s' % abs(dyn_v22 / np.sin(inclination_galfit))
    print '1D_ALONG_DYN_PA_3Rd: %s' % abs(dyn_v3Rd / np.sin(inclination_galfit))
    print '1D_ALONG_ROTATED_PA_2.2: %s' % abs(b_v22 / np.sin(inclination_galfit))
    print '1D_ALONG_ROTATED_PA_3Rd: %s' % abs(b_v3Rd / np.sin(inclination_galfit))
    print '2D_ALONG_DYN_PA_2.2: %s' % abs(v_2d_r22 / np.sin(inclination_galfit))
    print 'DURHAM 2D_2.2: %s' % (dur_vel_val / np.sin(inclination_galfit))
    print '2D_ALONG_DYN_PA_3Rd: %s' % abs(v_2d_3Rd / np.sin(inclination_galfit))
    print 'DURHAM 2D_3.0: %s' % (dur_vel_val_3 / np.sin(inclination_galfit))
    print 'V_OVER_SIGMA: %s' % v_over_sigma
    print 'V_OVER_SIGMA_ERROR_UPPER: %s' % v_over_sigma_error_upper
    print 'V_OVER_SIGMA_ERROR_LOWER: %s' % v_over_sigma_error_lower
    print 'AXIS RATIO: %s' % axis_r
    print 'GALFIT INCLINATION %s' % inclination_galfit
    print 'HST_PA: %s' % hst_pa
    if pa > np.pi:
        pa = pa - np.pi
    print 'DYN_PA: %s' % pa
    if best_pa > np.pi:
        best_pa = best_pa - np.pi
    print 'BEST_PA: %s' % best_pa
    print 'EFFECTIVE RADIUS: %s' % r_e_arc
    print 'EFFECTIVE RADIUS Kpcs %s' % r_e
    print 'Rd/RPSF: %s' % rd_psf

    return data_values

def multi_make_all_plots_mcmc_version(infile,
                                      r_aper,
                                      d_aper,
                                      seeing,
                                      sersic_n,
                                      sigma,
                                      pix_scale,
                                      psf_factor,
                                      sersic_factor,
                                      m_factor,
                                      galaxy_boundaries_file,
                                      inc_error=0.1,
                                      sig_error=10,
                                      smear=True):

    # create the table names

    column_names = ['Name',
                    'observed_vmax_dyn_pa',
                    'observed_vmax_dyn_pa_error',
                    'observed_vmax_rt_pa',
                    'observed_vmax_rt_pa_error',
                    'mean_observed_sigma',
                    'mean_sigma_error',
                    'mean_intrinsic_sigma',
                    'Sigma_upper_error',
                    'Sigma_lower_error',
                    'observed_sigma_edges',
                    'observed_sigma_edges_error',
                    'intrinsic_sigma_edges',
                    'intrinsic_sigma_edges_error',
                    '1d_dyn_pa_r_2.2',
                    '1d_dyn_pa_r_2.2_error',
                    '1d_dyn_pa_3Rd',
                    '1d_dyn_pa_3Rd_error',
                    '1d_rpa_r2.2',
                    '1d_rpa_r2.2_error',
                    '1d_rpa_3Rd',
                    '1d_rpa_3Rd_error',
                    '2d_beam_smeared_Vmax_r2.2',
                    '2d_beam_smeared_Vmax_r2.2_error',
                    '2d_beam_smeared_Vmax_3Rd',
                    '2d_beam_smeared_Vmax_3Rd_error',
                    '2d_intrinsic_Vmax_r2.2',
                    '2d_intrinsic_Vmax_r2.2_upper_error',
                    '2d_intrinsic_Vmax_r2.2_lower_error',
                    '2d_intrinsic_Vmax_3Rd',
                    '2d_intrinsic_Vmax_3Rd_upper_error',
                    '2d_intrinsic_Vmax_3Rd_lower_error',
                    '2d_blurred_Vmax_r2.2',
                    '2d_blurred_Vmax_r2.2_upper_error',
                    '2d_blurred_Vmax_r2.2_lower_error',
                    '2d_blurred_Vmax_3Rd',
                    '2d_blurred_Vmax_3Rd_upper_error',
                    '2d_blurred_Vmax_3Rd_lower_error',
                    'Mdyn_2.2',
                    'Mdyn_3.0',
                    'v_over_sigma',
                    'v_over_sigma_error_upper',
                    'v_over_sigma_error_lower',
                    'Last_data_radius',
                    'axis_ratio',
                    'inclination',
                    'HST_PA',
                    'DYN_PA',
                    'DYN_PA_upper_error',
                    'DYN_PA_lower_error',
                    'R_PA',
                    'Delta_PA',
                    'R_e(arcsec)',
                    'R_d(arcsec)',
                    '2.2R_d(arcsec)',
                    '3R_d(arcsec)',
                    'R_e(Kpc)',
                    'R_d(Kpc)',
                    '2.2R_d(Kpc)',
                    '3R_d(Kpc)',
                    'Rd/Rpsf',
                    'Durham_Vel_corr_2Rd',
                    'Durham_Vel_corr_3Rd',
                    'Durham_Sig_mean_corr',
                    'Durham_Sig_outer_corr',
                    's0',
                    's0_upper_error',
                    's0_lower_error',
                    'log_s0',
                    'log_s0_upper_error',
                    'log_s0_lower_error']


    save_dir = '/disk2/turner/disk1/turner/DATA/kmos_dynamics_paper_plots/'

    big_list = []

    # read in the table of cube names
    Table = ascii.read(infile)

    # counter for galaxy boundaries file
    gal_num = 0

    # assign variables to the different items in the infile
    for entry in Table:

        obj_name = entry[0]

        cube = cubeOps(obj_name)

        xpix = cube.data.shape[1]

        ypix = cube.data.shape[2]

        wave_array = cube.wave_array

        redshift = entry[1]

        xcen = entry[10]

        ycen = entry[11]

        inc = entry[12]

        r_e = entry[16]

        sersic_pa = entry[17]

        hst_x_cen = entry[20]

        hst_y_cen = entry[21]

        a_r = np.sqrt((np.cos(inc) * np.cos(inc)) * (1 - (0.2**2)) + 0.2 ** 2)

        sersic_field = psf.sersic_2d_astropy(dim_x=ypix,
                                             dim_y=xpix,
                                             rt=r_e,
                                             n=1.0,
                                             a_r=a_r,
                                             pa=sersic_pa,
                                             xcen=xcen,
                                             ycen=ycen,
                                             sersic_factor=sersic_factor)

        big_list.append(make_all_plots_mcmc_version(inc,
                                                    redshift,
                                                    wave_array,
                                                    xcen,
                                                    ycen,
                                                    obj_name,
                                                    r_aper,
                                                    d_aper,
                                                    seeing,
                                                    sersic_n,
                                                    sigma,
                                                    pix_scale,
                                                    psf_factor,
                                                    sersic_factor,
                                                    m_factor,
                                                    sersic_field,
                                                    galaxy_boundaries_file,
                                                    gal_num,
                                                    hst_x_cen,
                                                    hst_y_cen,
                                                    inc_error,
                                                    sig_error,
                                                    smear))

        gal_num += 1
    
    # create the table
    make_table.table_create(column_names,
                            big_list,
                            save_dir,
                            'lbg_121.cat')

infile = '/disk2/turner/disk1/turner/PhD/PAPER_3_KDS_METALLICITIES/OBJECT_NAMES/k_band_names.txt'
boundaries_file = '/disk2/turner/disk1/turner/DATA/goods_iso_boundaries.txt'
#multi_make_all_plots_fixed_inc_fixed(infile=infile,
#                                     r_aper=0.8,
#                                     d_aper=0.6,
#                                     seeing=0.5,
#                                     sersic_n=1.0,
#                                     sigma=50,
#                                     pix_scale=0.1,
#                                     psf_factor=1,
#                                     sersic_factor=50,
#                                     m_factor=4,
#                                     smear=True)

multi_make_all_plots_mcmc_version(infile=infile,
                                  r_aper=0.8,
                                  d_aper=0.6,
                                  seeing=0.5,
                                  sersic_n=1.0,
                                  sigma=50,
                                  pix_scale=0.1,
                                  psf_factor=1,
                                  sersic_factor=50,
                                  m_factor=6,
                                  galaxy_boundaries_file=boundaries_file,
                                  inc_error=0.1,
                                  sig_error=10,
                                  smear=True)