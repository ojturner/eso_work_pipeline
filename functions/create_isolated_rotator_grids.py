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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import poly1d
from sys import stdout
from matplotlib import rc
from photutils import CircularAperture
from photutils import EllipticalAperture
from photutils import aperture_photometry

# add the class file to the PYTHONPATH
sys.path.append('/scratch2/oturner/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/Class')

# add the functions folder to the PYTHONPATH
sys.path.append('/scratch2/oturner/disk1/turner/PhD'
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

    half_light_dict = ap_growth.find_aperture_parameters(hst_stamp_name)

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

    # Converting back to arcseconds

    r_e_arc = r_e / scale

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
               label='hst_pa')
    ax[1][0].plot([y_low, y_high], [x_low, x_high],
               ls='--',
               color='lightcoral',
               lw=2,
               label='dyn_pa')
    ax[1][0].plot([y_low_best, y_high_best], [x_low_best, x_high_best],
               ls='--',
               color='darkorange',
               lw=2,
               label='rot_pa')
    ax[1][0].plot([y_low_num, y_high_num], [x_low_num, x_high_num],
               ls='--',
               color='wheat',
               lw=2,
               label='num_pa')
    l = ax[1][0].legend(loc='best',
                        frameon=False,
                        prop={'size':10})
    for text in l.get_texts():
        text.set_color("white")
    # velocity plot
    ax[1][1].plot([y_h_low, y_h_high], [x_h_low, x_h_high],
               ls='--',
               color='aquamarine',
               label='hst_pa')
    ax[1][1].plot([y_low, y_high], [x_low, x_high],
               ls='--',
               color='lightcoral',
               lw=2,
               label='dyn_pa')
    ax[1][1].plot([y_low_best, y_high_best], [x_low_best, x_high_best],
               ls='--',
               color='darkorange',
               lw=2,
               label='rot_pa')
    l = ax[1][1].legend(loc='best',
                        frameon=False,
                        prop={'size':10})
    for text in l.get_texts():
        text.set_color("white")
    ax[1][2].plot([y_h_low, y_h_high], [x_h_low, x_h_high],
               ls='--',
               color='aquamarine',
               label='hst_pa')
    ax[1][2].plot([y_low, y_high], [x_low, x_high],
               ls='--',
               color='lightcoral',
               lw=2,
               label='dyn_pa')
    ax[1][2].plot([y_low_best, y_high_best], [x_low_best, x_high_best],
               ls='--',
               color='darkorange',
               lw=2,
               label='rot_pa')
    l = ax[1][2].legend(loc='best',
                        frameon=False,
                        prop={'size':10})
    for text in l.get_texts():
        text.set_color("white")
    ax[1][3].plot([y_h_low, y_h_high], [x_h_low, x_h_high],
               ls='--',
               color='aquamarine',
               label='hst_pa')
    ax[1][3].plot([y_low, y_high], [x_low, x_high],
               ls='--',
               color='lightcoral',
               lw=2,
               label='dyn_pa')
    ax[1][3].plot([y_low_best, y_high_best], [x_low_best, x_high_best],
               ls='--',
               color='darkorange',
               lw=2,
               label='rot_pa')
    l = ax[1][3].legend(loc='best',
                        frameon=False,
                        prop={'size':10})
    for text in l.get_texts():
        text.set_color("white")
    ax[1][4].plot([y_h_low, y_h_high], [x_h_low, x_h_high],
               ls='--',
               color='aquamarine',
               label='hst_pa')
    ax[1][4].plot([y_low, y_high], [x_low, x_high],
               ls='--',
               color='lightcoral',
               lw=2,
               label='dyn_pa')
    ax[1][4].plot([y_low_best, y_high_best], [x_low_best, x_high_best],
               ls='--',
               color='darkorange',
               lw=2,
               label='rot_pa')
    l = ax[1][4].legend(loc='best',
                        frameon=False,
                        prop={'size':10})
    for text in l.get_texts():
        text.set_color("white")
    ax[2][4].plot([y_h_low, y_h_high], [x_h_low, x_h_high],
               ls='--',
               color='aquamarine',
               label='hst_pa')
    ax[2][4].plot([y_low, y_high], [x_low, x_high],
               ls='--',
               color='lightcoral',
               lw=2,
               label='dyn_pa')
    ax[2][4].plot([y_low_best, y_high_best], [x_low_best, x_high_best],
               ls='--',
               color='darkorange',
               lw=2,
               label='rot_pa')
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
                         vmax=5,
                         vmin=0)

    # HST - blurred

    blurred_hst = psf.blur_by_psf(data_hst,
                                  0.46,
                                  pix_scale,
                                  psf_factor)

    im = ax[3][3].imshow(blurred_hst,
                         cmap=cmap,
                         vmax=3,
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
                      vmax=5,
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
                      vmax=5,
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
                  label='int_model')

    ax[2][0].scatter(one_d_model_x,
                    one_d_mod_vel_intrinsic,
                    marker='o',
                    color='red')

    ax[2][0].plot(one_d_model_x,
                  one_d_mod_vel_blurred,
                  color='blue',
                  label='blurred_model')

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

    ax[2][0].plot([1.8*r_e_arc, 1.8*r_e_arc], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[2][0].plot([-1*r_e_arc, -1*r_e_arc], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[2][0].plot([-1.8*r_e_arc, -1.8*r_e_arc], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[2][0].plot([-1*scaled_num_r_e / scale, -1*scaled_num_r_e / scale], [-1*va, va],
                  color='wheat',
                  ls='--',
                  lw=2)

    ax[2][0].plot([-1*scaled_num_r_9 / scale, -1*scaled_num_r_9 / scale], [-1*va, va],
                  color='wheat',
                  ls='--',
                  lw=2)

    ax[2][0].plot([1*scaled_num_r_e / scale, 1*scaled_num_r_e / scale], [-1*va, va],
                  color='wheat',
                  ls='--',
                  lw=2)

    ax[2][0].plot([1*scaled_num_r_9 / scale, 1*scaled_num_r_9 / scale], [-1*va, va],
                  color='wheat',
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
                   label='obs_sig')

    ax[2][1].errorbar(one_d_model_x,
                   one_d_sig_int,
                   yerr=one_d_data_sig_errors,
                   fmt='o',
                   color='blue',
                   label='int_sig')

    ax[2][1].plot(one_d_model_x,
                  one_d_sig_model_full,
                  color='orange',
                  label='sig_model')

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
                  label='sig_residuals')   

    ax[2][1].axvline(0, color='silver', ls='-.')
    ax[2][1].axvline(r_e_arc, color='maroon', ls='--', lw=2)
    ax[2][1].axvline(-1*r_e_arc, color='maroon', ls='--', lw=2)
    ax[2][1].axvline(1.8*r_e_arc, color='maroon', ls='--', lw=2)
    ax[2][1].axvline(-1.8*r_e_arc, color='maroon', ls='--', lw=2)
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

    sigma_o = np.average(data_sig[indices],
                         weights=1.0 / data_sig_error[indices])

    indices = ~np.isnan(sig_int)

    sigma_o_i = np.average(sig_int[indices],
                           weights=1.0 / data_sig_error[indices])

    c = 2.99792458E5

    print 'THESE ARE THE SIGMAS'
    print 'INTRINSIC SIGMA: %s' % (sigma_int * c / central_l)
    print 'ROT SIGMA: %s' % best_mean_sigma
    print 'HST SIGMA: %s' % hst_mean_sigma
    print 'DYN SIGMA: %s' % dyn_mean_sigma
    print 'MEAN_OBSERVED_SIGMA: %s' % sigma_o
    print 'MEAN INTRINSIC SIGMA: %s' % sigma_o_i

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
                   label='dyn_pa')

    ax[3][2].plot(best_pa_x,
                  best_pa_fit.eval(r=best_pa_x),
                  color='darkorange')

    ax[3][2].errorbar(best_pa_x,
                      best_pa_vel,
                      yerr=best_pa_error,
                      fmt='o',
                      color='darkorange',
                      label='rot_pa')

    ax[3][2].plot(hst_pa_x,
                  hst_pa_fit.eval(r=hst_pa_x),
                  color='aquamarine')

    ax[3][2].errorbar(hst_pa_x,
                      hst_pa_vel,
                      yerr=hst_pa_error,
                      fmt='o',
                      color='aquamarine',
                      label='hst_pa')

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

    ax[3][2].plot([1.8*r_e_arc, 1.8*r_e_arc], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].plot([-1*r_e_arc, -1*r_e_arc], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].plot([-1.8*r_e_arc, -1.8*r_e_arc], [-1*va, va],
                  color='maroon',
                  ls='--',
                  lw=2)

    ax[3][2].plot([-1*scaled_num_r_e / scale, -1*scaled_num_r_e / scale], [-1*va, va],
                  color='wheat',
                  ls='--',
                  lw=2)

    ax[3][2].plot([-1*scaled_num_r_9 / scale, -1*scaled_num_r_9 / scale], [-1*va, va],
                  color='wheat',
                  ls='--',
                  lw=2)

    ax[3][2].plot([1*scaled_num_r_e / scale, 1*scaled_num_r_e / scale], [-1*va, va],
                  color='wheat',
                  ls='--',
                  lw=2)

    ax[3][2].plot([1*scaled_num_r_9 / scale, 1*scaled_num_r_9 / scale], [-1*va, va],
                  color='wheat',
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

    ex_r_18_idx = np.argmin(abs(1.8*r_e_arc - extended_r))

    ex_3Rd_idx = np.argmin(abs(5.04*r_e_arc - extended_r))

    ex_r_9_idx = np.argmin(abs(arc_num_r_9 - extended_r))

    one_d_model_x_r_18_idx = np.argmin(abs(1.8*r_e_arc - one_d_model_x))

    one_d_model_x_3Rd_idx = np.argmin(abs(5.04*r_e_arc - one_d_model_x))

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

    dyn_v18 = d_extrapolation[ex_r_18_idx] - dyn_constant

    dyn_v3Rd = d_extrapolation[ex_3Rd_idx] - dyn_constant

    dyn_v9 = d_extrapolation[ex_r_9_idx] - dyn_constant

    b_v18 = b_extrapolation[ex_r_18_idx] - rot_constant

    b_v3Rd = b_extrapolation[ex_3Rd_idx] - rot_constant

    b_v9 = b_extrapolation[ex_r_9_idx] - rot_constant

    h_v18 = h_extrapolation[ex_r_18_idx] - hst_constant

    h_v9 = h_extrapolation[ex_r_18_idx] - hst_constant

    v_2d_r18 = one_d_mod_vel_intrinsic[one_d_model_x_r_18_idx]

    v_2d_3Rd = one_d_mod_vel_intrinsic[one_d_model_x_3Rd_idx]

    v_2d_r9 = one_d_mod_vel_intrinsic[one_d_model_x_r_9_idx]

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

    data_values = [gal_name[26:-5],
                   r_e,
                   scaled_num_r_e,
                   scaled_num_r_9,
                   dyn_pa_extent,
                   rot_pa_extent,
                   hst_pa_extent,
                   axis_r,
                   inclination_galfit,
                   num_axis_ratio,
                   inclination_num,
                   hst_pa,
                   pa,
                   best_pa,
                   num_pa,
                   abs(data_velocity_value),
                   abs(data_velocity_value / np.sin(inclination_galfit)),
                   abs(data_velocity_value / np.sin(inclination_num)),
                   abs(max_data_velocity_value),
                   abs(max_data_velocity_value / np.sin(inclination_galfit)),
                   abs(max_data_velocity_value / np.sin(inclination_num)),
                   abs(b_data_velocity_value),
                   abs(b_data_velocity_value / np.sin(inclination_galfit)),
                   abs(b_data_velocity_value / np.sin(inclination_num)),
                   abs(b_max_data_velocity_value),
                   abs(b_max_data_velocity_value / np.sin(inclination_galfit)),
                   abs(b_max_data_velocity_value / np.sin(inclination_num)),
                   abs(h_data_velocity_value),
                   abs(h_data_velocity_value / np.sin(inclination_galfit)),
                   abs(h_data_velocity_value / np.sin(inclination_num)),
                   abs(h_max_data_velocity_value),
                   abs(h_max_data_velocity_value / np.sin(inclination_galfit)),
                   abs(h_max_data_velocity_value / np.sin(inclination_num)),
                   abs(va),
                   abs(va / np.sin(inclination_galfit)),
                   abs(va / np.sin(inclination_num)),
                   abs(v_2d_r18),
                   abs(v_2d_r18 / np.sin(inclination_galfit)),
                   abs(v_2d_r18 / np.sin(inclination_num)),
                   abs(v_2d_r9),
                   abs(v_2d_r9 / np.sin(inclination_galfit)),
                   abs(v_2d_r9 / np.sin(inclination_num)),
                   abs(dyn_v18),
                   abs(dyn_v18 / np.sin(inclination_galfit)),
                   abs(dyn_v18 / np.sin(inclination_num)),
                   abs(dyn_v9),
                   abs(dyn_v9 / np.sin(inclination_galfit)),
                   abs(dyn_v9 / np.sin(inclination_num)),
                   abs(dyn_pa_fit.best_values['vasy'] - dyn_constant),
                   abs(b_v18),
                   abs(b_v18 / np.sin(inclination_galfit)),
                   abs(b_v18 / np.sin(inclination_num)),
                   abs(b_v9),
                   abs(b_v9 / np.sin(inclination_galfit)),
                   abs(b_v9 / np.sin(inclination_num)),
                   abs(best_pa_fit.best_values['vasy'] - rot_constant),
                   abs(h_v18),
                   abs(h_v18 / np.sin(inclination_galfit)),
                   abs(h_v18 / np.sin(inclination_num)),
                   abs(h_v9),
                   abs(h_v9 / np.sin(inclination_galfit)),
                   abs(h_v9 / np.sin(inclination_num)),
                   abs(hst_pa_fit.best_values['vasy'] - hst_constant),
                   hst_mean_sigma,
                   dyn_mean_sigma,
                   best_mean_sigma,
                   (sigma_int * c) / central_l,
                   sigma_o]

    print 'CONSTANTS: %s %s %s' % (dyn_constant, rot_constant, hst_constant)
    print 'OBSERVED_VELOCITY_DYNAMIC_PA: %s' % abs(data_velocity_value / np.sin(inclination_galfit))
    print 'OBSERVED_VEL_ERROR_DYNAMIC_PA: %s' % data_velocity_error
    print 'OBSERVED_VELOCITY_ROTATED_PA: %s' % abs(b_data_velocity_value / np.sin(inclination_galfit))
    print 'OBSERVED_VELOCITY_ROTATED_PA_ERROR: %s' % rt_pa_observed_velocity_error
    print 'MEAN SIGMA ERROR: %s' % mean_sigma_error
    print 'OBSERVED_SIGMA_DYNAMIC_PA: %s' % data_sigma_value
    print 'OBSERVED_SIGMA_ERROR: %s' % data_sigma_error
    print '1D_ALONG_DYN_PA_1.8: %s' % abs(dyn_v18 / np.sin(inclination_galfit))
    print '1D_ALONG_DYN_PA_3Rd: %s' % abs(dyn_v3Rd / np.sin(inclination_galfit))
    print '1D_ALONG_ROTATED_PA_1.8: %s' % abs(b_v18 / np.sin(inclination_galfit))
    print '1D_ALONG_ROTATED_PA_3Rd: %s' % abs(b_v3Rd / np.sin(inclination_galfit))
    print '2D_ALONG_DYN_PA_1.8: %s' % abs(v_2d_r18 / np.sin(inclination_galfit))
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
                    'Galfit_R_e(Kpc)',
                    'Numerical_R_e(Kpc)',
                    'Numerical_R_9(Kpc)',
                    'Dynamic_PA_extent(Kpc)',
                    'Rot_PA_extent(Kpc)',
                    'HST_PA_extent(Kpc)',
                    'Galfit_Ar',
                    'i_galfit',
                    'Numerical_Ar',
                    'i_num',
                    'hst_pa',
                    'dynamical_pa',
                    'rotation_pa',
                    'Numerical_pa',
                    'half_method_data_Velocity',
                    'half_method_data_Velocity_g_ar',
                    'half_method_data_Velocity_m_ar',
                    'max_method_data_Velocity',
                    'max_method_data_Velocity_g_ar',
                    'max_method_data_Velocity_m_ar',
                    'half_method_rot_pa_data_Velocity',
                    'half_method_rot_pa_data_Velocity_g_ar',
                    'half_method_rot_pa_data_Velocity_m_ar',
                    'max_method_rot_pa_data_Velocity',
                    'max_method_rot_pa_data_Velocity_g_ar',
                    'max_method_rot_pa_data_Velocity_m_ar',
                    'half_method_hst_pa_data_Velocity',
                    'half_method_hst_pa_data_Velocity_g_ar',
                    'half_method_hst_pa_data_Velocity_m_ar',
                    'max_method_hst_pa_data_Velocity',
                    'max_method_hst_pa_data_Velocity_g_ar',
                    'max_method_hst_pa_data_Velocity_m_ar',
                    'Maximum_2d_model_velocity',
                    'Maximum_2d_model_velocity_g_ar',
                    'Maximum_2d_model_velocity_m_ar',
                    '2d_model_velocity_1.8',
                    '2d_model_velocity_1.8_g_ar',
                    '2d_model_velocity_1.8_m_ar',
                    '2d_model_velocity_9',
                    '2d_model_velocity_9_g_ar',
                    '2d_model_velocity_9_m_ar',
                    '1d_model_velocity_dyn_pa_1.8',
                    '1d_model_velocity_dyn_pa_1.8_g_ar',
                    '1d_model_velocity_dyn_pa_1.8_m_ar',
                    '1d_model_velocity_dyn_pa_9',
                    '1d_model_velocity_dyn_pa_9_g_ar',
                    '1d_model_velocity_dyn_pa_9_m_ar',
                    '1d_model_velocity_dyn_limit',
                    '1d_model_velocity_rot_pa_1.8',
                    '1d_model_velocity_rot_pa_1.8_g_ar',
                    '1d_model_velocity_rot_pa_1.8_m_ar',
                    '1d_model_velocity_rot_pa_9',
                    '1d_model_velocity_rot_pa_9_g_ar',
                    '1d_model_velocity_rot_pa_9_m_ar',
                    '1d_model_velocity_rot_limit',
                    '1d_model_velocity_hst_pa_1.8',
                    '1d_model_velocity_hst_pa_1.8_g_ar',
                    '1d_model_velocity_hst_pa_1.8_m_ar',
                    '1d_model_velocity_hst_pa_9',
                    '1d_model_velocity_hst_pa_9_g_ar',
                    '1d_model_velocity_hst_pa_9_m_ar',
                    '1d_model_velocity_hst_limit',
                    'HST_sigma',
                    'DYN_sigma',
                    'ROT_sigma',
                    'INTRINSIC_sigma',
                    'WEIGHTED_sigma']


    save_dir = '/disk1/turner/DATA/new_comb_calibrated/'

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
                            save_dir)

infile = '/scratch2/oturner/disk1/turner/DATA/goods_isolated_rotators_names.txt'
multi_make_all_plots_fixed_inc_fixed(infile=infile,
                                     r_aper=0.8,
                                     d_aper=0.6,
                                     seeing=0.5,
                                     sersic_n=1.0,
                                     sigma=50,
                                     pix_scale=0.1,
                                     psf_factor=1,
                                     sersic_factor=50,
                                     m_factor=4,
                                     smear=True)