# modelling seeing conditions
# given a set of input data, blur by PSF of given seeing
# and with given pixel scale
# O.J.Turner 2016

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
import scipy.ndimage.filters as scifilt
from scipy.optimize import curve_fit
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
from astropy.modeling.models import Sersic2D
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel


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

def multi_model_comparison(infile,
                           param_file,
                           seeing,
                           sersic_n=1.0,
                           sigma=50,
                           pix_scale=0.1,
                           psf_factor=1,
                           sersic_factor=50,
                           m_factor=6):

    """
    Def:
    Take the model parameters recovered from the chi-squared minimisation
    and use those to construct the best model to compare with the data.

    Also plot the recovered sigma distribution and the comparison of that
    profile extracted along the kinematic major axis with the observed
    profiles. Output the mean sigma value from that and the sigma value at the
    edges after beam smearing has been corrected for
    """

    # read the parameters infile

    Table_objects = ascii.read(infile)

    Table_params = ascii.read(param_file)

    length = len(Table_objects['Filename'])

    # assign the column names for each row

    for i in range(length):

        # parameter definitions

        cube_name = Table_params[i][0]

        cube = cubeOps(cube_name)

        wave_array = cube.wave_array

        vel_field_name = Table_params[i][0][:-5] + '_vel_field.fits'

        vel_data = fits.open(vel_field_name)[0].data

        sigma_name = Table_params[i][0][:-5] + '_sig_field.fits'

        sigma_data = fits.open(sigma_name)[0].data

        sigma_sky_name = Table_params[i][0][:-5] + '_sig_sky_field.fits'

        sigma_sky_data = fits.open(sigma_sky_name)[0].data

        xcen = Table_params[i][1]

        ycen = Table_params[i][2]

        inc = Table_params[i][3]

        pa = Table_params[i][4]

        rt = Table_params[i][5]

        vmax = Table_params[i][6]

        # sersic definitions

        redshift = Table_objects[i][1]

        r_e = Table_objects[i][16]

        sersic_pa = Table_objects[i][17]

        v_field = vel_field(vel_field_name, xcen, ycen)

        dim_x = v_field.xpix

        dim_y = v_field.ypix

        a_r = np.sqrt((np.cos(inc) * np.cos(inc)) * (1 - (0.2**2)) + 0.2 ** 2)

        # construct the sersic light profile which will accompany the
        # model velocity data for the beam smearing

        sersic_field = psf.sersic_2d_astropy(dim_x=dim_y,
                                             dim_y=dim_x,
                                             rt=r_e,
                                             n=1.0,
                                             a_r=a_r,
                                             pa=sersic_pa,
                                             xcen=xcen,
                                             ycen=ycen,
                                             sersic_factor=sersic_factor)

        # Now Construct the blurred cube

        theta = [pa, rt, vmax]

        vel_list = v_field.compute_model_grid_for_chi_squared(theta,
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
                                                               sersic_field,
                                                               smear=True)

        mod_vel, mod_vel_blurred, mod_sig_blurred = vel_list

        # also make masked versions of the model grids

        mask_array = np.empty(shape=(dim_x, dim_y))

        for i in range(0, dim_x):

            for j in range(0, dim_y):

                if np.isnan(vel_data[i][j]):

                    mask_array[i][j] = np.nan

                else:

                    mask_array[i][j] = 1.0

        # take product of model and mask_array to return new data

        mod_vel_masked = mod_vel * mask_array

        mod_vel_blurred_masked = mod_vel_blurred * mask_array

        mod_sig_blurred_masked = mod_sig_blurred * mask_array

        # and compute the intrinsic sigma each time by subtracting
        # the beam smeared model linearly and the sky in quadrature

        intrinsic_sigma = np.sqrt((sigma_data - mod_sig_blurred)**2 - sigma_sky_data**2)

        # and plot the results to interpret

        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        rc('font', weight='bold')
        rc('text', usetex=True)
        rc('axes', linewidth=2)

        # extract the raw data and model data along the dynamical pa

        # velocities 
        one_d_data_vel, x_arcsec = rt_pa.extract(0.4,
                                                 0.6,
                                                 pa,
                                                 vel_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        one_d_model_vel, x_arcsec = rt_pa.extract(0.4,
                                                  0.6,
                                                  pa,
                                                  mod_vel,
                                                  xcen,
                                                  ycen,
                                                  pix_scale)

        one_d_model_vel_blurred, x_arcsec = rt_pa.extract(0.4,
                                                          0.6,
                                                          pa,
                                                          mod_vel_blurred,
                                                          xcen,
                                                          ycen,
                                                          pix_scale)

        # sigmas
        one_d_data_sig, x_arcsec = rt_pa.extract(0.4,
                                                 0.6,
                                                 pa,
                                                 sigma_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        one_d_model_sig_blurred, x_arcsec = rt_pa.extract(0.4,
                                                          0.6,
                                                          pa,
                                                          mod_sig_blurred,
                                                          xcen,
                                                          ycen,
                                                          pix_scale)

        one_d_intrinsic_sig, x_arcsec = rt_pa.extract(0.4,
                                                      0.6,
                                                      pa,
                                                      intrinsic_sigma,
                                                      xcen,
                                                      ycen,
                                                      pix_scale)

        one_d_sky_sig, x_arcsec = rt_pa.extract(0.4,
                                                0.6,
                                                pa,
                                                intrinsic_sigma,
                                                xcen,
                                                ycen,
                                                pix_scale)

        # first a grid of the velocities and the models
        # both smeared and unsmeared
        cmap = plt.cm.jet
        cmap.set_bad('black', 1.)

#        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#        im = ax.imshow(mod_vel_blurred, interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(ax)
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)        
#        plt.show()
#        plt.close('all')
        # CONSTRUCT THE FIGURE
        fig, ax = plt.subplots(2, 4, figsize=(16, 9))
        # DATA VELOCITY
        vel_min, vel_max = np.nanpercentile(vel_data,
                                            [5.0, 95.0])
        im = ax[0][0].imshow(vel_data,
                             cmap=cmap,
                             interpolation='nearest',
                             vmin=vel_min,
                             vmax=vel_max)
        ax[0][0].tick_params(axis='x',
                          labelbottom='off')
        ax[0][0].tick_params(axis='y',
                          labelleft='off')
        # add colourbar to each plot
        divider = make_axes_locatable(ax[0][0])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)
        # UNSMEARED MODEL VELOCITY
        vel_min, vel_max = np.nanpercentile(mod_vel_masked,
                                            [5.0, 95.0])
        im = ax[0][1].imshow(mod_vel_masked,
                             cmap=cmap,
                             interpolation='nearest',
                             vmin=vel_min,
                             vmax=vel_max)
        ax[0][1].tick_params(axis='x',
                          labelbottom='off')
        ax[0][1].tick_params(axis='y',
                          labelleft='off')
        # add colourbar to each plot
        divider = make_axes_locatable(ax[0][1])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)
        # SMEARED MODEL VELOCITY
        vel_min, vel_max = np.nanpercentile(mod_vel_blurred_masked,
                                            [5.0, 95.0])
        im = ax[0][2].imshow(mod_vel_blurred_masked,
                             cmap=cmap,
                             interpolation='nearest',
                             vmin=vel_min,
                             vmax=vel_max)
        ax[0][2].tick_params(axis='x',
                          labelbottom='off')
        ax[0][2].tick_params(axis='y',
                          labelleft='off')
        # add colourbar to each plot
        divider = make_axes_locatable(ax[0][2])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)
        # VELOCITY RESIDUALS
        smeared_vel_residual = vel_data - mod_vel_blurred_masked
        one_d_vel_res, x_res = rt_pa.extract(0.4,
                                                0.6,
                                                pa,
                                                smeared_vel_residual,
                                                xcen,
                                                ycen,
                                                pix_scale)
        vel_min, vel_max = np.nanpercentile(smeared_vel_residual,
                                            [5.0, 95.0])
        im = ax[0][3].imshow(smeared_vel_residual,
                             cmap=cmap,
                             interpolation='nearest',
                             vmin=vel_min,
                             vmax=vel_max)
        ax[0][3].tick_params(axis='x',
                          labelbottom='off')
        ax[0][3].tick_params(axis='y',
                          labelleft='off')
        # add colourbar to each plot
        divider = make_axes_locatable(ax[0][3])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)
        # SECOND ROW - SIGMA PLOTS
        # DATA SIGMA
        vel_min, vel_max = np.nanpercentile(sigma_data,
                                            [5.0, 95.0])
        im = ax[1][0].imshow(sigma_data,
                             cmap=cmap,
                             interpolation='nearest',
                             vmin=vel_min,
                             vmax=vel_max)
        ax[1][0].tick_params(axis='x',
                          labelbottom='off')
        ax[1][0].tick_params(axis='y',
                          labelleft='off')
        # add colourbar to each plot
        divider = make_axes_locatable(ax[1][0])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)        
        # SIGMA MODEL
        sigma_full_model = np.sqrt((mod_sig_blurred_masked + sigma)**2 + sigma_sky_data**2)
        one_d_sig_full_model, x_res = rt_pa.extract(0.4,
                                                0.6,
                                                pa,
                                                sigma_full_model,
                                                xcen,
                                                ycen,
                                                pix_scale)
        vel_min, vel_max = np.nanpercentile(sigma_full_model,
                                            [5.0, 95.0])
        im = ax[1][1].imshow(sigma_full_model,
                             cmap=cmap,
                             interpolation='nearest',
                             vmin=vel_min,
                             vmax=vel_max)
        ax[1][1].tick_params(axis='x',
                          labelbottom='off')
        ax[1][1].tick_params(axis='y',
                          labelleft='off')
        # add colourbar to each plot
        divider = make_axes_locatable(ax[1][1])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new) 
        # SIGMA INTRINSIC
        vel_min, vel_max = np.nanpercentile(intrinsic_sigma,
                                            [5.0, 95.0])
        im = ax[1][2].imshow(intrinsic_sigma,
                             cmap=cmap,
                             interpolation='nearest',
                             vmin=vel_min,
                             vmax=vel_max)
        ax[1][2].tick_params(axis='x',
                          labelbottom='off')
        ax[1][2].tick_params(axis='y',
                          labelleft='off')
        # add colourbar to each plot
        divider = make_axes_locatable(ax[1][2])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)
        # SIGMA RESIDUALS
        sigma_residuals = intrinsic_sigma - sigma_full_model
        one_d_sig_res, x_res = rt_pa.extract(0.4,
                                                0.6,
                                                pa,
                                                sigma_residuals,
                                                xcen,
                                                ycen,
                                                pix_scale)
        vel_min, vel_max = np.nanpercentile(sigma_residuals,
                                            [5.0, 95.0])
        im = ax[1][3].imshow(sigma_residuals,
                             cmap=cmap,
                             interpolation='nearest',
                             vmin=vel_min,
                             vmax=vel_max)
        ax[1][3].tick_params(axis='x',
                          labelbottom='off')
        ax[1][3].tick_params(axis='y',
                          labelleft='off')
        # add colourbar to each plot
        divider = make_axes_locatable(ax[1][3])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)
        fig.tight_layout()
        plt.show()
        plt.close('all')

        # ONE DIMENSIONAL PLOTS (EVENTUALLY INCLUDING ERRORS)
        # VELOCITY IN ONE DIMENSION
        fig, ax = plt.subplots(1, 1, figsize=(8,8))
        ax.set_ylabel(r'V$_{c}$[kms$^{-1}$]',
                          fontsize=24,
                          fontweight='bold')
        ax.set_xlabel(r'r [arcsec]',
                          fontsize=24,
                          fontweight='bold')
        # tick parameters 
        ax.tick_params(axis='both',
                           which='major',
                           labelsize=22,
                           length=12,
                           width=2)
        ax.tick_params(axis='both',
                           which='minor',
                           labelsize=22,
                           length=5,
                           width=1)
        ax.scatter(x_arcsec, one_d_model_vel, marker='o', s=75, color='red')
        ax.plot(x_arcsec, one_d_model_vel)
        ax.scatter(x_arcsec, one_d_model_vel_blurred, marker='o', s=75, color='olive')
        ax.plot(x_arcsec, one_d_model_vel_blurred, color='orange')
        ax.scatter(x_arcsec, one_d_data_vel, marker='+', s=75, color='black')
        ax.plot(x_arcsec, one_d_data_vel)
        ax.scatter(x_arcsec, one_d_vel_res, marker='^', s=75, color='purple')
        ax.plot(x_arcsec, one_d_vel_res)
        fig.tight_layout()
        ax.minorticks_on()
        plt.show()
        plt.close('all')

        # SIGMA IN ONE DIMENSION
        fig, ax = plt.subplots(1, 1, figsize=(8,8))
        ax.set_ylabel(r'$\sigma$[kms$^{-1}$]',
                          fontsize=24,
                          fontweight='bold')
        ax.set_xlabel(r'r [arcsec]',
                          fontsize=24,
                          fontweight='bold')
        # tick parameters 
        ax.tick_params(axis='both',
                           which='major',
                           labelsize=22,
                           length=12,
                           width=2)
        ax.tick_params(axis='both',
                           which='minor',
                           labelsize=22,
                           length=5,
                           width=1)
        ax.scatter(x_arcsec, one_d_sig_full_model, marker='o', s=75, color='red')
        ax.plot(x_arcsec, one_d_sig_full_model)
        ax.scatter(x_arcsec, one_d_intrinsic_sig, marker='o', s=75, color='olive')
        ax.plot(x_arcsec, one_d_intrinsic_sig, color='orange')
        ax.scatter(x_arcsec, one_d_data_sig, marker='+', s=75, color='black')
        ax.plot(x_arcsec, one_d_data_sig)
        ax.scatter(x_arcsec, one_d_sig_res, marker='^', s=75, color='purple')
        ax.plot(x_arcsec, one_d_sig_res)
        ax.plot(x_arcsec, one_d_sky_sig, color='brown')
        fig.tight_layout()
        ax.minorticks_on()
        plt.show()
        plt.close('all')

infile='/scratch2/oturner/disk1/turner/DATA/goods_isolated_rotators_names.txt'
param_file='/scratch2/oturner/disk1/turner/DATA/new_comb_calibrated/grid_method_parameters_new.txt'
multi_model_comparison(infile=infile,
                       param_file=param_file,
                       seeing=0.5)