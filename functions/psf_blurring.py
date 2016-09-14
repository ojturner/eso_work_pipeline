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

# Create a gaussian function for use with lmfit
def gaussian(x1,
             x2,
             xcen,
             ycen,
             width):
    """
    Def: Return a two dimensional gaussian function
    """

    # make sure we have floating point values
    width = float(width)

    norm = 1.0 / (2 * np.pi * width**2)
    num = (xcen - x1)**2 + (ycen - x2)**2
    den = 2 * width**2

    # Specify the gaussian function here
    func = norm * np.exp(-1.0*num / den)

    return func

def gauss2dMod():

    mod = Model(gaussian,
                independent_vars=['x1', 'x2'],
                param_names=['xcen',
                             'ycen',
                             'width'],
                missing='drop')

    # print mod.independent_vars
    # print mod.param_names

    return mod

def sersic_2d(x1,
              x2,
              n,
              xcen,
              ycen):

    """
    Def:
    2 dimensional sersic function, in analogy to the 2d gaussian
    function above.

    Input:
            x1,x2 - dependent parameters (x, y)
            n - the sersic index
            xcen - center point in x direction
            ycen - center point in y direction

    Output:
            return the function
    """

    n = float(n)

    # first set the distance from the center

    r = np.sqrt((xcen - x1)**2 + (ycen - x2)**2)

    # and define the function

    func = np.exp(-r**(1.0/n))

    return func

def sersic_2d_mod():

    """
    Def: create lmfit model of 2d sersic profile
    """

    mod = Model(sersic_2d,
                independent_vars=['x1', 'x2'],
                param_names=['n',
                             'xcen',
                             'ycen'],
                missing='drop')

    return mod

def sersic_grid(dim_x,
                dim_y,
                n,
                xcen,
                ycen,
                sersic_factor):

    """
    Def:
    Create a grid of dimensions dim_x, dim_y containing the
    normalised sersic profile of a galaxy

    Input:
    dim_x - dimensions of the grid in rows
    dim_y - dimensions of the grid in columns
    n - sersic index
    res_factor - factor by which to increase resolution
    xcen - center of sersic in x direction
    ycen - center of sersic in y direction

    Output:
    normalised sersic profile centred at chosen location, which can be
    outside the dimensions of the grid if desired, although that would
    be totally contrary to the point of doing this
    """

    # set up the sersic model with the chosen parameters

    s_mod = sersic_2d_mod()

    # set up the grid over which to evaluate the gaussian

    xbin = np.arange(0, dim_x, 1 / float(sersic_factor))

    ybin = np.arange(0, dim_y, 1 / float(sersic_factor))

    ybin, xbin = np.meshgrid(ybin,
                             xbin)

    xbin = np.ravel(xbin)

    ybin = np.ravel(ybin)

    s_mod_eval_1d = s_mod.eval(x1=xbin,
                               x2=ybin,
                               n=n,
                               xcen=xcen,
                               ycen=ycen)

    s_mod_eval = np.reshape(s_mod_eval_1d, (dim_x * sersic_factor,
                                            dim_y * sersic_factor))


#    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#    im = ax.imshow(bin_by_factor(s_mod_eval,sersic_factor),
#                   cmap=plt.get_cmap('jet'),
#                   interpolation='nearest')
#    # add colourbar to each plot
#    divider = make_axes_locatable(ax)
#    cax_new = divider.append_axes('right', size='10%', pad=0.05)
#    plt.colorbar(im, cax=cax_new)
#    # set the title
#    ax.set_title('Sersic model')
#    plt.show()
#    plt.close('all')
    
    return bin_by_factor(s_mod_eval,sersic_factor)

def psf_grid(dim_x,
             dim_y,
             xcen,
             ycen,
             seeing,
             pix_scale,
             psf_factor):

    """
    Def:
    Create a grid of dimensions dim_x, dim_y containing the
    normalised seeing profile of a given atmosphere,

    Input:
    dim_x - dimensions of the grid in rows
    dim_y - dimensions of the grid in columns
    height - amplitude of the gaussian
    xcen - center of gaussian in x direction
    ycen - center of gaussian in y direction
    seeing - given in arcseconds
    pix_scale - given in arcseconds, dimension of individual pixel

    Output:
    normalised seeing profile centred at chosen location, which can be
    outside the dimensions of the grid if desired, although that would
    be totally contrary to the point of doing this
    """

    # set up the gaussian model with the chosen parameters

    g_mod = gauss2dMod()

    # set up the grid over which to evaluate the gaussian

    xbin = np.arange(0, dim_x, 1 / float(psf_factor))

    ybin = np.arange(0, dim_y, 1 / float(psf_factor))

    ybin, xbin = np.meshgrid(ybin, xbin)

    xbin = np.ravel(xbin)

    ybin = np.ravel(ybin)

    # the width is determined from the seeing and the pixel scale

    width = (seeing / 2.355) / pix_scale

    g_mod_eval_1d = g_mod.eval(x1=xbin,
                               x2=ybin,
                               xcen=xcen,
                               ycen=ycen,
                               width=width)

    g_mod_eval = np.reshape(g_mod_eval_1d, (dim_x * float(psf_factor),
                                            dim_y * float(psf_factor)))

#    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#    im = ax.imshow(bin_by_factor(g_mod_eval,psf_factor),
#                   cmap=plt.get_cmap('jet'),
#                   interpolation='nearest')
#    # add colourbar to each plot
#    divider = make_axes_locatable(ax)
#    cax_new = divider.append_axes('right', size='10%', pad=0.05)
#    plt.colorbar(im, cax=cax_new)
#    # set the title
#    ax.set_title('Seeing model')
#    plt.show()
#    plt.close('all')
    
    n_hdu = fits.PrimaryHDU(g_mod_eval)

    n_hdu.writeto('/home/oturner/disk1/turner/DATA/Victoria_galfit/n_band_outputs_flatfield/psf_0.2.fits',
                  clobber=True)

    if float(psf_factor) != 1.0:

        # print 'not equal 1' 
        return bin_by_factor(g_mod_eval, psf_factor)

    else:

        return g_mod_eval

def blur_by_psf(data,
                seeing,
                pix_scale,
                psf_factor):

    """
    Def:
    Take simulated data (intrinsic to an object) and simulate the
    effect of passing through an atmosphere which is modelled by
    a particular PSF.

    Input:
            data - 2d grid of data points, every spatial location will
                    be given a PSF profile centred there

            seeing - the seeing value with which to smear the object
            pix_scale - pixel scale of the observations

    Output:
            blurred 2d grid of flux values
    """

    dim_x = data.shape[0]
    dim_y = data.shape[1]

    final_flux = []

    for i in range(dim_x):

        for j in range(dim_y):

            temp_flux_grid = np.full((dim_x,dim_y), data[i,j])

            seeing_profile = psf_grid(dim_x,
                                      dim_y,
                                      i,
                                      j,
                                      seeing,
                                      pix_scale,
                                      psf_factor)

            final_flux.append(temp_flux_grid * seeing_profile)

    final_flux = np.nansum(final_flux, axis=0)

#    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#    im = ax.imshow(final_flux,
#                   cmap=plt.get_cmap('jet'),
#                   interpolation='nearest')
#    # add colourbar to each plot
#    divider = make_axes_locatable(ax)
#    cax_new = divider.append_axes('right', size='10%', pad=0.05)
#    plt.colorbar(im, cax=cax_new)
#    # set the title
#    ax.set_title('Sersic model')
#    plt.show()
#    plt.close('all')

    return final_flux

def construct_shifted_cube(vel_data,
                           redshift,
                           sigma,
                           wave_array):

    """
    Def:
    Evaluate a gaussian function with the wavelength array as the
    argument and centre computed using the known redshifted centre of
    [OIII] offset by the velocity value.
    Return a datacube of dimensions (wave_array, vel_data.shape) with the
    shifted [OIII] line profiles (which are at this point all the same
    flux values).
    The idea is that these values will be convolved with a gaussian profile
    and a sersic profile, assuming that the gaussian flux fall off follows
    exactly the sersic fall off (Physics?) to be able to start recovering
    the effect of beam smearing on the simulated data.

    Input:
            vel_data - 2d velocity values representing a galaxy disk
            redshift - the redshift of the galaxy in question
            sigma - the starting width of the line (in kms-1)
            wave_array - 1d array of wavelength points over which 
                        the gaussians shall be evaluated.

    Output:
            3d cube containing in each spaxel the shifted wavelength arrays
    """

    # what is the actual central wavelength value for [OIII]

    central_l = (1 + redshift) * 0.500824

    c = 2.99792458E5

    # set up the gaussian model

    g_mod = GaussianModel()

    # initialise the cube array

    cube_array = []

    # loop over the velocity data dimensions
    # and evaluate the gaussians

    for i in range(vel_data.shape[0]):

        for j in range(vel_data.shape[1]):

            # use the velocity value to compute the new observed
            # wavelength (which will become the gaussian centre)

            vel_value = vel_data[i, j]

            if np.isnan(vel_value):

                cube_array.append(np.repeat(np.nan, len(wave_array)))

            else:

                l_o = central_l + central_l * (vel_value / c)

                # convert the given gaussian width in kms-1 into a
                # wavelength width

                sig_l = (central_l * sigma) / c

                # append the evaluated gaussian to the cube_array

                cube_array.append(g_mod.eval(x=wave_array,
                                             amplitude=1.0,
                                             sigma=sig_l,
                                             center=l_o))

    # in theory that's it - now reshape the resultant array and
    # return that data cube
    cube_array = np.array(cube_array)

    cube = np.reshape(cube_array.T, (len(wave_array),
                                     vel_data.shape[0],
                                     vel_data.shape[1]))

    return cube

def cube_blur(vel_data,
              redshift,
              wave_array,
              xcen,
              ycen,
              seeing,
              pix_scale,
              psf_factor,
              sersic_factor,
              sigma=60,
              sersic_n=2.0):

    """
    Def: constructs a 3D cube from a 2D velocity field and passes
    that through the atmosphere.
    """

    # define speed of light and central wavelength

    central_l = (1 + redshift) * 0.500824

    # define a truncated wavelength array 10 points either
    # side of the central wavelength point

    c_indx = np.argmin(abs(central_l - wave_array))

    wave_array = wave_array[c_indx - 10:c_indx + 10]

    c = 2.99792458E5

    # set up some basic quantities

    dim_x = vel_data.shape[0]

    dim_y = vel_data.shape[1]

    # first feed in the velocity data in order to define the shifted cube

    cube = construct_shifted_cube(vel_data,
                                  redshift,
                                  sigma,
                                  wave_array)

    # now do something very similar to psf blur but in 3 dimensions 
    # this involves evaluating the sersic profile only once in the
    # center of the galaxy and evaluating the PSF profile each time
    # in the spaxel under consideration

    sersic_profile = sersic_grid(dim_x,
                                 dim_y,
                                 sersic_n,
                                 xcen,
                                 ycen,
                                 sersic_factor)

    # initialise array to house all of the shifted cubes

    cube_list = []

    for i in range(dim_x):

        for j in range(dim_y):

            # build the seeing profile at the current position

            seeing_profile = psf_grid(dim_x,
                                      dim_y,
                                      i,
                                      j,
                                      seeing,
                                      pix_scale,
                                      psf_factor)

            # convolve this with the sersic profile to get the
            # reduction factor

            grid_factors = seeing_profile * sersic_profile

            # construct temporary cube, which is an object of the same
            # dimensions as cube, but with every spaxel filled with the
            # values at spaxel[i, j] - this was my mistake

            temp_cube = []

            for entry in cube[:, i, j]:

                temp_cube.append(np.full((dim_x, dim_y), entry))

            temp_cube = np.array(temp_cube)

            # append grid_factors multiplied by the temp cube to the cube list

            cube_list.append(grid_factors * temp_cube)

    # the final cube is then the sum along the 0th axis
    # of all the other averaged cubes
    cube_blurred = np.sum(cube_list, axis=0)

    # Now do gaussian fitting to every spaxel to recover the shifted
    # velocity values and save to a new 2D surface which is then saved

    shifted_velocities = np.empty(shape=(dim_x, dim_y))

    #shifted_sigma = np.empty(shape=(dim_x, dim_y))

    # fit a gaussian to every profile in the gauss_array_cube
    # and store in the smear_array

    for g in range(dim_x):

        for h in range(dim_y):

            gauss_values, cov = gauss_fit(wave_array,
                                          cube_blurred[:, g, h])

            shifted_velocities[g, h] = gauss_values['center']
            #shifted_sigma[g, h] = gauss_values['sigma']

    # convert back to a kilometres per second value

    shifted_velocities = ((shifted_velocities - central_l) / central_l) * c
    #shifted_sigma = (((shifted_sigma) / central_l) * c) - 60

#    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#    im = ax.imshow(vel_data,
#                   cmap=plt.get_cmap('jet'),
#                   interpolation='nearest')
#    # add colourbar to each plot
#    divider = make_axes_locatable(ax)
#    cax_new = divider.append_axes('right', size='10%', pad=0.05)
#    plt.colorbar(im, cax=cax_new)
#    # set the title
#    ax.set_title('Model Velocities')
#    plt.show()
#    fig.savefig('/disk1/turner/DATA/experiments/non_smeared_sigma%s_sersic%s_seeing%s_psf%s_sersic%s.png' % (sigma,
#                                                                                                             sersic_n,
#                                                                                                             seeing,
#                                                                                                             psf_factor,
#                                                                                                             sersic_factor))
#    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(shifted_velocities,
                   cmap=plt.get_cmap('jet'),
                   interpolation='nearest')
    # add colourbar to each plot
    divider = make_axes_locatable(ax)
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)
    # set the title
    ax.set_title('Shifted Velocities')
    plt.show()
    fig.savefig('/disk1/turner/DATA/experiments/smeared_sigma%s_sersic%s_seeing%s_psf%s_sersic%s.png' % (sigma,
                                                                                                             sersic_n,
                                                                                                             seeing,
                                                                                                             psf_factor,
                                                                                                             sersic_factor))
#    plt.close('all')
#    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#    im = ax.imshow(shifted_sigma,
#                   cmap=plt.get_cmap('jet'),
#                   interpolation='nearest')
#    # add colourbar to each plot
#    divider = make_axes_locatable(ax)
#    cax_new = divider.append_axes('right', size='10%', pad=0.05)
#    plt.colorbar(im, cax=cax_new)
#    # set the title
#    ax.set_title('Shifted Dispersions')
#    plt.show()
#    fig.savefig('/disk1/turner/DATA/experiments/smeared_dispersions_sigma%s_sersic%s_seeing%s_psf%s_sersic%s.png' % (sigma,
#                                                                                                             sersic_n,
#                                                                                                             seeing,
#                                                                                                             psf_factor,
#                                                                                                             sersic_factor))
#    plt.close('all')

    return shifted_velocities

def gauss_fit(fit_wl,
              fit_flux):

    """
    Def:
    Performs simple gaussian fit, guessing initial parameters from the data
    and given input wavelength and input flux values

    Input:
            fit_wl - wavelength of spectrum to fit
            fit_flux - flux of spectrum to fitsWavelength

    Output:
            fit_params - dictionary containing the best fit parameters
                        for each of the spectra
    """

    # construct gaussian model using lmfit

    gmod = GaussianModel()

    # set the initial parameter values

    pars = gmod.guess(fit_flux, x=fit_wl)

    # perform the fit
    out = gmod.fit(fit_flux, pars, x=fit_wl)

    # print the fit report
    # print out.fit_report()

    # plot to make sure things are working
#    fig, ax = plt.subplots(figsize=(14, 6))
#    ax.plot(fit_wl, fit_flux, color='blue')
#    ax.plot(fit_wl, out.best_fit, color='red')
#    plt.show()
#    plt.close('all')

    return out.best_values, out.covar

def compute_velocity_smear(vel_data,
                           sersic_n,
                           xcen,
                           ycen,
                           seeing,
                           pix_scale,
                           psf_factor,
                           sersic_factor):

    """
    Def:
    Compute the effects of beam smearing on all the pixels given
    the shifted_cube computed from the model/observed velocity field.
    First construct a sersic profile centered at the chosen location
    and convolve with the seeing for each spaxel.

    Input:
            vel_data - 2D velocity field
            sersic_n - sersic index of the light profile
            xcen - sersic center in the x-direction
            ycen - sersic center in the y-direction
            seeing - the atmospheric seeing conditions
            pix_scale - the pixel scale of the observations
            res_factor - the factor boost to resolution measurements

    Output:
            Not sure yet
    """

    # initiate the sersic profile

    dim_x = vel_data.shape[0]

    dim_y = vel_data.shape[1]

    sersic_2d = sersic_grid(dim_x,
                            dim_y,
                            sersic_n,
                            xcen,
                            ycen,
                            sersic_factor)

#    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#    im = ax.imshow(vel_data,
#                   cmap=plt.get_cmap('jet'),
#                   interpolation='nearest')
#    # add colourbar to each plot
#    divider = make_axes_locatable(ax)
#    cax_new = divider.append_axes('right', size='10%', pad=0.05)
#    plt.colorbar(im, cax=cax_new)
#    # set the title
#    ax.set_title('Sersic model')
#    plt.show()
#    plt.close('all')

    # initialise the overall smear array
    final_array = []

    # loop around the velocity data and blur
    # taking into account the intrinsic light profile

    for i in range(dim_x):
        # print i

        for j in range(dim_y):
            # print j

            # initiate list to hold the contributions to each spaxel
            temp_list = []

            # if there is a nan value in the vel data - not interested
            if np.isnan(vel_data[i, j]):

                final_array.append(np.nan)

            # else compute the effects of smearing in the light from
            # everywhere else

            else:

                seeing_profile = psf_grid(dim_x,
                                          dim_y,
                                          i,
                                          j,
                                          seeing,
                                          pix_scale,
                                          psf_factor)

                # compute the factor with which to ammend the
                # cube gaussian array

                factor = sersic_2d[i, j] * seeing_profile[i, j]

                # and append this to the temp_list as the starting point

                temp_list.append(factor * vel_data[i, j])

                # now for the tricky part - computing the contributions
                # from all other spaxels (smeared by the PSF)
                # obviously less contribution as you get further away
                # (translating to a decrease in this factor parameter)
                # effect greatest when the flux and velocity centers are
                # co-indicent

                for new_i in range(dim_x):

                    for new_j in range(dim_y):

                        # if the new loop values are not equivalent to the
                        # spaxel that we're trying to figure out
                        # execute this block of code

                        if not((new_i == i) and (new_j == j)):

                            # seeing profile initiated at new spatial loc

                            seeing_profile = psf_grid(dim_x,
                                                      dim_y,
                                                      new_i,
                                                      new_j,
                                                      seeing,
                                                      pix_scale,
                                                      psf_factor)

                            # factor evaluated at old spatial location in
                            # seeing profile but new spatial location in
                            # sersic profile - because we are computing the
                            # effect of blurring sersic new by psf at
                            # the initial spatial location

                            factor = sersic_2d[new_i, new_j] * \
                                seeing_profile[i, j]

                            # then append to the temp_list the factor mult
                            # by the cube_array at the new spatial location

                            temp_list.append(factor *
                                             vel_data[new_i, new_j])

                # append to the gauss array the summed contributions to
                # that spaxel from all others, the final line profile
                # for that spaxel

                # remember nansum because some of the blurred values
                # will be nan (actually most)

                final_array.append(np.nansum(temp_list))

                # loop back and do that for every spaxel

    # now reshape the gauss_array in the same way as we did above
    final_array = np.array(final_array)

    print final_array.shape
    print dim_x, dim_y

    final_array = final_array.reshape(dim_x,
                                      dim_y)

#    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#    im = ax.imshow(final_array,
#                   cmap=plt.get_cmap('jet'),
#                   interpolation='nearest')
#    # add colourbar to each plot
#    divider = make_axes_locatable(ax)
#    cax_new = divider.append_axes('right', size='10%', pad=0.05)
#    plt.colorbar(im, cax=cax_new)
#    # set the title
#    ax.set_title('Sersic model')
#    plt.show()
#    plt.close('all')

    print np.nansum(abs(vel_data))
    print np.nansum(abs(final_array))

    return final_array

def bin_by_factor(data,
                  res_factor):

    """
    Def: Bin up a 2D grid of points by a factor of ten
    for using with the higher resolution velocity map
    and beam smearing corrections

    Inputs:
           res_factor - make it an integer
           data - the data for binning
    """

    res_factor = float(res_factor)
    # define the x dimension of the data
    x_dim = data.shape[0] / res_factor
    y_dim = data.shape[1] / res_factor

    data_view = data.reshape(x_dim, res_factor, y_dim, res_factor)

    final_data = data_view.mean(axis=3).mean(axis=1)

    return final_data


psf_grid(35,
         37,
         17,
         18,
         0.2,
         0.1,
         1)