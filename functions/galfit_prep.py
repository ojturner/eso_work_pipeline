# Function to prepare the inputs for galfit
# for Victoria. Not sure if I need a separate 
# PSF for every galaxy? These have different 
# pixel sizes, may complicate things



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

# add the class file to the PYTHONPATH
sys.path.append('/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/Class')

# add the functions folder to the PYTHONPATH
sys.path.append('/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/functions')

import cube_background_subtract as b_s



from cubeClass import cubeOps

def galfit_input(infile,
                 redshift,
                 name,
                 sv_dir):
    
    """
    Def:
    Given an input datacube - construct the narrowband OIII image and
    the associated rms image and save as separate fits files. Save in the 
    save directory with the given name.
    """

    cube = cubeOps(infile)

    table = cube.Table

    # find the velocity profile data for masking the region which
    # contains the object. This is the same as the object file but
    # truncated and with vel_field.fits on the end

    mask_data_name = infile[:-5] + '_vel_field.fits'

    m_data = fits.open(mask_data_name)[0].data

    cube_data = b_s.back_subtract(infile,
                                  m_data,
                                  redshift)

    noise_cube = table[2].data

    #  OIII wavelength
    central_l = (1 + redshift) * 0.500824

    # OIII peak index
    o_peak = np.argmin(abs(central_l - cube.wave_array))

    # make the narrowband image
    o_nband = np.nanmedian(cube_data[o_peak-2:o_peak+2, :, :], axis=0)

    # make the noise image
    noise_image = np.nanmedian(noise_cube[o_peak-2:o_peak+2, :, :], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(o_nband)
    plt.show()
    plt.close('all')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(noise_image)
    plt.show()
    plt.close('all')

    # save both to new fits files
    o_name = sv_dir + name + '.fits'
    n_name = sv_dir + name + '_rms.fits'

    o_hdu = fits.PrimaryHDU(o_nband)

    o_hdu.writeto(o_name,
                  clobber=True)

    n_hdu = fits.PrimaryHDU(noise_image)

    n_hdu.writeto(n_name,
                  clobber=True)

master = ascii.read('/disk1/turner/DATA/all_names_new.txt')
names = ascii.read('/disk1/turner/Catalogues/goods_s/just_names.txt')

for f, r, n in zip(master['Filename'], master['redshift'], names['name']):

    print 'preparing: %s' % n

    galfit_input(f,
                 r,
                 n,
                 '/disk1/turner/DATA/Victoria_galfit/n_band_outputs_flatfield/')