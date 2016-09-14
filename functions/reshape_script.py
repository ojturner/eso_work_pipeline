# for reshaping and moving around
# the galfit images
# want to make this as painfree as possible
# O.J.Turner 2016

# import the relevant modules
import scipy.optimize as opt
import pylab as pyplt
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from numpy import poly1d
import scipy
import numpy.ma as ma
from lmfit.models import GaussianModel
from lmfit import Model
from astropy.io import fits
from astropy.modeling import models, fitting
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import *
from numpy import *

def reshape_galfit(bulge,
                   disk):

    cube_dir = '/disk1/turner/DATA/new_comb_calibrated/goods_p2_0.8_10_better/Science/'

    galfit_dir = '/disk1/turner/DATA/Victoria_remasked/'

    sig_dir = '/disk1/turner/DATA/Victoria_remasked/rms_maps/'

    cube_name = 'combine_sci_reconstructed_' + raw_input('Cube Name is: ') + '.fits'

    load_cube = cube_dir + cube_name

    save_name = cube_dir + cube_name[:-5] + '_galfit.fits'

    galfit_number = raw_input('Galfit Number is: ')

    if bulge:

        galfit_name = galfit_number + '_bulge_output.fits'

    if disk:

        galfit_name = galfit_number + '_disk_output.fits'

    if (not(bulge) and not(disk)):

        galfit_name = galfit_number + '_output_test.fits'

    sigma_name = galfit_number + '_sigmastamp.fits'

    load_galfit = galfit_dir + galfit_name

    load_sigma = sig_dir + sigma_name

    load_flux = load_cube[:-5] + '_flux_field.fits'

    ## load in the data - want to load in the hst image in update mode

    table_galfit = fits.open(load_galfit, mode='update')

    galfit_stamp = table_galfit[1].data

    table_sigma = fits.open(load_sigma)

    sigma_map = table_sigma[0].data

    table_flux = fits.open(load_flux)

    flux_stamp = table_flux[0].data

    ## plots for determining center 

    ## object

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.imshow(flux_stamp)

    ax.minorticks_on()

    ax.grid(b=True, which='major', color='b', linestyle='-')

    ax.grid(b=True, which='minor', color='r', linestyle='--')

    plt.show()

    plt.close('all')

    x_obj = int(raw_input('Vertical Object centre is: '))

    y_obj = int(raw_input('Horizontal Object centre is: '))

    ## galfit

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.imshow(galfit_stamp)

    ax.minorticks_on()

    ax.grid(b=True, which='major', color='b', linestyle='-')

    ax.grid(b=True, which='minor', color='r', linestyle='--')

    plt.show()

    plt.close('all')

    x_galfit = int(raw_input('Vertical galfit centre is: '))

    y_galfit = int(raw_input('Horizontal galfit centre is: '))

    x_dim = flux_stamp.shape[0]

    y_dim = flux_stamp.shape[1]

    x_dim_new = int(np.round(x_dim / 0.6))

    y_dim_new = int(np.round(y_dim / 0.6))

    print x_dim_new, y_dim_new

    start_x = int(np.round(x_galfit - (x_obj / 0.6)))

    end_x = start_x + x_dim_new

    start_y = int(np.round(y_galfit - (y_obj / 0.6)))

    end_y = start_y + y_dim_new

    print start_x, end_x, start_y, end_y

    ## now start doing some reformatting of the data

    table_galfit[1].data = table_galfit[1].data[start_x:end_x, start_y:end_y] / table_sigma[0].data[start_x:end_x, start_y:end_y]

    table_galfit[2].data = table_galfit[2].data[start_x:end_x, start_y:end_y] / table_sigma[0].data[start_x:end_x, start_y:end_y]

    table_galfit[3].data = table_galfit[3].data[start_x:end_x, start_y:end_y] / table_sigma[0].data[start_x:end_x, start_y:end_y]

    ## Everything should now be updated and can be flushed to save the original galfit table

    table_galfit.flush()

    table_galfit.close()

    ## Now want to rename the galfit object into the proper directory

    os.system('mv %s %s' % (load_galfit, save_name))

reshape_galfit(False, False)

