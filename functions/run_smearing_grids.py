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

# the purpose of this script is to take the list of galaxy names from the
# isolated rotators sample and calculate model parameters with the
# new grid based approach

# load in the relevant information from the infile

param_names = ['Gal_name',
               'xcen',
               'ycen',
               'inc',
               'position_angle',
               'Rt',
               'Vmax']

param_file = '/scratch2/oturner/disk1/turner/DATA/new_comb_calibrated/ssa_grid_method_parameters_fixed.txt'

if os.path.isfile(param_file):

    os.system('rm %s' % param_file)

# write all of these values to file

with open(param_file, 'w') as f:

    for item in param_names:

        f.write('%s\t' % item)

    # read in the table of cube names
    Table = ascii.read('/scratch2/oturner/disk1/turner/DATA/ssa_isolated_rotators_names.txt')

    # assign variables to the different items in the infile
    for entry in Table:

        cube_name = entry[0]

        print cube_name

        cube = cubeOps(cube_name)

        wave_array = cube.wave_array

        obj_name = entry[0][:-5] + '_vel_field.fits'

        param_file_idv = entry[0][:-5] + '_chi_squared_params.txt'

        redshift = entry[1]

        xcen = entry[10]

        ycen = entry[11]

        inc = entry[12]

        pa = entry[13]

        r_e = entry[16]

        sersic_pa = entry[17]

        v_field = vel_field(obj_name, xcen, ycen)

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
                                             sersic_factor=50)

#        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#        ax.imshow(np.log10(sersic_field))
#        plt.show()
#        plt.close('all')

        vasym, rt = v_field.grid_fixed_inc_fixed_params(inc=inc,
                                                        pa=pa,
                                                        redshift=redshift,
                                                        wave_array=wave_array,
                                                        xcen=xcen,
                                                        ycen=ycen,
                                                        seeing=0.6,
                                                        sersic_n=1,
                                                        sigma=50,
                                                        pix_scale=0.1,
                                                        psf_factor=1,
                                                        sersic_factor=1,
                                                        m_factor=4,
                                                        light_profile=sersic_field,
                                                        smear=True)

        # write to an individual parameter file for reading in later
        if os.path.isfile(param_file_idv):

            os.system('rm %s' % param_file_idv)

        with open(param_file_idv, 'w') as f_2:

            for item in param_names:

                f_2.write('%s\t' % item)

            f_2.write('\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t' % (cube_name,
                                                          xcen,
                                                          ycen,
                                                          inc,
                                                          pa,
                                                          rt,
                                                          vasym))

            f_2.close()



        # write to the overall parameter file
        f.write('\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t' % (cube_name,
                                                    xcen,
                                                    ycen,
                                                    inc,
                                                    pa,
                                                    rt,
                                                    vasym))

    f.close()
