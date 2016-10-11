# create colour images in the goods-s
# region from the images in the given directories
# at the moment will have a blue, green and red
# file which is read in, the coordinates of the 
# given object converted to pixel coordinates 
# the pixel scale digested and a postage stamp colour image
# created. Need to make sure that each of the filters used
# has coverage over the full GOODS-S region

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


def get_filter_data(filter_file,
                    ra,
                    dec):

    """
    Def:
    Load in the filter file and the ra/dec of the object to
    extract a 4'' postage stamp around the object and return the
    data
    """

    w = WCS(filter_file)

    lon, lat = w.wcs_world2pix(ra, dec, 1)

    lon_scale, lat_scale = autils.proj_plane_pixel_scales(w)

    lon_unit = int(np.round((4 / (lon_scale * 3600)) / 2.0))

    lat_unit = int(np.round((4 / (lat_scale * 3600)) / 2.0))

    extraction_stamp = fits.open(filter_file)[0].data[lat - lat_unit:lat + lat_unit,
                                                      lon - lon_unit:lon + lon_unit]

    return extraction_stamp

def save_stamp(extraction_stamp,
               name):

    """
    Def:
    take the extraction stamp from the above method and save
    as a postage stamp with that name. Will be used for the galfit
    inputs.
    """

def make_image(r,
               g,
               b,
               ra,
               dec,
               name, 
               save_dir):

    """
    Def:
    Build a colour image of a galaxy centred on RA and DEC, where
    r, g, b, are the filepaths to the filter_files and name is the name 
    of the galaxy being processed. 
    """
    r_stamp = get_filter_data(r, ra, dec)

    r_stamp = abs(1 / np.nanmax(r_stamp) * r_stamp * 255.9)

    g_stamp = get_filter_data(g, ra, dec)

    g_stamp = abs(1 / np.nanmax(g_stamp) * g_stamp * 255.9)

    b_stamp = get_filter_data(b, ra, dec)

    b_stamp = abs(1 / np.nanmax(b_stamp) * b_stamp * 255.9)

    rgbArray = np.zeros((b_stamp.shape[0],
                         b_stamp.shape[1],
                         3),
                        'uint8')

    rgbArray[..., 0] = r_stamp

    rgbArray[..., 1] = g_stamp

    rgbArray[..., 2] = b_stamp

    img = Image.fromarray(rgbArray)

    save_name = save_dir + name + '.jpg'
    img.save(save_name)



b = '/scratch2/oturner/disk1/turner/DATA/IMAGING/CANDELS_GOODSS_ACS/CANDELS_850/gs_acs_old_f850l_060mas_v2_drz.fits'
g = '/scratch2/oturner/disk1/turner/DATA/IMAGING/CANDELS_GOODSS_WFCAM3/CANDELS_125/gs_all_candels_ers_udf_f125w_060mas_v0.5_drz.fits'
r = '/scratch2/oturner/disk1/turner/DATA/IMAGING/CANDELS_GOODSS_WFCAM3/CANDELS_160/gs_all_candels_ers_udf_f160w_060mas_v0.5_drz.fits'
sv = '/scratch2/oturner/disk1/turner/DATA/IMAGING/GOODS_COLOUR_160/'

table = ascii.read('/scratch2/oturner/disk1/turner/Catalogues/goods_s/goods_p2.txt')

for entry in table:

    print 'Making Colour Image for %s' % entry['name']

    make_image(r, g, b, entry['ra'], entry['dec'], entry['name'], sv)

