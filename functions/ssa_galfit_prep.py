#DATE: 9-9-16
#Description: Used for getting the ssa galaxies ready for galfitting.
#Will contain recipes for extracting the galaxy from the relevant fits
#file as a postage stamp, which contains the relevant header info for galfit,
#and for rotating the file so that North is up to match the KMOS observations,
#after this the rotated postage stamp will be saved with a different name.
#Then for running sextractor on the object, not sure exactly how this will
#work in an automated way given that there will definitely be some masking of
#background contaminants required, and possibly a script for creating the
#galfit input and running galfit.

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

def extract_stamp(galaxy_name,
                  extraction_image,
                  ra,
                  dec,
                  region_size,
                  out_dir='/home/oturner/disk1/turner/DATA/GALFIT/SSA_F160W/'):
    """
    Def:
    For a given galaxy, name must be specified, take the ra and dec
    and extract a postage stamp centred on that point of the given
    region size. For galfitting an appropriate size might be something
    like a 6 arcsecond square box. This postage stamp is then save with
    the galaxy name along with stamp.fits - the header information from the
    given file is also saved.
    Exactly the same procedure will be applied to the weight map which is in
    the extension following the drz.fits file.   
    Inputs: galaxy_name - a string specifying which galaxy this is e.g. 'nc47'
            extraction_image - name of the fits file from which the extraction should happen
            ra - the ra of the galaxy, which must be within the boundaries of the extraction image
            dec - the declination of the galaxy which must be within the boundaries of the extraction image
            region size - specified side of box, in arcseconds, for the postage stamp extraction 
    Outputs: galaxy_name_stamp.fits
            galaxy_name_rms.fits 
    Note that this assumes the extraction image is in the classic extension 1 = drz, extension 2 = wht format.
    """

    # first open the extraction_image fits file as a table
    table = fits.open(extraction_image)

    # save the SCI and WHT headers as different things
    sci_header = table[1].header
    sci_data = table[1].data
    sci_name = out_dir + galaxy_name + '_stamp.fits'

    wht_header = table[2].header
    wht_data = table[2].data
    wht_name = out_dir + galaxy_name + '_rms.fits'

    # little trick here - want an rms map so take one over sqrt(wht_data)
    wht_data = 1 / np.sqrt(wht_data)

    # now find the coordinates for extraction in the usual way
    w = WCS(sci_header)

    lon, lat = w.wcs_world2pix(ra, dec, 1)

    lon_scale, lat_scale = autils.proj_plane_pixel_scales(w)

    lon_unit = int(np.round((region_size / (lon_scale * 3600)) / 2.0))

    lat_unit = int(np.round((region_size / (lat_scale * 3600)) / 2.0))

    # and get the extraction stamps for sci and wht, ready to save
    extraction_stamp_sci = sci_data[lat - lat_unit:lat + lat_unit,
                                    lon - lon_unit:lon + lon_unit]

    extraction_stamp_wht = wht_data[lat - lat_unit:lat + lat_unit,
                                    lon - lon_unit:lon + lon_unit]

    # now save these as separate fits stamps

    sci_hdu = fits.PrimaryHDU(header=sci_header, data=extraction_stamp_sci)
    sci_hdu.writeto(sci_name, clobber=True)

    wht_hdu = fits.PrimaryHDU(header=wht_header, data=extraction_stamp_wht)
    wht_hdu.writeto(wht_name, clobber=True)

def rotate_outputs(galaxy_name,
                   search_dir='/home/oturner/disk1/turner/DATA/GALFIT/SSA_F160W/'):

    """
    Def:
    The data I found for these SSA galaxies is unhelpfully rotated wrt the
    KMOS galaxies. So need to apply a shift to put North in the right place
    and then apply galfit so that the position angles are directly comparable.
    Although currently still unclear exactly how to compare these two quantities
    The rotation will be performed using pyraf, which by default saves to a
    different file without a header. So can read the data from that back in
    and then save the rotated stamps for sextractor and galfit.
    Ideally the orientation angle will be updated after the rotation is appl.


    Input:
            galaxy_name - to be able to pick up the stamp and rms files
            search_dir - where to look for the above files.

    Output: 
            stamp_rot.fits
            rms_rot.fits
            in the same directory as the initial files were taken from
    """

    # start by opening the two files again
    sci_name = search_dir + galaxy_name + '_stamp.fits'
    sci_rot_name = search_dir + galaxy_name + '_stamp_rot.fits'

    rms_name = search_dir + galaxy_name + '_rms.fits'
    rms_rot_name = search_dir + galaxy_name + '_rms_rot.fits'

    table_sci = fits.open(sci_name)

    table_rms = fits.open(rms_name)

    # read in the rotation keyword
    rot_angle = table_sci[0].header['ORIENTAT']

    # and apply the rotations
    pyraf.iraf.rotate(sci_name,
                      sci_rot_name,
                      rot_angle,
                      interp='spline3')

    pyraf.iraf.rotate(rms_name,
                      rms_rot_name,
                      rot_angle,
                      interp='spline3')

    # This saves the rms_rot_name in the same directory but
    # without the header. So must read it back in, save with a
    # header and update the orientation angle to read 0

#    sci_rot_data = fits.open(sci_rot_name)[0].data
#    rms_rot_data = fits.open(rms_rot_name)[0].data
#    sci_rot_header = table_sci[0].header
#    rms_rot_header = table_rms[0].header
#    sci_hdu = fits.PrimaryHDU(header=sci_rot_header,
#                              data=sci_rot_data)
#    sci_hdu.writeto(sci_rot_name, clobber=True)
#    rms_hdu = fits.PrimaryHDU(header=rms_rot_header,
#                              data=rms_rot_data)
#    rms_hdu.writeto(rms_rot_name, clobber=True)


extract_stamp('nc47', '/home/oturner/disk1/turner/DATA/IMAGING/HST_SSA_F160W/icl802010_drz.fits', 334.3343208, 0.2921527778, 8)
rotate_outputs('nc47')