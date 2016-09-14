from astropy.io import fits
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import copy


def flatfield(cube,
              m_data,
              redshift):
    
    """
    Def:
    Trying to extract continuum from the KMOS galaxies.
    First plot the continuum without flatfielding, then 
    subtract the median flux from every row and column of
    each slice of the masked cube to flatfield correctly
    and plot the result. 

    Input:
            cube - The cube object to perform operation on
            m_data - data for constructing the mask

    Output:
            cube_flatfielded - new copy of the cube with the
                                flatfielding applied

    Notes: Also want to exclude the edge spaxels from the analysis as these 
            contain different exposure times to the rest of the cube. 

    """
    #  OIII wavelength
    central_l = (1 + redshift) * 0.500824

    # find the gal_name
    gal_name = cube[len(cube) -
                            cube[::-1].find("/"):][:-5]
    # read in the object 
    cube_table = fits.open(cube)

    data = cube_table[1].data

    data_header = cube_table[1].header

    start_l = data_header['CRVAL3']

    dl = data_header['CDELT3']

    wave_array = start_l + np.arange(0, 2048*dl, dl)

    o_peak = np.argmin(abs(central_l - wave_array))

    f_data = copy(data)

    # Now for each wavelength slice - flatfield using the mask limits
    # start with i, j > 3 in both cases to mitigate edge effects with lower
    # exposure times

    dim_x = f_data.shape[1]
    dim_y = f_data.shape[2]

    # create the mask and masked cube to take the medians from

    mask = np.ones(shape=(dim_x, dim_y))

    # every position near the edge and on top of the object is
    # masked with a np.nan value

    mask[0:3, :] = np.nan
    mask[-3:, :] = np.nan
    mask[:, 0:3] = np.nan
    mask[:, -3:] = np.nan

    # construct the mask from the accepted spaxels in the fitting process
    for i in range(m_data.shape[0]):

        for j in range(m_data.shape[1]):

            if np.isnan(m_data[i,j]):

                m_data[i,j] = 1.0

            else:

                m_data[i,j] = np.nan

    mask = mask * m_data

    # masked cube is given by multiplying the mask
    # with the cube and using the mapping function

    masked_data = f_data*mask[np.newaxis,:,:]

    # loop round each surface and subtract the median values
    for i in range(f_data.shape[0]):

        for x in range(3, f_data.shape[1] - 3):

            f_data[i, x, :] = f_data[i, x, :] - np.nanmedian(masked_data[i, x, :])

        for y in range(3, data.shape[2] - 3):

            f_data[i, :, y] = f_data[i, :, y] - np.nanmedian(masked_data[i, :, y])

    o_nband = np.nanmedian(f_data[o_peak-5:o_peak+5, :, :], axis=0)

#    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
#    im = ax[0].imshow(np.nanmedian(data[100:1350, :, :], axis=0),
#                   cmap=plt.get_cmap('hot'),
#                   interpolation='nearest',
#                   vmin=-0.0,
#                   vmax=0.4)
#    # add colourbar to each plot
#    divider = make_axes_locatable(ax[0])
#    cax_new = divider.append_axes('right', size='10%', pad=0.05)
#    plt.colorbar(im, cax=cax_new)
#    im = ax[1].imshow(np.nanmedian(f_data[100:1350, :, :], axis=0),
#                   cmap=plt.get_cmap('hot'),
#                   interpolation='nearest',
#                   vmin=-0.3,
#                   vmax=-0.098918)
#    # add colourbar to each plot
#    divider = make_axes_locatable(ax[1])
#    cax_new = divider.append_axes('right', size='10%', pad=0.05)
#    plt.colorbar(im, cax=cax_new)
#    im = ax[2].imshow(o_nband,
#                   cmap=plt.get_cmap('hot'),
#                   interpolation='nearest',
#                   vmin=-0.3,
#                   vmax=3)
#    # add colourbar to each plot
#    divider = make_axes_locatable(ax[2])
#    cax_new = divider.append_axes('right', size='10%', pad=0.05)
#    plt.colorbar(im, cax=cax_new)
#    # set the title
#    ax[0].set_title('Cont_1')
#    ax[1].set_title('Cont_2')
#    ax[2].set_title('OIII')
#    fig.tight_layout()
#    plt.suptitle('%s' % gal_name, fontsize=25)
#    #plt.show()
#    fig.savefig('%s_flatfield.png' % cube[:-5])
#    plt.close('all')

    return {'cont1' : np.nanmedian(data[100:1350, :, :], axis=0),
            'cont2' : np.nanmedian(f_data[100:1350, :, :], axis=0),
            'OIII' : o_nband}

#c = '/Users/owenturner/DATA/uncalibrated_goods_p1_0.8_10_better/Science/combine_sci_reconstructed_bs008543.fits'
#v = fits.open('/Users/owenturner/DATA/uncalibrated_goods_p1_0.8_10_better/Science/combine_sci_reconstructed_bs008543_vel_field.fits')
#v = v[0].data
#flatfield(c, v, 3.473288)

