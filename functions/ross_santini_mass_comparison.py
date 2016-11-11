import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.io import fits
from scipy.stats import binned_statistic
from matplotlib import rc
import matplotlib.ticker as ticker


def compare_mass(catalogue):

    table = ascii.read(catalogue)

    # redshift 3-4 distribution plot
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    # prepare the plotting window

    fig, ax = plt.subplots(1, 1, figsize=(8,8))

    ax.set_xlabel(r'Ross Z$\odot$ log(M$^{*}$/M$\odot$)',
                  fontsize=24,
                  fontweight='bold',
                  labelpad=30)

    ax.set_ylabel(r'Santini (Holden) log(M$^{*}$/M$\odot$)',
                  fontsize=24,
                  fontweight='bold',
                  labelpad=30)

    # tick parameters 
    ax.tick_params(axis='both',
                       which='major',
                       labelsize=18,
                       length=10,
                       width=2)

    ax.tick_params(axis='both',
                       which='minor',
                       labelsize=18,
                       length=5,
                       width=1)

    # read in the different mass values and convert to same baseline

    ross_mass = table['Mstar']

    sant_mass = np.log10(table['M_med'])

    sant_mass_neb = np.log10(table['M_neb_med'])

    ax.scatter(ross_mass, sant_mass, marker='o', s=75,color='red',label='santini no neb')
    ax.scatter(ross_mass, sant_mass_neb, marker='+', s=75,color='blue',label='santini with neb')
    ax.plot([8.5, 11],[8.5, 11])

    # also want to read in the Holden values to plot

    table_2 = ascii.read('/scratch2/oturner/disk1/turner/DATA/MASS_CATALOGS/KDS_GOODS_SAMPLE/ross_holden_mass_comparison.txt')
    ross_mass_2 = table_2['Ross_Mstar']
    ross_sfr = table_2['Ross_SFR']
    holden_mass = table_2['Holden_Mstar']
    holden_sfr_UV = table_2['Holden_SFR_UV']
    holden_sfr_SED = table_2['Holden_SFR_SED']

    ax.scatter(ross_mass_2, holden_mass, marker='^', s=75,color='orange',label='holden vs ross')

    fig.tight_layout()
    ax.minorticks_on()
    ax.legend(loc='upper left')

    plt.show()

    fig.savefig('/scratch2/oturner/disk1/turner/DATA/MASS_CATALOGS/ross_santini_mass_comparison.png')

    plt.close('all')

    # make second plot comparing the SFR of Holden and Ross

    fig, ax = plt.subplots(1, 1, figsize=(8,8))

    ax.set_xlabel(r'Ross 0.2Z$\odot$ SFR M$\odot$yr$^{-1}$',
                  fontsize=24,
                  fontweight='bold',
                  labelpad=30)

    ax.set_ylabel(r'Holden SFR M$\odot$yr$^{-1}$',
                  fontsize=24,
                  fontweight='bold',
                  labelpad=30)

    # tick parameters 
    ax.tick_params(axis='both',
                       which='major',
                       labelsize=18,
                       length=10,
                       width=2)

    ax.tick_params(axis='both',
                       which='minor',
                       labelsize=18,
                       length=5,
                       width=1)

    ax.scatter(ross_sfr, holden_sfr_UV, marker='o', s=75,color='blue',label='holden UV sfr')
    ax.scatter(ross_sfr, holden_sfr_SED, marker='+', s=75,color='red',label='holden SED sfr')
    ax.plot([0, 400],[0,400])
    ax.set_ylim(0, 400)

    fig.tight_layout()
    ax.minorticks_on()
    ax.legend(loc='upper left')

    plt.show()

    fig.savefig('/scratch2/oturner/disk1/turner/DATA/MASS_CATALOGS/ross_holden_sfr_comparison_0.2z.png')

    plt.close('all')



compare_mass('/scratch2/oturner/disk1/turner/DATA/MASS_CATALOGS/KDS_GOODS_SAMPLE/kds_sample_ross_santini_matched.cat')