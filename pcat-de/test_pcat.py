import numpy as np
from astropy.io import fits

import config 
from params import *
from pcat_main import *


''' 
Here we specify the components we want to add into our image. We use controlled mocks in this example, adding:
    - mock CIB realization ('CIB_SIG')
    - galactic cirrus at the 4x-Planck level ('CIRRUS_4XP)
    - a Gaussian realization of instrument noise ('NOISE')
If we were using real data, we would instead pass a single keyword such as 'SIGNAL', or pass the image to PCAT-DE
by specifying the image_fpath parameter.
'''

band = 'S'
bands = [band]
sim_idx = 1

# images
dataname = 'mock_cib'
tail_names = ['P'+band+'W_10arcmin_sim0']

nplanck = 4
image_extnames = ['CIB_SIG', 'CIRRUS_'+str(nplanck)+'XP', 'NOISE']
# image_extnames = ['CIB_SIG', 'NOISE']

uncertainty_map_extname = 'ERROR'
data_fpaths = [config.data_dir+'mock_cib_obs_PSW_sim1.fits']

# print('Will load data from data_fpaths:', data_fpaths)
panel_list = ['data0','model0','residual0']

# verbtype indicates the verbosity of PCAT-DE where verbtype=0 means no additional print statements included,
# verbtype=1 includes print statements related to data parsing and 2 is "diagnostic" mode, printing out
# a boat load of information related to the proposals, likelihoods and drawn samples in PCAT.
verbtype=0

float_background = True
# float_background, bias, nominal_nsrc, generate_condensed_catalog, n_condensed_samp, err_f_divfac, bkg_sig_fac, init_seed
use_mask = False # mask could be a data product in PCAT but dont do any cropping based on it. 

# Flux densities are given in units of Jy. For the example provided (SPIRE), the flux density distribution is well approximated 
# by a single power law with index 3, however other assumptions may be made.
fmin = 0.003
truealpha = 3.
flux_prior_type = 'single_power_law'

# Fourier components can be included with float_fourier_comps=True, with order set by fourier_order.
float_fourier_comps = True
fourier_order=10
# the ridge factor imposes a Gaussian prior on the Fourier components, 
# which is proportional to the power spectrum of cirrus dust on 10 arcminute scales.
ridge_fac = 5e5/(nplanck/8.)**2

# moveweight_byprop = dict({'movestar':0., 'birth_death':0., 'merge_split':0., 'bkg':0., 'template':40., 'fc':40.})


pcat_obj = lion(bands=bands, nsamp=100, max_nsrc=1000, fmin=fmin, truealpha=truealpha, flux_prior_type=flux_prior_type, \
                image_extnames = image_extnames, uncertainty_map_extname=uncertainty_map_extname, \
                data_fpaths=data_fpaths, show_input_maps=False, visual=False,  make_post_plots=True, \
                float_fourier_comps=float_fourier_comps, fourier_order=fourier_order, verbtype=verbtype, \
               save_outputs=True, load_param_file=False, init_data_and_modl=True, \
               bkg_moore_penrose_inv=float_fourier_comps, ridge_fac=ridge_fac, \
               float_background=float_background, nregion=2)

pcat_obj.main()


# If the chain is saved but you want to iterate on something in the result plots, simply query the timestring associated 
# with the run to reload its contents. 

# result_plots(timestr='20230614-180802', generate_condensed_cat=False, n_condensed_samp=100, prevalence_cut=0.95,\
#              mask_hwhm=2, condensed_catalog_plots=False)










