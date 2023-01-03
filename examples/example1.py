import numpy as np
from pcat_main import *
import config

''' 
This script includes an example for running PCAT on single- and multi-band data, with Fourier components. 

The full set of configurable parameters may be found in params.py

'''

bands = ['S', 'M', 'L']

# Planck extrapolated cirrus
cirrus_fpaths = [config.data_dir+'/mock_cirrus/mock_cirrus_P'+band+'W.fits' for band in bands]

# images
dataname = 'rxj1347'
tail_names = ['rxj1347_P'+band+'W_nr_1_ext' for band in bands]
data_fpaths = [config.data_dir+dataname+'/'+tail_name for tail_name in tail_names]
print('Will load data from data_fpaths:', data_fpaths)
# instantiate the lion class


panel_list = ['data0', 'data1', 'data2',  'residual0','residual1','residual2'] # get default for this instead of specifying

image_extnames = ['SIGNAL']
uncertainty_map_extname = 'ERROR'

# verbtype indicates the verbosity of PCAT-DE where verbtype=0 means no additional print statements included,
# while 1 includes print statements related to data parsing and 2 is typically for diagnostic mode, printing out
# information related to the drawn samples in PCAT.
verbtype=1

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

pcat_obj = lion(bands, nsamp=1000, max_nsrc=1000, fmin=fmin, truealpha=truealpha, flux_prior_type=flux_prior_type, \
				image_extnames = image_extnames, uncertainty_map_extname=uncertainty_map_extname, diffuse_comp_fpaths=diffuse_comp_fpaths, \
				data_fpaths=data_fpaths, show_input_maps=True, visual=False,  make_post_plots=True, \
				float_fourier_comps=float_fourier_comps, fourier_order=fourier_order, verbtype=verbtype)

# run it!
pcat_obj.main()


'''
Once PCAT-DE is finished running, you may find plots with results of the run in the result_path directory, 
which is set in config.py. 
'''



