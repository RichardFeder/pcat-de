''' 
This file stores the parameters for PCAT-DE.

In pcat_main(), these parameters are loaded in as default to the PCAT global object unless otherwise
specified as kwargs in the Lion instantiation.

These parameters can be broken down into the following groups:
- Computational Routine Options
- Data Configuration
- Image bands/sizing
- Sampler parameters
- Background parameters


 '''

import numpy as np
import config

# ---------------------- parameters unique to SPIRE data -----------------------------

sb_conversion_dict = dict({'S': 86.29e-4, 'M':16.65e-3, 'L':34.52e-3})
temp_mock_amps_dict = dict({'S':0.03, 'M': 0.2, 'L': 0.8}) # MJy/sr, this was for RXJ1347

band_dict = dict({0:'S',1:'M',2:'L'}) # for accessing different wavelength filenames
lam_dict = dict({'S':250, 'M':350, 'L':500})
pixsize_dict = dict({'S':6., 'M':8., 'L':12.})

spire_bands = ['S', 'M', 'L']
nbl = np.arange(len(spire_bands))
fourier_band_idxs = nbl.copy()
template_bands_dict = dict({'sze':['M', 'L']}) # should just integrate with the same thing in Lion main

template_band_idxs_dict = dict({'sze':nbl, 'cib':nbl})

fourier_band_idxs = None # if left unspecified, assume user wants Fourier component templates fit across all bands
temp_amplitude_sigs = dict({'sze':0.001, 'fc':0.001, 'binned_cib':0.0005}) # binned cib
sz_amp_sig = None

color_mus = dict({'S-M':0.0, 'M-L':0.5, 'L-S':0.5, 'M-S':0.0, 'S-L':-0.5, 'L-M':-0.5})
color_sigs = dict({'S-M':1.5, 'M-L':1.5, 'L-S':1.5, 'M-S':1.5, 'S-L':1.5, 'L-M':1.5}) #very broad color prior


# ----------------------------------- COMPUTATIONAL ROUTINE OPTIONS -------------------------------

# set to True if using CBLAS library
cblas=False
# set to True if using OpenBLAS library for non-Intel processors
openblas=False

# --------------------------------- DATA CONFIGURATION ----------------------------------------

# Configure these for individual directory structure
# config
base_path = config.base_path
result_basedir = config.result_basedir

data_path = None
# the tail name can be configured when reading files from a specific dataset if the name space changes.
# the default tail name should be for PSW, as this is picked up in a later routine and modified to the appropriate band.
tail_name = None
# file_path can be provided if only one image is desired with a specific path not consistent with the larger directory structure
file_path = None

im_fpath = None
err_fpath = None
dataname = None
# filepath for previous catalog if using as an initial state. loads in .npy files
load_state_timestr = None 
# set flag to True if you want posterior plots/catalog samples/etc from run saved
save_outputs = True
# initialize data object 
init_data_and_modl = True

# can specify these either to save parameter files (save_param_file=True) or to load them for another run (load_param_file=True)
save_param_file = True
load_param_file = False
param_filepath = 'params.txt'
param_read_filepath = 'params_read.txt'

image_extnames=['SIGNAL']
uncertainty_map_extname = 'ERROR'

use_uncertainty_map = True

# if set to true, Gaussian realization of error model is added to signal image
add_noise = False
# if scalar_noise_sigma specified, error map assumed to be gaussian with variance scalar_noise_sigma**2
scalar_noise_sigma = None
noise_fpath = None # ? track down

# Full width at half maximum for the PSF of the instrument/observation. Currently assumed to be Gaussian, but other 
# PCAT implementations have used a PSF template, so perhaps a more detailed PSF model could be added as another FITS header
psf_pixel_fwhm = 3.0
psf_fwhms = None

# TBD
psf_postage_stamps = None
nbin = 5 # upsamples template by factor of nbin

# if true catalog provided, passes on to posterior analysis
truth_catalog = None

# Image bands/sizing
# resizes images to largest square dimension modulo nregion
auto_resize = True
# don't use 'down' configuration yet, not implemented consistently in all data parsing routines
round_up_or_down = 'up'

# replace with specifying masked pixels as zero or infinity in the uncertainty map.
use_mask = False
mask_file = None

#indices of bands used in fit, where 0->250um, 1->350um and 2->500um.
band_indices = None
band0 = 0
band1 = None
band2 = None

# ---------------------------------- SAMPLER PARAMS ------------------------------------------

# number of thinned samples
nsamp = 1000
# factor by which the chain is thinned
nloop = 1000

print_movetypes = dict({'movestar':'P *','birth_death':'BD *','merge_split':'MS *','bkg':'BKG', 'template':'TEMPLATE','fourier_comp':'FC','bincib':'BINCIB'})

moveweight_byprop = dict({'movestar':80., 'birth_death':60., 'merge_split':60., 'bkg':10., 'template':40., 'fc':40.})
# number of thinned samples before proposals of various types are included in the fit. The default is for these to all be zero.
sample_delay_byprop = dict({'movestar':0, 'birth_death':0, 'merge_split':0, 'bkg':0, 'template':0, 'fc':0}) 

# scalar factor in regularization prior, scales dlogL penalty when adding/subtracting a source
alph = 1.0
# if set to True, computes parsimony prior using F-statistic, nominal_nsrc and the number of pixels in the images
F_statistic_alph = False
# scale for merge proposal i.e. how far you look for neighbors to merge
kickrange = 1.0
# used in subregion model evaluation
margin = 10
# maximum number of sources allowed in the code, might change depending on the image
max_nsrc = 2000
# nominal number of sources expected in a given image, helps set sample step sizes during MCMC
nominal_nsrc = 1000
# splits up image into subregions to do proposals within
nregion = 5
# used when splitting sources and determining colors of resulting objects
split_col_sig = 0.2
# power law type
flux_prior_type = 'single_power_law'
# number counts single power law slope for sources
truealpha = 3.0
# minimum flux allowed in fit for sources (in Jy)
trueminf = 0.005
# two parameters for double power law, one for pivot flux density
alpha_1 = 1.01
alpha_2 = 3.5
pivot_dpl = 0.01

# # these two, if specified, should be dictionaries with the color prior mean and width (assuming Gaussian)
# color_mus = None
# color_sigs = None

temp_prop_sig_fudge_facs = None

# if specified, nsrc_init is the initial number of sources drawn from the model. otherwise a random integer between 1 and max_nsrc is drawn
nsrc_init = None
err_f_divfac = 2.

# if specified, delays all point source modeling until point_src_delay samples have passed. 
point_src_delay = None


# ---------------------------------- BACKGROUND PARAMS --------------------------------

# bkg_level is used now for the initial background level for each band
bkg_level = None
# mean offset can be used if one wants to subtract some initial level from the input map, but setting the bias to the value 
# is functionally the same
mean_offsets = None
# boolean determining whether to use background proposals
float_background = False
# bkg_sig_fac scales the width of the background proposal distribution
bkg_sig_fac = 5.
# if set to True, includes Gaussian prior on background amplitude with mean bkg_mus[bkg_idx] and scale bkg_prior_sig. this is not used in practice but could be useful
dc_bkg_prior = False
# background amplitude Gaussian prior mean [in Jy/beam]
bkg_prior_mus = None
# background amplitude Gaussian prior width [in Jy/beam]
bkg_prior_sig = 0.01

# ---------------------------------- TEMPLATE PARAMS ----------------------------------------

# boolean determining whether to float emission template amplitudes, e.g. for SZ or lensing templates
float_templates = False
# names of templates to use in fit, I think there will be a separate template folder where the names specify which files to read in
template_names = None

n_templates = 0

# initial amplitudes for specified templates
init_template_amplitude_dicts = None
# if template file name is not None then it will grab the template from this path and replace PSW with appropriate band
template_filename = None

# boolean which when True results in a delta function color prior for dust templates 
delta_cp_bool = False

# inject_diffuse_comp = False
# diffuse_comp_path = None
diffuse_comp_fpaths = None

# coupled proposals between template describing profile (e.g., SZ surface brightness profile) and point sources
# coupled_profile_temp_prop = False
# number of sources to perturb each time
# coupled_profile_temp_nsrc = 1

# float_cib_templates = False
# cib_nregion = 5
# binned_cib_amp_sig = None, # binned cib \
# binned_cib_moveweight = 40.
# binned_cib_sample_delay = 0.
# binned_cib_relamps = None

# ---------------------------------- FOURIER COMPONENT PARAMS ----------------------------------------

# bool determining whether to fit fourier comps 
float_fourier_comps = False

# if there is some previous model component derived in terms of the 2D fourier expansion, they can be specified with this param
init_fourier_coeffs = None

# for multiple bands this sets the relative normalization
fc_rel_amps = None

# for a given proposal, this is the probability that the fourier coefficients are perturbed rather than 
# the relative amplitude of the coefficients across bands
dfc_prob = 0.5

# this specifies the order of the fourier expansion. the number fourier components that are fit for is equal to 4 x fourier_order**2
fourier_order = 5

# this is for perturbing the relative amplitudes of a fixed Fourier comp model across bands
fourier_amp_sig = 0.0005

# perturb multiple FCs at once? this doesn't appear to change things much 
n_fc_perturb = 1

# specifies the proposal distribution width for the largest spatial mode of the Fourier component model, or for all of them if fc_prop_alpha=None
fc_amp_sig = None

bkg_moore_penrose_inv = False

ridge_fac = 10.

# if specified, ridge factor is proportional to wavenumber when added to diagonal model covariance matrix, effectively a power spectrum prior on diffuse component
ridge_fac_alpha = 1.3

# number of times to apply FC marg during burn in
n_marg_updates = 10

# number of thinned samples between each marginalization step
fc_marg_period = 5

# if True, PCAT couples Fourier component proposals with change in point source fluxes
coupled_fc_prop = False

# fraction of FC proposals that are coupled with pt src fluxes
coupled_fc_prop_frac = 0.5

# ----------------------------------- DIAGNOSTICS/POSTERIOR ANALYSIS -------------------------------------

# interactive backend should be loaded before importing pyplot
visual = False
# used for visual mode
weighted_residual = True
# panel list controls what is shown in six panels plotted by PCAT intermittently when visual=True  
panel_list = ['data0', 'model0', 'residual0', 'data_zoom0', 'dNdS0', 'residual_zoom0']
# plots visual frames every "plot_sample_period" thinned samples
plot_sample_period = 1
# can have fully deterministic trials by specifying a random initial seed 
init_seed = None
# verbosity during program execution
verbtype = 0
# number of residual samples to average for final product
residual_samples = 200
# set to True to automatically make posterior/diagnostic plots after the run 
make_post_plots = True
# used for computing posteriors
burn_in_frac = 0.75
# save posterior plots
bool_plot_save = True
# if PCAT run is part of larger ensemble of test realizations, a file with the associated run IDs (time strings) can be specified
# and updated with the current run ID.
timestr_list_file = None
# print script output to log file for debugging
print_log=False
# this parameter can be set to true when validating the input data products are correct
show_input_maps=False

n_frames = 0

# ----------------------------------------- CONDENSED CATALOG --------------------------------------

# if True, takes last n_condensed_samp catalog realizations and groups together samples to produce a marginalized 
# catalog with reported uncertainties for each source coming from the several realizations
generate_condensed_catalog = False
# number of samples to construct condensed catalog from. Condensing the catalog can take a long time, so I usually choose like 100 for a "quick" answer
# and closer to 300 for a science-grade catalog.
n_condensed_samp = 100
# cuts catalog sources that appear in less than {prevalence_cut} fraction of {n_condensed_samp} samples
prevalence_cut = 0.1
# removes sources within {mask_hwhm} pixels of image border, in case there are weird artifacts
mask_hwhm = 2

# used when grouping together catalog sources across realizations
search_radius=0.75
matching_dist = 0.75


