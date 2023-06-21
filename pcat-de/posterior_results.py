import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fftpack import fft, ifft
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import sys
import pandas as pd
from pcat_main import *
# from spire_roc_condensed_cat import *
from diffuse_gen import *

from fourier_bkg_modl import *

def add_directory(dirpath):
	if not os.path.isdir(dirpath):
		os.makedirs(dirpath)
	return dirpath


def compute_dNdS(trueminf, stars, nsrc, _F=2):

	""" function for computing number counts """

	binz = np.linspace(np.log10(trueminf)+3., np.ceil(np.log10(np.max(stars[_F, 0:nsrc]))+3.), 20)
	hist = np.histogram(np.log10(stars[_F, 0:nsrc])+3., bins=binz)
	logSv = 0.5*(hist[1][1:]+hist[1][:-1])-3.
	binz_Sz = 10**(binz-3)
	dSz = binz_Sz[1:]-binz_Sz[:-1]
	dNdS = hist[0]

	return logSv, dSz, dNdS


def compute_degradation_fac(condensed_cat, err, flux_err_idx, smooth_fac=5, xidx=0, yidx=2, psf_fwhm=3.):

	""" 
	Given a condensed catalog and underlying noise model, compute the departure of PCAT estimated source uncertainties from 
	what one would expect for an isolated source.

	Parameters
	----------
	condensed_cat : `~numpy.ndarray' of shape (nsrc, n_features)
		Input condensed catalog
	err : `~numpy.ndarray'
		Noise model for field being cataloged.
	smooth_fac : float, optional
		Because there may be significant pixel variation in the noise model, smooth_fac sets the scale for smoothing the error map, 
		such that when the error is quoted based on a sources position, it represents something closer to the effective (beam averaged) noise.
		Default is 5 pixels. 
	psf_fwhm : float, optional
		Full width at half maximum for point spread function. Default is 3 pixels. 

	Returns
	-------
	flux_deg_fac : `~numpy.ndarray' of length (N_src)
		array of flux degradation factors for each source in the condensed catalog.

	"""

	optimal_ferr_map = np.sqrt(err**2/(4*np.pi*(psf_fwhm/2.355)**2))
	smoothed_optimal_ferr_map = gaussian_filter(optimal_ferr_map, smooth_fac)
	flux_deg_fac = np.zeros_like(condensed_cat[:,0])

	for s, src in enumerate(condensed_cat):
		flux_deg_fac[s] = src[flux_err_idx]/smoothed_optimal_ferr_map[int(src[xidx]), int(src[yidx])]
	
	return flux_deg_fac


def grab_extent(bins_var1, bins_var2):
    extent = [bins_var1[0], bins_var1[-1], bins_var2[0], bins_var2[-1]]
    
    return extent



def compute_gelman_rubin_diagnostic(list_of_chains, i0=0):
    
    list_of_chains = np.array(list_of_chains)
    print('list of chains has shape ', list_of_chains.shape)
    m = len(list_of_chains)
    n = len(list_of_chains[0])-i0
    
    print('n=',n,' m=',m)
    
    B = (n/(m-1))*np.sum((np.mean(list_of_chains[:,i0:], axis=1)-np.mean(list_of_chains[:,i0:]))**2)
    
    W = 0.
    for j in range(m):
        sumsq = np.sum((list_of_chains[j,i0:]-np.mean(list_of_chains[j,i0:]))**2)
                
        W += (1./m)*(1./(n-1.))*sumsq
    
    var_th = ((n-1.)/n)*W + (B/n)
    
    Rhat = np.sqrt(var_th/W)
    
    print("rhat = ", Rhat)
    
    return Rhat, m, n


def compute_chain_rhats(all_chains, labels=[''], i0=0, nthin=1):
    
    rhats = []
    for chains in all_chains:
        chains = np.array(chains)
        print(chains.shape)
        rhat, m, n = compute_gelman_rubin_diagnostic(chains[:,::nthin], i0=i0//nthin)
                    
        rhats.append(rhat)
        
    f = plt.figure()
    plt.title('Gelman Rubin statistic $\\hat{R}$ ($N_{c}=$'+str(m)+', $N_s=$'+str(n)+')', fontsize=14)
    barlabel = None
    if nthin > 1:
        barlabel = '$N_{thin}=$'+str(nthin)
    plt.bar(labels, rhats, width=0.5, alpha=0.4, label=barlabel)
    plt.axhline(1.2, linestyle='dashed', label='$\\hat{R}$=1.2')
    plt.axhline(1.1, linestyle='dashed', label='$\\hat{R}$=1.1')

    plt.legend()
    plt.xticks(fontsize=16)
    plt.ylabel('$\\hat{R}$', fontsize=16)
    plt.show()
    
    return f, rhats

def spec(x, order=2):
    from statsmodels.regression.linear_model import yule_walker
    beta, sigma = yule_walker(x, order)
    return sigma**2 / (1. - np.sum(beta))**2
    
def geweke_test(chain, first=0.1, last=0.5, intervals=20):
    ''' Adapted from pymc's diagnostics.py script '''
    
    assert first+last <= 1.0
    zscores = [None] * intervals
    starts = np.linspace(0, int(len(chain)*(1.-last)), intervals).astype(int)

    # Loop over start indices
    for i,s in enumerate(starts):

        # Size of remaining array
        x_trunc = chain[s:]
        n = len(x_trunc)

        # Calculate slices
        first_slice = x_trunc[:int(first * n)]
        last_slice = x_trunc[int(last * n):]

        z = (first_slice.mean() - last_slice.mean())
        z /= np.sqrt(spec(first_slice)/len(first_slice) +
                     spec(last_slice)/len(last_slice))
        zscores[i] = len(chain) - n, z

    return zscores  
    


def result_plots(timestr=None, burn_in_frac=0.8, boolplotsave=True, boolplotshow=False, \
				plttype='png', gdat=None, dpi=150, sb_unit='MJy/sr', \
				accept_fraction_plots=True, chi2_plots=True, dc_background_plots=True, fourier_comp_plots=True, \
				template_plots=True, flux_dist_plots=True, flux_color_plots=False, flux_color_color_plots=True, \
				comp_resources_plot=True, source_number_plots=True, residual_plots=True, condensed_catalog_plots=False, condensed_catalog_fpath=None, generate_condensed_cat=False, \
				n_condensed_samp=None, prevalence_cut=None, mask_hwhm=None, search_radius=None, matching_dist=None, truth_catalog=None, \
				title_band_dict=None, temp_mock_amps_dict=None, lam_dict=None, pixel_sizes=None):

	"""
	Main wrapper function for producing PCAT result diagnostics and posteriors.
	This is an in place operation.

	Parameters
	----------
	timestr : 
	burn_in_frac : 
	boolplotsave : 
	boolplotshow : 
	plttype : 
	gdat : 
	dpi : 
	sb_unit : 
	accept_fraction_plots, chi2_plots, dc_background_plots, fourier_comp_plots,\
		template_plots, flux_dist_plots, flux_color_plots, flux_color_color_plots, 
		comp_resources_plot, source_number_plots, residual_plots, condensed_catalog_plots : 'bools'. These determine which plots are produced and which are not.
		Most are defaulted to True, aside from some which are either deprecated or incomplete.

	condensed_catalog_fpath : 
	n_condensed_samp :
	prevalence_cut : 
	mask_hwhm : 
	search_radius : 
	matching_dist : 
	truth_catalog : 
		Default is None.
	title_band_dict : 
	temp_mock_amps_dict : 
	lam_dict :
	pixel_sizes : 
	
	"""
	if title_band_dict is None:
		title_band_dict = dict({0:'250 micron', 1:'350 micron', 2:'500 micron'})
	if temp_mock_amps_dict is None:
		temp_mock_amps_dict = dict({'S':0.4, 'M': 0.2, 'L': 0.8}) # MJy/sr, this was to test how adding a signal at 250 um (like dust) would affect the fit if not modeled

	
	if gdat is None:
		# gdat, filepath, result_path = load_param_dict(timestr=timestr)
		gdat, filepath = load_param_dict(timestr=timestr)
		gdat.burn_in_frac = burn_in_frac
		gdat.boolplotshow = boolplotshow
		gdat.boolplotsave = boolplotsave
		gdat.filepath = filepath
		gdat.result_path = config.result_basedir
		gdat.timestr = timestr
		gdat.psf_fwhms = [3., 3., 3.]

	else:
		gdat.filepath = config.result_basedir+'/'+gdat.timestr
		gdat.result_path = config.result_basedir

	if lam_dict is None:
		lam_dict = gdat.lam_dict
		# lam_dict = dict({0:250, 1:350, 2:500}) # microns
	if pixel_sizes is None:
		pixel_sizes = gdat.pixsize_dict
		# pixel_sizes = dict({'S':6, 'M':8, 'L':12}) # arcseconds
	# if truth_catalog is None:
		# if gdat.truth_catalog is not None:
			# truth_catalog = gdat.truth_catalog

	if matching_dist is not None:
		gdat.matching_dist = matching_dist
	if n_condensed_samp is not None:
		gdat.n_condensed_samp = n_condensed_samp
	if prevalence_cut is not None:
		gdat.prevalence_cut = prevalence_cut
	if mask_hwhm is not None:
		gdat.mask_hwhm = mask_hwhm
	if search_radius is not None:
		gdat.search_radius = search_radius

	condensed_cat = None

	if condensed_catalog_plots:

		condensed_cat_dir = add_directory(gdat.result_path+'/'+timestr+'/condensed_catalog')

		if generate_condensed_cat:
			print('Generating condensed catalog from last '+str(gdat.n_condensed_samp)+' samples of catalog ensemble..')
			print('prevalence_cut = '+str(gdat.prevalence_cut))
			print('search_radius = '+str(gdat.search_radius))
			print('mask_hwhm = '+str(mask_hwhm))

			xmatch_roc = cross_match_roc(timestr=gdat.timestr, nsamp=gdat.n_condensed_samp)
			xmatch_roc.load_chain(gdat.result_path+'/'+gdat.timestr+'/chain.npz')
			xmatch_roc.load_gdat_params(gdat=gdat)
			condensed_cat, seed_cat, column_names = xmatch_roc.condense_catalogs(prevalence_cut=gdat.prevalence_cut, save_cats=True, make_seed_bool=True, \
																				mask_hwhm=gdat.mask_hwhm, search_radius=gdat.search_radius)
			np.savetxt(gdat.result_path+'/'+gdat.timestr+'/raw_seed_catalog_nsamp='+str(gdat.n_condensed_samp)+'_matching_dist='+str(gdat.matching_dist)+'_maskhwhm='+str(gdat.mask_hwhm)+'.txt', seed_cat)
			np.savetxt(gdat.result_path+'/'+gdat.timestr+'/condensed_catalog_nsamp='+str(gdat.n_condensed_samp)+'_matching_dist='+str(gdat.matching_dist)+'_maskhwhm='+str(gdat.mask_hwhm)+'.txt', condensed_cat)

		else:
			if condensed_catalog_fpath is None:
				tail_name = 'condensed_catalog_nsamp='+str(gdat.n_condensed_samp)+'_prevcut='+str(gdat.prevalence_cut)+'_searchradius='+str(gdat.search_radius)+'_maskhwhm='+str(gdat.mask_hwhm)
				condensed_catalog_fpath = gdat.result_path+'/'+gdat.timestr+'/'+tail_name+'.npz'
				print('Loading condensed catalog from ', condensed_catalog_fpath)
				condensed_cat = np.load(condensed_catalog_fpath)['condensed_catalog']
				print('Condensed_cat has shape ', condensed_cat.shape)

	gdat.show_input_maps=False

	dat = pcat_data(gdat, nregion=gdat.nregion)
	dat.load_in_data(gdat)

	chain = np.load(gdat.result_path+'/'+gdat.timestr+'/chain.npz')

	# sb_conversion_dict = dict({'S': 86.29e-4, 'M':16.65e-3, 'L':34.52e-3})

	fd_conv_fac = None # if units are MJy/sr this changes to a number, otherwise default flux density units are mJy/beam
	nsrcs = chain['n']
	xsrcs = chain['x']
	ysrcs = chain['y']
	fsrcs = chain['f']
	chi2 = chain['chi2']
	timestats = chain['times']
	accept_stats = chain['accept']
	diff2s = chain['diff2s']

	burn_in = int(gdat.nsamp*burn_in_frac)
	bands = gdat.bands
	print('Bands are ', bands)

	if gdat.float_background:
		bkgs = chain['bkg']

	if gdat.float_templates:
		if gdat.n_templates > 0:
			template_amplitudes = chain['template_amplitudes']

	# if gdat.float_cib_templates:
		# binned_cib_coeffs = chain['binned_cib_coeffs']

	if gdat.float_fourier_comps: # fourier comps
		fourier_coeffs = chain['fourier_coeffs']

	# ------------------- mean residual ---------------------------

	if residual_plots:

		for b in range(gdat.nbands):
			residz = chain['residuals'+str(b)]
			median_resid = np.median(residz, axis=0)

			print('Median residual has shape '+str(median_resid.shape))
			smoothed_resid = gaussian_filter(median_resid, sigma=3)

			if b==0:
				resid_map_dir = add_directory(gdat.result_path+'/'+gdat.timestr+'/residual_maps')
				onept_dir = add_directory(gdat.result_path+'/'+gdat.timestr+'/residual_1pt')

			if sb_unit=='MJy/sr':
				fd_conv_fac = gdat.sb_conversion_dict[gdat.band_dict[bands[b]]]
				print('fd conv fac is ', fd_conv_fac)
			

			f_last = plot_residual_map(residz[-1], mode='last', band=title_band_dict[bands[b]], show=boolplotshow, convert_to_MJy_sr_fac=None)
			f_last.savefig(resid_map_dir +'/last_residual_and_smoothed_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)

			f_median = plot_residual_map(median_resid, mode='median', band=title_band_dict[bands[b]], show=boolplotshow, convert_to_MJy_sr_fac=None)

			f_median.savefig(resid_map_dir +'/median_residual_and_smoothed_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)

			median_resid_rav = median_resid[dat.weights[b] != 0.].ravel()

			noise_mod = dat.uncertainty_maps[b]

			f_1pt_resid = plot_residual_1pt_function(median_resid_rav, mode='median', band=title_band_dict[bands[b]], show=False, convert_to_MJy_sr_fac=None)
			f_1pt_resid.savefig(onept_dir +'/median_residual_1pt_function_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)

			plt.close()	

	# -------------------- CHI2 ------------------------------------

	if chi2_plots:
		sample_number = np.arange(burn_in, gdat.nsamp)
		full_sample = range(gdat.nsamp)

		chi2_dir = add_directory(gdat.result_path+'/'+gdat.timestr+'/chi2')


		ndof_list = [dat.fracs[b]*dat.uncertainty_maps[b].shape[0]*dat.uncertainty_maps[b].shape[1] for b in range(gdat.nbands)]

		f_multi_chi2_reduced = plot_multiband_chi_squared([chi2[:,b] for b in range(chi2.shape[1])], sample_number, band_list=[title_band_dict[band] for band in bands], show=False, \
			ndof_list=ndof_list)
		f_multi_chi2_reduced.savefig(chi2_dir+'/multiband_reduced_chi2.'+plttype, bbox_inches='tight', dpi=dpi)
		
		for b in range(gdat.nbands):

			fchi2 = plot_chi_squared(chi2[:,b], sample_number, band=title_band_dict[bands[b]], show=False)
			fchi2.savefig(chi2_dir + '/chi2_sample_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)

			plt.close()

	# ------------------------- BACKGROUND AMPLITUDE ---------------------

	if gdat.float_background and dc_background_plots:

		bkg_dir = add_directory(gdat.result_path+'/'+gdat.timestr+'/bkg')

		for b in range(gdat.nbands):

			if sb_unit=='MJy/sr':
				fd_conv_fac = gdat.sb_conversion_dict[gdat.band_dict[bands[b]]]
				print('fd conv fac is ', fd_conv_fac)

			f_bkg_chain = plot_bkg_sample_chain(bkgs[:,b], band=title_band_dict[bands[b]], show=False, convert_to_MJy_sr_fac=fd_conv_fac)
			f_bkg_chain.savefig(bkg_dir+'/bkg_amp_chain_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)
			
			if gdat.nsamp > 50:
				f_bkg_atcr = plot_atcr(bkgs[burn_in:, b], title='Background level, '+title_band_dict[bands[b]])
				f_bkg_atcr.savefig(bkg_dir+'/bkg_amp_autocorr_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)

			f_bkg_post = plot_posterior_bkg_amplitude(bkgs[burn_in:,b], band=title_band_dict[bands[b]], show=False, convert_to_MJy_sr_fac=None)
			f_bkg_post.savefig(bkg_dir+'/bkg_amp_posterior_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)

			plt.close()

	# ------------------------- BINNED CIB TEMPLATES --------------------

	# if gdat.float_cib_templates:

	# 	bcib_dir = add_directory(gdat.filepath+'/binned_cib')

	# 	print('Computing binned CIB posterior..')

	# 	dimxs = [gdat.imszs[b][0] for b in range(gdat.nbands)]
	# 	dimys = [gdat.imszs[b][1] for b in range(gdat.nbands)]

	# 	coarse_cib_templates = generate_subregion_cib_templates(dimxs, dimys, gdat.cib_nregion, cib_rel_amps=gdat.binned_cib_relamps)

	# 	for b in range(gdat.nbands):

	# 		f_bcib_median_std = plot_bcib_median_std(binned_cib_coeffs[burn_in:], coarse_cib_templates[b])
	# 		f_bcib_median_std.savefig(bcib_dir+'/bcib_model_median_std_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)

	# 	f_bcib_chain = plot_bcib_sample_chains(binned_cib_coeffs[burn_in:])
	# 	f_bcib_chain.savefig(bcib_dir+'/bcib_sample_chains.'+plttype, bbox_inches='tight', dpi=dpi)


	# ------------------------- FOURIER COMPONENTS ----------------------

	if gdat.float_fourier_comps and fourier_comp_plots:

		fc_dir = add_directory(gdat.result_path+'/'+gdat.timestr+'/fourier_comps')

		# median and variance of fourier component model posterior
		print('Computing Fourier component posterior..')
		f_fc_median_std = plot_fc_median_std(fourier_coeffs[burn_in:], gdat.imszs[0], ref_img=dat.data_array[0], convert_to_MJy_sr_fac=gdat.sb_conversion_dict['S'], psf_fwhm=3.)
		f_fc_median_std.savefig(fc_dir+'/fourier_comp_model_median_std.'+plttype, bbox_inches='tight', dpi=dpi)

		# sample chain for fourier coeffs
		f_fc_amp_chain = plot_fourier_coeffs_sample_chains(fourier_coeffs)
		f_fc_amp_chain.savefig(fc_dir+'/fourier_coeffs_sample_chains.'+plttype, bbox_inches='tight', dpi=dpi)


	
	# ------------------------- TEMPLATE AMPLITUDES ---------------------

	if gdat.float_templates and template_plots:

		template_dir = add_directory(gdat.result_path+'/'+gdat.timestr+'/templates')

		for t in range(gdat.n_templates):
			for b in range(gdat.nbands):

				if sb_unit=='MJy/sr':
					fd_conv_fac = gdat.sb_conversion_dict[gdat.band_dict[bands[b]]]
					print('fd conv fac is ', fd_conv_fac)

				if not np.isnan(gdat.template_band_idxs[t,b]):

					if gdat.template_order[t]=='dust' or gdat.template_order[t]=='planck':
						f_temp_amp_chain = plot_template_amplitude_sample_chain(template_amplitudes[:, t, b], template_name=gdat.template_order[t], band=title_band_dict[bands[b]], ylabel='Relative amplitude', convert_to_MJy_sr_fac=None) # newt
						f_temp_amp_post = plot_posterior_template_amplitude(template_amplitudes[burn_in:, t, b], template_name=gdat.template_order[t], band=title_band_dict[bands[b]], xlabel='Relative amplitude', convert_to_MJy_sr_fac=None) # newt
						f_temp_median_and_variance = plot_template_median_std(dat.template_array[b][t], template_amplitudes[burn_in:, t, b], template_name=gdat.template_order[t], band=title_band_dict[bands[b]], show=False, convert_to_MJy_sr_fac=fd_conv_fac)
						f_temp_median_and_variance.savefig(template_dir+'/'+gdat.template_order[t]+'_template_median_std_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)

					else:
						mock_truth = None
						if gdat.template_order[t]=='sze':
							mock_truth = None
							if gdat.inject_sz_frac is not None:
								mock_truth = temp_mock_amps_dict[gdat.band_dict[bands[b]]]*gdat.inject_sz_frac
								print('mock truth is ', mock_truth)


						else:
							if gdat.template_moveweight > 0:
								f_temp_amp_chain = plot_template_amplitude_sample_chain(template_amplitudes[:, t, b], template_name=gdat.template_order[t], band=title_band_dict[bands[b]], convert_to_MJy_sr_fac=fd_conv_fac)
								f_temp_amp_post = plot_posterior_template_amplitude(template_amplitudes[burn_in:, t, b],mock_truth=mock_truth,	template_name=gdat.template_order[t], band=title_band_dict[bands[b]], convert_to_MJy_sr_fac=fd_conv_fac)


					if gdat.template_order != 'sze' or b > 0: # test specific
						f_temp_amp_chain.savefig(template_dir+'/'+gdat.template_order[t]+'_template_amp_chain_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)
						f_temp_amp_post.savefig(template_dir+'/'+gdat.template_order[t]+'_template_amp_posterior_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)
					if gdat.nsamp > 50:

						if np.std(template_amplitudes[burn_in:, t, b]) != 0:
							f_temp_amp_atcr = plot_atcr(template_amplitudes[burn_in:, t, b], title='Template amplitude, '+gdat.template_order[t]+', '+title_band_dict[bands[b]]) # newt
							f_temp_amp_atcr.savefig(template_dir+'/'+gdat.template_order[t]+'_template_amp_autocorr_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)


	# ---------------------------- COMPUTATIONAL RESOURCES --------------------------------

	if comp_resources_plot:
		labels = ['Proposal', 'Likelihood', 'Implement']
		f_comp = plot_comp_resources(timestats, gdat.nsamp, labels=labels)
		f_comp.savefig(gdat.result_path+'/'+gdat.timestr+'/time_resource_statistics.'+plttype, bbox_inches='tight', dpi=dpi)
		plt.close()

	# ------------------------------ ACCEPTANCE FRACTION -----------------------------------------
	
	if accept_fraction_plots:	

		proposal_types = ['All', 'Move', 'Birth/Death', 'Merge/Split', 'Background', 'Templates', 'Fourier comps']

		skip_idxs = []
		if not gdat.float_background:
			skip_idxs.append(4)
		if not gdat.float_templates:
			skip_idxs.append(5)
		if not gdat.float_fourier_comps:
			skip_idxs.append(6)

		f_proposal_acceptance = plot_acceptance_fractions(accept_stats, proposal_types=proposal_types, skip_idxs=skip_idxs)
		f_proposal_acceptance.savefig(gdat.result_path+'/'+gdat.timestr+'/acceptance_fraction.'+plttype, bbox_inches='tight', dpi=dpi)


	# -------------------------------- ITERATE OVER BANDS -------------------------------------

	nsrc_fov = []
	color_lin_post_bins = np.linspace(0.0, 5.0, 30)

	flux_color_dir = add_directory(gdat.result_path+'/'+gdat.timestr+'/fluxes_and_colors')

	pairs = []

	fov_sources = [[] for x in range(gdat.nbands)]

	if condensed_catalog_plots:
		all_xs, all_ys = [condensed_cat[:,0]], [condensed_cat[:,2]]
		all_fs = [condensed_cat[:,5]]

	ndeg = None

	for b in range(gdat.nbands):

		color_lin_post = []
		residz = chain['residuals'+str(b)]
		median_resid = np.median(residz, axis=0)

		nbins = 20
		lit_number_counts = np.zeros((gdat.nsamp - burn_in, nbins-1)).astype(np.float32)
		raw_number_counts = np.zeros((gdat.nsamp - burn_in, nbins-1)).astype(np.float32)
		binz = np.linspace(np.log10(gdat.trueminf)+3.-1., 3., nbins)
		weight = dat.weights[b]
		
		pixel_sizes_nc = dict({0:6, 1:8, 2:12}) # arcseconds
		ratio = pixel_sizes_nc[b]/pixel_sizes_nc[0]


		if condensed_catalog_plots:

			if b > 0:
				dat.fast_astrom.fit_astrom_arrays(0, b, bounds0=gdat.bounds[0], bounds1=gdat.bounds[b])
				xp, yp = dat.fast_astrom.transform_q(condensed_cat[:,0], condensed_cat[:,2], b-1)
				xp[xp > dat.uncertainty_maps[b].shape[0]] = dat.uncertainty_maps[b].shape[0]-1.
				yp[yp > dat.uncertainty_maps[b].shape[1]] = dat.uncertainty_maps[b].shape[1]-1.

				fs = condensed_cat[:,5+4*b]

				all_xs.append(xp)
				all_ys.append(yp)
				all_fs.append(fs)

			psf_fwhm = 3.
			optimal_ferr_map = np.sqrt(dat.uncertainty_maps[b]**2/(4*np.pi*(psf_fwhm/2.355)**2))
			smoothed_optimal_ferr_map = gaussian_filter(optimal_ferr_map, 5)
			flux_err_idx = 6+4*b
			flux_deg_fac = np.zeros_like(condensed_cat[:,0])
			for s, src in enumerate(condensed_cat):
				if b > 0:
					flux_deg_fac[s] = src[flux_err_idx]/smoothed_optimal_ferr_map[int(np.floor(xp[s])), int(np.floor(yp[s]))]
				else:
					flux_deg_fac[s] = src[flux_err_idx]/smoothed_optimal_ferr_map[int(src[0]), int(src[1])]
				
			fdf_plot = plot_degradation_factor_vs_flux(1e3*condensed_cat[:,5+4*b], flux_deg_fac, deg_fac_mode='Flux')
			fdf_plot.savefig(condensed_cat_dir + '/flux_deg_fac_vs_flux_'+str(title_band_dict[bands[b]]) + '.'+plttype, bbox_inches='tight', dpi=dpi)

			flux_vs_fluxerr_plot = plot_flux_vs_fluxerr(1e3*condensed_cat[:,5+4*b], 1e3*condensed_cat[:,6+4*b])
			flux_vs_fluxerr_plot.savefig(condensed_cat_dir+'/flux_vs_fluxerr_'+str(title_band_dict[bands[b]])+'.'+plttype, bbox_inches='tight', dpi=dpi)


		if flux_dist_plots:
			for i, j in enumerate(np.arange(burn_in, gdat.nsamp)):
		
				fsrcs_in_fov = np.array([fsrcs[b][j][k] for k in range(nsrcs[j]) if dat.weights[0][int(ysrcs[j][k]),int(xsrcs[j][k])] != 0. and xsrcs[j][k] < dat.weights[0].shape[0] and ysrcs[j][k] < dat.weights[0].shape[1]])
				fov_sources[b].extend(fsrcs_in_fov)

				if b==0:
					nsrc_fov.append(len(fsrcs_in_fov))

				hist = np.histogram(np.log10(fsrcs_in_fov)+3, bins=binz)
				logSv = 0.5*(hist[1][1:]+hist[1][:-1])-3
				binz_Sz = 10**(binz-3)
				dSz = binz_Sz[1:]-binz_Sz[:-1]
				dNdS = hist[0]
				raw_number_counts[i,:] = hist[0]


			f_post_flux_dist = plot_posterior_flux_dist(logSv, raw_number_counts, band=title_band_dict[bands[b]])
			f_post_flux_dist.savefig(flux_color_dir+'/posterior_flux_histogram_'+str(title_band_dict[bands[b]])+'.'+plttype, bbox_inches='tight', dpi=dpi)



	# flux_cut_idx= 2
	flux_cut_idx = 0
	flux_cut = 0.025

	if condensed_catalog_plots:

		overlaid_condensed_cat = condensed_catalog_overlaid_data(dat.data_array, all_xs, all_ys, all_fs=all_fs, bands=[title_band_dict[band] for band in bands], flux_cut_idx=flux_cut_idx, flux_cut=flux_cut)
		overlaid_condensed_cat.savefig(condensed_cat_dir+'/condensed_catalog_overlaid_data_allbands_fluxcutidx='+str(flux_cut_idx)+'_fluxcut='+str(flux_cut)+'.'+plttype, bbox_inches='tight', dpi=dpi)

		all_overlaid_condensed_cat = condensed_catalog_overlaid_data(dat.data_array, all_xs, all_ys, all_fs=all_fs, bands=[title_band_dict[band] for band in bands])
		all_overlaid_condensed_cat.savefig(condensed_cat_dir+'/condensed_catalog_all_overlaid_data_allbands_maskhwhm='+str(mask_hwhm)+'.'+plttype, bbox_inches='tight', dpi=dpi)

		red_cut = True
		all_overlaid_condensed_cat = condensed_catalog_overlaid_data(dat.data_array, all_xs, all_ys, all_fs=all_fs, red_cut=red_cut, bands=[title_band_dict[band] for band in bands])
		all_overlaid_condensed_cat.savefig(condensed_cat_dir+'/condensed_catalog_red_cut_overlaid_data_allbands_maskhwhm='+str(mask_hwhm)+'.'+plttype, bbox_inches='tight', dpi=dpi)



	# ------------------- SOURCE NUMBER ---------------------------

	if source_number_plots:

		nsrc_fov_truth = None
		if truth_catalog is not None:
			nsrc_fov_truth = len(truth_catalog)

		f_nsrc = plot_src_number_posterior(nsrc_fov, nsrc_truth=nsrc_fov_truth, fmin=1e3*gdat.trueminf, units='mJy')
		f_nsrc.savefig(gdat.result_path+'/'+gdat.timestr+'/posterior_histogram_nstar.'+plttype, bbox_inches='tight', dpi=dpi)

		f_nsrc_trace = plot_src_number_trace(nsrc_fov)
		f_nsrc_trace.savefig(gdat.result_path+'/'+gdat.timestr+'/nstar_traceplot.'+plttype, bbox_inches='tight', dpi=dpi)


		nsrc_full = []
		for i, j in enumerate(np.arange(0, gdat.nsamp)):
		
			fsrcs_full = np.array([fsrcs[0][j][k] for k in range(nsrcs[j]) if dat.weights[0][int(ysrcs[j][k]),int(xsrcs[j][k])] != 0.])

			nsrc_full.append(len(fsrcs_full))

		f_nsrc_trace_full = plot_src_number_trace(nsrc_full)
		f_nsrc_trace_full.savefig(gdat.result_path+'/'+gdat.timestr+'/nstar_traceplot_full.'+plttype, bbox_inches='tight', dpi=dpi)

		











