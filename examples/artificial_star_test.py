import numpy as np

from pcat_main import *
import config





dataname = 'rxj1347'
bands = ['S']



tail_names = ['rxj1347_P'+band+'W_nr_1_ext' for band in bands]
data_fpaths = [config.data_dir+dataname+'/'+tail_name for tail_name in tail_names]
print('Will load data from data_fpaths:', data_fpaths)
# Planck extrapolated cirrus
cirrus_fpaths = [config.data_dir+'/mock_cirrus/mock_cirrus_P'+band+'W.fits' for band in bands]

panel_list = ['data0', 'data1', 'data2',  'residual0','residual1', 'residual2'] # get default for this instead of specifying

image_extnames = ['SIGNAL']
error_extname = 'ERROR'

# instantiate the lion class

artif_star_params = dict({'nsrc_perbin':10, 'inject_fmin':0.01, 'inject_fmax':0.2, 'nbins':20, 'fluxbins':None, \
							'frac_flux_thresh':2., 'pos_thresh':0.5, 'inject_color_means':[1.0, 0.7], 'inject_color_sigs':[0.25, 0.25], \
							'nborder':3})


def make_inject_catalog(artif_star_params, save_fpath=None):
	if type(artif_star_params['n_src_perbin'])==list:
		nsrc_inject_total = int(np.sum(artif_star_params['n_src_perbin']))
	else:
		nsrc_inject_total = int(artif_star_params['n_src_perbin']*(artif_star_params['nbins']-1))

	catalog_inject = np.zeros((nsrc_inject_total, 2+artif_star_params['nbands'])).astype(np.float)
	
	catalog_inject[:, :2] = np.random.uniform(artif_star_params['nborder'], ob.data.data_array[0].shape[0]-5., (catalog_inject.shape[0], 2))

	for f in range(len(fluxbins)):
		if type(n_src_perbin)==list:
				print('here')

			pivot_fluxes = np.array(np.random.uniform(fluxbins[f], fluxbins[f+1], n_src_perbin[f]), dtype=np.float32)
				catalog_inject[idxctr:idxctr+int(n_src_perbin[f]), 2] = pivot_fluxes
				
				# if multiband, draw colors from prior and multiply pivot band fluxes
				if nbands > 0:
					for b in range(nbands - 1):
						colors = np.random.normal(inject_color_means[b], inject_color_sigs[b], n_src_perbin[f])
						
						print('injected sources in band ', b+1, 'are ')
						print(pivot_fluxes*colors)
						catalog_inject[idxctr:idxctr+int(n_src_perbin[f]), 3+b] = pivot_fluxes*colors

				idxctr += int(n_src_perbin[f])
		else:
			pivot_fluxes = np.array(np.random.uniform(fluxbins[f], fluxbins[f+1], n_src_perbin), dtype=np.float32)

			catalog_inject[f*n_src_perbin:(f+1)*n_src_perbin, 2] = pivot_fluxes
			if nbands > 0:
				for b in range(nbands - 1):
					colors = np.random.normal(color_means[b], color_sigs[b], n_src_perbin)
					catalog_inject[f*n_src_perbin:(f+1)*n_src_perbin, 3+b] = pivot_fluxes*colors

	if save_fpath is not None:
		print('Saving to '+save_fpath+'..')
		np.savetxt(save_fpath, catalog_inject, comments='x y fluxes')


	return catalog_inject

	

def artificial_star_test(pcat_obj, artif_star_params, catalog_inject=None, inject_catalog_path=None, nsrc_perbin=10, \
	load_timestr=None):
	''' This script is for artificial star tests..

	This will assume the object pcat_obj already has data loaded into it.

	Either provide catalog_inject in form from make_inject_catalog(), i.e., x, y , fluxes, 
	or through inject_catalog_path which is the saved output of make_inject_catalog(). 

	'''

	if artif_star_params['fluxbins'] is None:
		artif_star_params['fluxbins'] = np.logspace(np.log10(artif_star_params['inject_fmin']), np.log10(artif_star_params['inject_fmax']), artif_star_params['nbins']+1)

	if catalog_inject is None:
		if inject_catalog_path is not None:
			print('loading catalog from ', inject_catalog_path)
			catalog_inject = np.load(inject_catalog_path)['catalog_inject']

		else:
			artif_star_params['nbands'] = pcat_obj.gdat.nbands
			artif_star_params['imszs'] = pcat_obj.imszs
			catalog_inject = make_inject_catalog(artif_star_params)

	print('catalog inject has shape ', catalog_inject.shape)

	flux_bin_idxs = [[np.where((catalog_inject[:,2+b] > fluxbins[i])&(catalog_inject[:,2+b] < fluxbins[i+1]))[0] for i in range(len(fluxbins)-1)] for b in range(nbands)]

	if load_timestr is None:
		inject_src_image = np.zeros_like(pcat_obj.data.data_array[0])

			ob.initialize_print_log()

			libmmult = ob.initialize_libmmult()

			initialize_c(ob.gdat, libmmult, cblas=ob.gdat.cblas)

			if ob.gdat.cblas:
				lib = libmmult.pcat_model_eval
			else:
				lib = libmmult.clib_eval_modl

			model = Model(ob.gdat, ob.data, libmmult)
			resid = ob.data.data_array[0].copy()

			resids = []

			for b in range(nbands):

				resid = ob.data.data_array[b].copy() # residual for zero image is data
				resids.append(resid)

			print('fluxes have shape ', np.array(catalog_inject[:,2:]).astype(np.float32()).shape)

			print('model pixel per beam:', model.pixel_per_beam)
			inject_src_images, diff2s, dt_transf = model.pcat_multiband_eval(catalog_inject[:,0].astype(np.float32()), catalog_inject[:,1].astype(np.float32()), np.array(catalog_inject[:,2:]).astype(np.float32()).transpose(),\
																 np.array([0. for x in range(nbands)]), ob.data.ncs, ob.data.cfs, weights=ob.data.weights, ref=resids, lib=libmmult.pcat_model_eval, beam_fac=model.pixel_per_beam)


			for b in range(nbands):
				print('models look like')
				plt.figure()
				plt.imshow(inject_src_images[b])
				plt.colorbar()
				plt.show()

			if show_input_maps:
				plot_multipanel(pcat_obj.)
				plt.figure()
				plt.subplot(1,2,1)
				plt.title('Original image')
				plt.imshow(ob.data.data_array[0], cmap='Greys', origin='lower', vmax=0.1)
				plt.colorbar()
				plt.subplot(1,2,2)
				plt.title('Injected source image')
				plt.imshow(inject_src_image, cmap='Greys', origin='lower')
				plt.colorbar()
				plt.show()

			# add the injected mdoel image into the real data
			for b in range(nbands):
				ob.data.data_array[b] += inject_src_images[b]

			if show_input_maps:
				plt.figure()
				plt.title('original + injected image')
				plt.imshow(ob.data.data_array[0], cmap='Greys', origin='lower', vmax=0.1)
				plt.colorbar()
				plt.show()

			# run the thing
			ob.main()
			# save the injected catalog to the result directory
			print(self.result_path+ob.gdat.timestr+'/inject_catalog.npz')
			np.savez(self.result_path+ob.gdat.timestr+'/inject_catalog.npz', catalog_inject=catalog_inject)



	print('About to begin PCAT run..')
	pcat_obj.main()




def artificial_star_test(self, bands, n_src_perbin=10, inject_fmin=0.01, inject_fmax=0.2, nbins=20, fluxbins=None, frac_flux_thresh=2., pos_thresh=0.5):

	nbands = len(bands)
	if load_timestr is not None:
		# if we are analyzing a run of Lion that is already finished, don't bother making a new directory structure when initializing PCAT
		save = False

	ob = lion(band0=band0, band1=band1, band2=band2, base_path=self.base_path, result_path=self.result_path, \
					float_background=float_background, burn_in_frac=0.75, \
	 				cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps, \
	 				tail_name=tail_name, dataname=dataname, bias=bias, max_nsrc=max_nsrc,\
	 				trueminf=fmin, nregion=5, make_post_plots=make_post_plots, nsamp=nsamp, use_mask=use_mask,\
	 				residual_samples=residual_samples, float_fourier_comps=float_fourier_comps, \
	 				n_fourier_terms=n_fc_terms, fc_sample_delay=fc_sample_delay, fourier_comp_moveweight=fourier_comp_moveweight,\
	 				alph=alph, dfc_prob=1.0, nsrc_init=nsrc_init, mask_file=mask_file, \
	 				point_src_delay=point_src_delay, n_frames=n_frames, image_extnames=image_extnames, fc_prop_alpha=fc_prop_alpha, \
	 				im_fpath=im_fpath, err_fpath=err_fpath, fc_amp_sig=fc_amp_sig, MP_order=MP_order, bkg_moore_penrose_inv=bkg_moore_penrose_inv, ridge_fac=ridge_fac, \
	 				save=save)

def artificial_star_test(self, n_src_perbin=10, inject_fmin=0.01, inject_fmax=0.2, nbins=20, fluxbins=None, frac_flux_thresh=2., pos_thresh=0.5):

		'''
		This function injects a population of artificial sources into a given image and provides diagnostics on what artificial sources PCAT recovers and to what accuracy fluxes are estimated.

		Unique parameters
		-----------------
		
		n_src_perbin : int, optional
			Number of injected sources per flux bin. Default is 10.

		inject_fmin : float, optional
			Minimum flux density of injected sources. Default is 0.01 [Jy].

		inject_fmax : float, optional
			Maximum flux density of injected sources. Default is 0.2 [Jy].

		nbins : int, optional
			Number of flux bins to create between inject_fmin and inject_fmax. Default is 20.

		fluxbins : list or `~numpy.ndarray', optional
			fluxbins can be specified if custom flux bins are desired. Default is None.

		frac_flux_thresh : float, optional
			When matching PCAT sources to injected sources, frac_flux_thresh defines part of the cross-matching criteria, 
			in which the fractional error of the PCAT source w.r.t. the injected source must be less than frac_flux_thresh. 
			Default is 2.0.

		pos_thresh : float, optional
			Same idea as frac_flux_thresh, but applies a position cross matching criterion. 
			Default is 0.5 [pixels].

		inject_color_means, inject_color_sigs : lists, optional
			Mean colors for injected sources. The colors of sources are drawn from a Gaussian distribution with mean injected_color_means[b] and  
			Default assumes bands are ordered as 250, 350, 500 micron, colors of S350/S250 = 1.0, S500/S200 = 0.7 and scatters of [0.25, 0.25]. 

		cond_cat_fpath : string, optional
			If PCAT has already been run and one wishes to use results from a condensed catalog, it can be specified here to bypass running PCAT again. 
			Default is 'None'.


		Returns 
		-------

		Nothing! 

		'''

		if nbands is None:
			nbands = 0
			if band0 is not None:
				nbands += 1
			if band1 is not None:
				nbands += 1
			if band2 is not None:
				nbands += 1

		if load_timestr is not None:
			# if we are analyzing a run of Lion that is already finished, don't bother making a new directory structure when initializing PCAT
			save = False

		ob = lion(band0=band0, band1=band1, band2=band2, base_path=self.base_path, result_path=self.result_path, round_up_or_down='down', \
						float_background=float_background, burn_in_frac=0.75, \
		 				cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps, \
		 				tail_name=tail_name, dataname=dataname, bias=bias, max_nsrc=max_nsrc,\
		 				trueminf=fmin, nregion=5, make_post_plots=make_post_plots, nsamp=nsamp, use_mask=use_mask,\
		 				residual_samples=residual_samples, float_fourier_comps=float_fourier_comps, \
		 				n_fourier_terms=n_fc_terms, fc_sample_delay=fc_sample_delay, fourier_comp_moveweight=fourier_comp_moveweight,\
		 				alph=alph, dfc_prob=1.0, nsrc_init=nsrc_init, mask_file=mask_file, \
		 				point_src_delay=point_src_delay, n_frames=n_frames, image_extnames=image_extnames, fc_prop_alpha=fc_prop_alpha, \
		 				im_fpath=im_fpath, err_fpath=err_fpath, fc_amp_sig=fc_amp_sig, MP_order=MP_order, bkg_moore_penrose_inv=bkg_moore_penrose_inv, ridge_fac=ridge_fac, \
		 				save=save)


		if fluxbins is None:
			fluxbins = np.logspace(np.log10(inject_fmin), np.log10(inject_fmax), nbins)

		if load_timestr is None:

			if inject_catalog_path is not None:
				print('loading catalog from ', inject_catalog_path)
				catalog_inject = np.load(inject_catalog_path)['catalog_inject']

			else:
				print('FLUXBINS ARE ', fluxbins)

				if type(n_src_perbin)==list:
					catalog_inject = np.zeros((np.sum(np.array(n_src_perbin)), 2+nbands)).astype(np.float32())
				else:
					catalog_inject = np.zeros((n_src_perbin*(nbins-1), 2+nbands)).astype(np.float32())

				print('catalog inject has shape ', catalog_inject.shape)

				catalog_inject[:, :2] = np.array(np.random.uniform(5., ob.data.data_array[0].shape[0]-5., (catalog_inject.shape[0], 2)), dtype=np.float32)

				idxctr = 0
				for f in range(len(fluxbins)-1):

					if type(n_src_perbin)==list:
						print('here')
						pivot_fluxes = np.array(np.random.uniform(fluxbins[f], fluxbins[f+1], n_src_perbin[f]), dtype=np.float32)
						catalog_inject[idxctr:idxctr+int(n_src_perbin[f]), 2] = pivot_fluxes
						
						# if multiband, draw colors from prior and multiply pivot band fluxes
						if nbands > 0:
							for b in range(nbands - 1):
								colors = np.random.normal(inject_color_means[b], inject_color_sigs[b], n_src_perbin[f])
								
								print('injected sources in band ', b+1, 'are ')
								print(pivot_fluxes*colors)
								catalog_inject[idxctr:idxctr+int(n_src_perbin[f]), 3+b] = pivot_fluxes*colors

						idxctr += int(n_src_perbin[f])
					else:
						pivot_fluxes = np.array(np.random.uniform(fluxbins[f], fluxbins[f+1], n_src_perbin), dtype=np.float32)

						catalog_inject[f*n_src_perbin:(f+1)*n_src_perbin, 2] = pivot_fluxes
						if nbands > 0:
							for b in range(nbands - 1):
								colors = np.random.normal(color_means[b], color_sigs[b], n_src_perbin)
								catalog_inject[f*n_src_perbin:(f+1)*n_src_perbin, 3+b] = pivot_fluxes*colors


			ob.gdat.catalog_inject = catalog_inject.copy()


		else:
			catalog_inject = np.load(self.result_path+load_timestr+'/inject_catalog.npz')['catalog_inject']
			ob.gdat.timestr = load_timestr
			print('ob.gdat.timestr is ', ob.gdat.timestr)


		flux_bin_idxs = [[np.where((catalog_inject[:,2+b] > fluxbins[i])&(catalog_inject[:,2+b] < fluxbins[i+1]))[0] for i in range(len(fluxbins)-1)] for b in range(nbands)]

		if load_timestr is None:
			inject_src_image = np.zeros_like(ob.data.data_array[0])

			ob.initialize_print_log()

			libmmult = ob.initialize_libmmult()

			initialize_c(ob.gdat, libmmult, cblas=ob.gdat.cblas)

			if ob.gdat.cblas:
				lib = libmmult.pcat_model_eval
			else:
				lib = libmmult.clib_eval_modl

			model = Model(ob.gdat, ob.data, libmmult)
			resid = ob.data.data_array[0].copy()

			resids = []

			for b in range(nbands):

				resid = ob.data.data_array[b].copy() # residual for zero image is data
				resids.append(resid)

			print('fluxes have shape ', np.array(catalog_inject[:,2:]).astype(np.float32()).shape)

			print('model pixel per beam:', model.pixel_per_beam)
			inject_src_images, diff2s, dt_transf = model.pcat_multiband_eval(catalog_inject[:,0].astype(np.float32()), catalog_inject[:,1].astype(np.float32()), np.array(catalog_inject[:,2:]).astype(np.float32()).transpose(),\
																 np.array([0. for x in range(nbands)]), ob.data.ncs, ob.data.cfs, weights=ob.data.weights, ref=resids, lib=libmmult.pcat_model_eval, beam_fac=model.pixel_per_beam)


			for b in range(nbands):
				print('models look like')
				plt.figure()
				plt.imshow(inject_src_images[b])
				plt.colorbar()
				plt.show()

			if show_input_maps:
				plot_multipanel(pcat_obj.)
				plt.figure()
				plt.subplot(1,2,1)
				plt.title('Original image')
				plt.imshow(ob.data.data_array[0], cmap='Greys', origin='lower', vmax=0.1)
				plt.colorbar()
				plt.subplot(1,2,2)
				plt.title('Injected source image')
				plt.imshow(inject_src_image, cmap='Greys', origin='lower')
				plt.colorbar()
				plt.show()

			# add the injected mdoel image into the real data
			for b in range(nbands):
				ob.data.data_array[b] += inject_src_images[b]

			if show_input_maps:
				plt.figure()
				plt.title('original + injected image')
				plt.imshow(ob.data.data_array[0], cmap='Greys', origin='lower', vmax=0.1)
				plt.colorbar()
				plt.show()

			# run the thing
			ob.main()
			# save the injected catalog to the result directory
			print(self.result_path+ob.gdat.timestr+'/inject_catalog.npz')
			np.savez(self.result_path+ob.gdat.timestr+'/inject_catalog.npz', catalog_inject=catalog_inject)

		# load the run and compute the completeness for each artificial source

		if load_timestr is not None:
			_, filepath, _ = load_param_dict(load_timestr, result_path=self.result_path)
			timestr = load_timestr
		else:
			_, filepath, _ = load_param_dict(ob.gdat.timestr, result_path=self.result_path)
			timestr = ob.gdat.timestr

		chain = np.load(filepath+'/chain.npz')

		xsrcs = chain['x']
		ysrcs = chain['y']
		fsrcs = chain['f']

		completeness_ensemble = np.zeros((residual_samples, nbands, catalog_inject.shape[0]))
		fluxerror_ensemble = np.zeros((residual_samples, nbands, catalog_inject.shape[0]))
		recovered_flux_ensemble = np.zeros((residual_samples, nbands, catalog_inject.shape[0]))

		for b in range(nbands):

			for i in range(residual_samples):

				for j in range(catalog_inject.shape[0]):

					# make position cut 
					idx_pos = np.where(np.sqrt((xsrcs[-i] - catalog_inject[j,0])**2 +(ysrcs[-i] - catalog_inject[j,1])**2)  < pos_thresh)[0]
					
					fluxes_poscutpass = fsrcs[b][-i][idx_pos]
					# fluxes_poscutpass = fsrcs[0][-i][idx_pos]
					
					# make flux cut
					# mask_flux = np.where(np.abs(fluxes_poscutpass - catalog_inject[j,2])/catalog_inject[j,2] < frac_flux_thresh)[0]
					mask_flux = np.where(np.abs(fluxes_poscutpass - catalog_inject[j,2+b])/catalog_inject[j,2+b] < frac_flux_thresh)[0]

					if len(mask_flux) >= 1:

						# print('we got one! true source is ', catalog_inject[j])
						# print('while PCAT source is ', xsrcs[-i][idx_pos][mask_flux], ysrcs[i][idx_pos][mask_flux], fluxes_poscutpass[mask_flux])

						# completeness_ensemble[i,j] = 1.
						completeness_ensemble[i,b,j] = 1.

						# compute the relative difference in flux densities between the true and PCAT source and add to list specific for each source 
						# (in practice, a numpy.ndarray with zeros truncated after the fact). 
						# For a given injected source, PCAT may or may not have a detection, so one is probing the probability distribution P(S_{Truth, i} - S_{PCAT} | N_{PCAT, i} == 1)
						# where N_{PCAT, i} is an indicator variable for whether PCAT has a source within the desired cross match criteria. 

						# if there is more than one source satisfying the cross-match criteria, choose the brighter source

						flux_candidates = fluxes_poscutpass[mask_flux]

						brighter_flux = np.max(flux_candidates)
						dists = np.sqrt((xsrcs[-i][mask_flux] - catalog_inject[j,0])**2 + (ysrcs[-i][mask_flux] - catalog_inject[j,1])**2)


						mindist_idx = np.argmin(dists)					
						mindist_flux = flux_candidates[mindist_idx]

						recovered_flux_ensemble[i,b,j] = mindist_flux
						# fluxerror_ensemble[i,j] = brighter_flux/catalog_inject[j,2]
						# fluxerror_ensemble[i,j] = (brighter_flux-catalog_inject[j,2])/catalog_inject[j,2]
						# fluxerror_ensemble[i,j] = (mindist_flux-catalog_inject[j,2])/catalog_inject[j,2]
						
						# fluxerror_ensemble[i,b,j] = (mindist_flux-catalog_inject[j,2+b])/catalog_inject[j,2+b]

						fluxerror_ensemble[i,b,j] = (mindist_flux-catalog_inject[j,2+b])


		mean_frac_flux_error = np.zeros((catalog_inject.shape[0],nbands))
		pct_16_fracflux = np.zeros((catalog_inject.shape[0],nbands))
		pct_84_fracflux = np.zeros((catalog_inject.shape[0],nbands))


		mean_recover_flux = np.zeros((catalog_inject.shape[0],nbands))
		pct_16_recover_flux = np.zeros((catalog_inject.shape[0],nbands))
		pct_84_recover_flux = np.zeros((catalog_inject.shape[0],nbands))


		prevalences = [[] for x in range(nbands)]

		
		for b in range(nbands):
			for j in range(catalog_inject.shape[0]):
				nonzero_fidx = np.where(fluxerror_ensemble[:,b,j] != 0)[0]
				prevalences[b].append(float(len(nonzero_fidx))/float(residual_samples))
				if len(nonzero_fidx) > 0:
					mean_frac_flux_error[j,b] = np.median(fluxerror_ensemble[nonzero_fidx,b, j])
					pct_16_fracflux[j,b] = np.percentile(fluxerror_ensemble[nonzero_fidx,b, j], 16)
					pct_84_fracflux[j,b] = np.percentile(fluxerror_ensemble[nonzero_fidx,b, j], 84)

					mean_recover_flux[j,b] = np.median(recovered_flux_ensemble[nonzero_fidx, b, j])
					pct_16_recover_flux[j,b] = np.percentile(recovered_flux_ensemble[nonzero_fidx, b, j], 16)
					pct_84_recover_flux[j,b] = np.percentile(recovered_flux_ensemble[nonzero_fidx, b, j], 84)


		# nonzero_ferridx = np.where(mean_frac_flux_error != 0)[0]
		nonzero_ferridx = np.where(mean_frac_flux_error[:,0] != 0)[0]
		lamtitlestrs = ['PSW', 'PMW', 'PLW']

		plt_colors = ['b', 'g', 'r']
		plt.figure(figsize=(5*nbands, 5))

		for b in range(nbands):
			plt.subplot(1,nbands, b+1)
			yerr_recover = [1e3*mean_recover_flux[nonzero_ferridx, b] - 1e3*pct_16_recover_flux[nonzero_ferridx, b], 1e3*pct_84_recover_flux[nonzero_ferridx, b]-1e3*mean_recover_flux[nonzero_ferridx, b]]

			plt.title(lamtitlestrs[b], fontsize=18)
			print(1e3*mean_recover_flux[nonzero_ferridx, b])
			print(1e3*catalog_inject[nonzero_ferridx, 2+b])
			print(yerr_recover)

			plt.errorbar(1e3*catalog_inject[nonzero_ferridx, 2+b], 1e3*mean_recover_flux[nonzero_ferridx, b], yerr=yerr_recover, capsize=5, fmt='.', linewidth=2, markersize=10, alpha=0.25, color=plt_colors[b])

			plt.xscale('log')
			plt.yscale('log')

			if b==0:
				plt.xlabel('$S_{True}^{250}$ [mJy]', fontsize=16)
				plt.ylabel('$S_{Recover}^{250}$ [mJy]', fontsize=16)
			elif b==1:
				plt.xlabel('$S_{True}^{350}$ [mJy]', fontsize=16)
				plt.ylabel('$S_{Recover}^{350}$ [mJy]', fontsize=16)
			elif b==2:
				plt.xlabel('$S_{True}^{500}$ [mJy]', fontsize=16)
				plt.ylabel('$S_{Recover}^{500}$ [mJy]', fontsize=16)

			plt.plot(np.logspace(-1, 3, 100), np.logspace(-1, 3, 100), linestyle='dashed', color='k', linewidth=3)
			plt.xlim(1e-1, 1e3)
			plt.ylim(1e-1, 1e3)

			# plt.xlim(5, 700)
			# plt.ylim(5, 700)
			# plt.savefig(filepath+'/injected_vs_recovered_flux_band'+str(b)+'_PCAT_dx='+str(pos_thresh)+'_dS-S='+str(frac_flux_thresh)+'_mindist.pdf', bbox_inches='tight')
		plt.tight_layout()
		# plt.savefig(filepath+'/injected_vs_recovered_flux_threeband_PCAT_dx='+str(pos_thresh)+'_dS-S='+str(frac_flux_thresh)+'_mindist_4.pdf', bbox_inches='tight')
		# plt.savefig(filepath+'/injected_vs_recovered_flux_threeband_PCAT_dx='+str(pos_thresh)+'_dS-S='+str(frac_flux_thresh)+'_mindist_3.png', bbox_inches='tight', dpi=300)
			
		plt.show()


		prevalences = np.array(prevalences)


		mean_ferr_nonzero_ferridx = mean_frac_flux_error[nonzero_ferridx,:]
		pct_16_nonzero_ferridx = pct_16_fracflux[nonzero_ferridx,:]
		pct_84_nonzero_ferridx = pct_84_fracflux[nonzero_ferridx,:]

		yerrs = [mean_ferr_nonzero_ferridx - pct_16_nonzero_ferridx, pct_84_nonzero_ferridx-mean_ferr_nonzero_ferridx]

		mean_frac_flux_error_binned = np.zeros((len(fluxbins)-1, nbands))
		pct16_frac_flux_error_binned = np.zeros((len(fluxbins)-1, nbands))
		pct84_frac_flux_error_binned = np.zeros((len(fluxbins)-1, nbands))

		print('flux bins in function are ', fluxbins)

		for b in range(nbands):
			for f in range(len(fluxbins)-1):

				finbin = np.where((catalog_inject[nonzero_ferridx, 2+b] >= fluxbins[f])&(catalog_inject[nonzero_ferridx, 2+b] < fluxbins[f+1]))[0]
				
				if len(finbin)>0:
					mean_frac_flux_error_binned[f,b] = np.median(mean_ferr_nonzero_ferridx[finbin,b])

					print(mean_ferr_nonzero_ferridx.shape)
					print(mean_ferr_nonzero_ferridx[finbin,b])
					pct16_frac_flux_error_binned[f,b] = np.percentile(mean_ferr_nonzero_ferridx[finbin,b], 16)

					pct84_frac_flux_error_binned[f,b] = np.percentile(mean_ferr_nonzero_ferridx[finbin,b], 84)

		for b in range(nbands):
			nsrc_perfbin = [len(fbin_idxs) for fbin_idxs in flux_bin_idxs[b]]

			g, _, yerr, geom_mean = plot_fluxbias_vs_flux(1e3*mean_frac_flux_error_binned[:,b], 1e3*pct16_frac_flux_error_binned[:,b], 1e3*pct84_frac_flux_error_binned[:,b], fluxbins, \
				band=b, nsrc_perfbin=nsrc_perfbin, ylim=[-20, 20], load_jank_txts=False)
			
			g.savefig(filepath+'/fluxerr_vs_fluxdensity_band'+str(b)+'_dx='+str(pos_thresh)+'_dS-S='+str(frac_flux_thresh)+'_mindist_nonfrac_012021.png', bbox_inches='tight', dpi=300)

			# np.savez('goodsn_band'+str(b)+'_fluxbias_vs_flux_nbands='+str(nbands)+'_012021.npz', mean_frac_flux_error_binned=mean_frac_flux_error_binned[:,b], yerr=yerr, geom_mean=geom_mean)

	
		# completeness_vs_flux = None
		xstack = xsrcs[-20:,:].ravel()
		ystack = ysrcs[-20:,:].ravel()
		fstack = fsrcs[0][-20:,:].ravel()

		lamstrs = ['250', '350', '500']

		nsrc_perfbin_bands, cvf_stderr_bands, completeness_vs_flux_bands = [], [], []

		for b in range(nbands):

			avg_completeness = np.mean(completeness_ensemble[:,b,:], axis=0)
			std_completeness = np.std(completeness_ensemble[:,b,:], axis=0)

			nsrc_perfbin = [len(fbin_idxs) for fbin_idxs in flux_bin_idxs[b]]

			nsrc_perfbin_bands.append(nsrc_perfbin)
			completeness_vs_flux = [np.mean(avg_completeness[fbin_idxs]) for fbin_idxs in flux_bin_idxs[b]]

			print('completeness vs flux? its')
			print(completeness_vs_flux)
			cvf_std = [np.mean(std_completeness[fbin_idxs]) for fbin_idxs in flux_bin_idxs[b]]



			completeness_vs_flux_bands.append(completeness_vs_flux)

			cvf_stderr_bands.append(np.array(cvf_std)/np.sqrt(np.array(nsrc_perfbin)))

			print(fluxbins)
			print(completeness_vs_flux)

			colors = ['b', 'g', 'r']

		# psc_spire_cosmos_comp = pd.read_csv('~/Downloads/PSW_completeness.csv', header=None)
		# plt.plot(np.array(psc_spire_cosmos_comp[0]), np.array(psc_spire_cosmos_comp[1]), marker='.', markersize=10, label='SPIRE Point Source Catalog (2017) \n 250 $\\mu m$', color='k')
		
		f = plot_completeness_vs_flux(pos_thresh, frac_flux_thresh, fluxbins, completeness_vs_flux_bands, cvf_stderr=cvf_stderr_bands,\
										 image=91.*ob.data.data_array[0][:-5,:-5] - 91.*np.nanmean(ob.data.data_array[0][:-5,:-5]), xstack=xstack, ystack=ystack, fstack=fstack, \
										 catalog_inject = catalog_inject)


		f.savefig(filepath+'/completeness_vs_fluxdensity_dx='+str(pos_thresh)+'_dS-S='+str(frac_flux_thresh)+'_multiband_012321.png', bbox_inches='tight', dpi=300)


		return fluxbins, completeness_vs_flux, f

