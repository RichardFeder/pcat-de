from fast_astrom import *
import config
import numpy as np
from astropy.convolution import Gaussian2DKernel
from image_eval import psf_poly_fit
import pickle
import matplotlib
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy import units as u
from astropy.io import fits



class objectview(object):
	def __init__(self, d):
		self.__dict__ = d

def verbprint(verbose, text, file=None, verbthresh=None):
	"""
	This is a verbosity dependent print function used throughout the code, useful for troubleshooting/debugging.

	Parameters
	----------
	verbose : int
		Verbosity level
	text : str
		Text to be printed.
	verbthresh : 

	"""
	if verbthresh is not None:
		if verbose > verbthresh:
			print(text)
	else:
		if verbose:
			print(text)


def get_gaussian_psf_template(pixel_fwhm = 3., nbin=5, psf_post_width=25):
	"""
	Computes Gaussian PSF kernel for fast model evaluation with lion

	Parameters
	----------
	pixel_fwhm : float
		Full width at half-maximum (FWHM) of PSF in units of pixels.
		Default is 3 (for SPIRE maps).
	nbin : int
		Upsampling factor for sub-pixel interpolation method in Lion
		Default is 5.
	psf_post_width : int
		Width of PSF postage stamp in original resolution map. Default is 25.

	Returns
	-------
	psfnew : `~numpy.ndarray`
		Sum normalized PSF template up sampled by factor nbin in both x and y
	cf : `~numpy.ndarray`
		Coefficients of polynomial fit to upsampled PSF. These are used by lion
	nbin : int
		Upsampling factor 

	"""
	nc = nbin**2
	psfnew = Gaussian2DKernel((pixel_fwhm/2.355)*nbin, x_size=psf_post_width*nbin, y_size=psf_post_width*nbin).array.astype(np.float32)
	cf = psf_poly_fit(psfnew, nbin=nbin)
	return psfnew, cf, nc, nbin



def load_in_map(gdat, file_path=None, band=0, astrom=None, show_input_maps=False, image_extnames=['SIGNAL'], err_hdu_idx=1):

	""" 
	This function does some of the initial data parsing needed in constructing the pcat_data object.
	Note that if gdat.add_noise is True and gdat.scalar_noise_sigma is not None, this script will add Gaussian noise *on top* of 
	the existing noise model if gdat.use_uncertainty_map is True. The uncertainty map is then increased accordingly.


	Parameters
	----------

	gdat : global object
		This is a super object which is used throughout different parts of PCAT. 
		Contains data configuration information

	band : int, optional
		Index of bandpass. Convention for SPIRE is 0->250um, 1->350um and 2->500um.
		Default is 0.

	astrom : object of type 'fast_astrom', optional
		astrometry object that can be loaded along with other data products
		Default is 'None'.

	show_input_maps : bool, optional. If True, shows maps as they are read in, cropped, combined to get observed dataset.
		Default is 'False'.

	image_extnames : list of strings, optional
		If list of extensions is specified, observed data will be combination of extension images.
		For example, one can test mock unlensed data with noise by including ['UNLENSED', 'NOISE'] 
		or mock lensed data with noise ['LENSED', 'NOISE']. Default is ['SIGNAL'].

	Returns
	-------

	image : '~numpy.ndarray'
		Observed data for PCAT. Can be composed of several mock components, or it can be real observed data.

	uncertainty_map : '~numpy.ndarray' of type 'float'.
		Noise model for field used when calculating likelihoods within PCAT. Masked pixels will have
		NaN values, which are modified to zeros in this script.

	mask : '~numpy.ndarray'
		Numpy array which contains mask used on data. 
		Masked regions are cropped out/down-weighted when preparing data for PCAT.

	file_path : str
		File path for map that is loaded in.

	"""

	if file_path is None:

		if gdat.file_path is None:
			file_path = config.data_path+gdat.dataname+'/'+gdat.tail_name+'.fits'
		else:
			file_path = gdat.file_path

	file_path = file_path.replace('PSW', 'P'+str(gdat.band_dict[band])+'W')
	verbprint(gdat.verbtype, 'Band is '+str(gdat.band_dict[band]), verbthresh=1)
	verbprint(gdat.verbtype, 'file path is '+str(file_path), verbthresh=1)

	if astrom is not None:
		verbprint(gdat.verbtype, 'Loading from ', gdat.band_dict[band], verbthresh=1)
		astrom.load_wcs_header_and_dim(file_path, round_up_or_down=gdat.round_up_or_down)

	#by loading in the image this way, we can compose maps from several components, e.g. noiseless CIB + noise realization
	
	if gdat.im_fpath is None:
		spire_dat = fits.open(file_path)
	else:
		spire_dat = fits.open(gdat.im_fpath)

	for e, extname in enumerate(image_extnames):
		if e==0:
			image = np.nan_to_num(spire_dat[extname].data)
		else:
			image += np.nan_to_num(spire_dat[extname].data)

		if gdat.show_input_maps:
			plot_single_map(image, title=extname, lopct=5, hipct=95)

	if not gdat.use_uncertainty_map:
		uncertainty_map = np.zeros_like(image)
	elif gdat.err_fpath is None:
		uncertainty_map = np.nan_to_num(spire_dat[gdat.uncertainty_map_extname].data)
	else:
		hdu = fits.open(gdat.err_fpath)[err_hdu_idx]
		uncertainty_map = np.nan_to_num(hdu.data)

	# main functionality is following eight lines
	if gdat.use_mask:
		if gdat.mask_file is not None:
			mask_fpath = gdat.mask_file.replace('PSW', 'P'+str(gdat.band_dict[band])+'W')
			mask = fits.open(mask_fpath)[0].data
		else:
			mask = spire_dat['MASK'].data
	else:
		print('Not using mask..')
		mask = np.ones_like(image)

	if gdat.add_noise:
		verbprint(gdat.verbtype, 'Adding Gaussian noise..')
		if gdat.scalar_noise_sigma is not None:
			if type(gdat.scalar_noise_sigma)==float:
				noise_sig = gdat.scalar_noise_sigma
			else:
				noise_sig = gdat.scalar_noise_sigma[band]

			noise_realization = np.random.normal(0, noise_sig, image.shape)
			image += noise_realization
			
			if not gdat.use_uncertainty_map:
				uncertainty_map = noise_sig*np.ones((image.shape[0], image.shape[1]))
			else:
				old_unc = uncertainty_map.copy()
				old_variance = old_unc**2
				old_variance[uncertainty_map != 0] += noise_sig**2
				uncertainty_map = np.sqrt(old_variance)

			if gdat.show_input_maps:
				plot_multipanel([noise_realization, showim, uncertainty_map], ['Noise realization', 'Image + Gaussian noise', 'Uncertainty map'], figsize=(15, 5), \
					lopct=5, hipct=95)

		else:
			print('Using uncertainty map to generate noise realization, assuming Gaussian distributed..')
			noise_realization = np.zeros_like(uncertainty_map)
			for rowidx in range(uncertainty_map.shape[0]):
				for colidx in range(uncertainty_map.shape[1]):
					if not np.isnan(uncertainty_map[rowidx,colidx]):
						noise_realization[rowidx,colidx] = np.random.normal(0, uncertainty_map[rowidx,colidx])
			if gdat.show_input_maps:
				plot_single_map(noise_realization, title='Noise realization', lopct=5, hipct=95)
			image += noise_realization

	if gdat.show_input_maps:
		plot_multipanel([image, uncertainty_map, mask], ['Data', 'Uncertainty map', 'Mask'], figsize=(12, 4), lopct=[5, 5, None], hipct=[95, 95, None])

	return image, uncertainty_map, mask, file_path



def load_param_dict(timestr=None, result_path=None, encoding=None):
	
	"""
	Loads dictionary of configuration parameters from prior run of PCAT.

	Parameters
	----------

	timestr : string
		time string associated with desired PCAT run.

	result_path : string, optional
		file location of PCAT run results.
		Default is '/Users/luminatech/Documents/multiband_pcat/spire_results/'.

	Returns
	-------

	opt : object containing parameter dictionary

	filepath : string
		file path of parameter file

	result_path : string
		file location of PCAT run results. not sure why its here

	"""

	if result_path is None:
		result_path = config.result_path
	filepath = result_path
	if timestr is not None:
		filepath += timestr
	filen = open(filepath+'/params.txt','rb')
	if encoding is not None:
		pdict = pickle.load(filen, encoding=encoding)
	else:
		pdict = pickle.load(filen)
	opt = objectview(pdict) 

	return opt, filepath, result_path



def get_rect_mask_bounds(mask):

	"""
	This function assumes the mask is rectangular in shape, with ones in the desired region and zero otherwise.
	
	Parameters
	----------
	mask : 'np.array' of type 'int'. Mask

	Returns
	-------
	bounds : 'np.array' of type 'float' and shape (2, 2). x and y bounds for mask.

	"""

	idxs = np.argwhere(mask == 1.0)
	bounds = np.array([[np.min(idxs[:,0]), np.max(idxs[:,0])], [np.min(idxs[:,1]), np.max(idxs[:,1])]])

	return bounds


class pcat_data():

	""" 
	This class sets up the data structures for data/data-related information. 
	
		- load_in_data() loads in data, generates the PSF template and computes weights from the noise model

	"""

	def __init__(self, gdat, auto_resize=False, nregion=1):

		self.ncs, self.nbins, self.psfs, self.cfs, self.biases, self.data_array, self.weights, self.masks, self.uncertainty_maps, \
			self.fracs, self.template_array = [[] for x in range(11)]

		self.fast_astrom = wcs_astrometry(auto_resize, nregion=nregion)

		self.gdat = gdat

		gdat.regsizes, gdat.margins, gdat.bounds,\
		 
		self.gdat.imszs, self.gdat.imszs_orig = np.zeros((self.gdat.nbands, 2))
		self.gdat.x_max_pivot_list, self.gdat.x_max_pivot_list = [np.zeros((self.gdat.nbands)) for x in range(2)]
		self.gdat.bounds = np.zeros((self.gdat.nbands, 2, 2))

	# def load_in_data(self, gdat, map_object=None, tail_name=None, show_input_maps=False, \
	# 	temp_mock_amps_dict=None, sb_conversion_dict=None, sed_cirrus=None, bands=None:
	
	# def load_in_data(self, tail_name=None, show_input_maps=False, \
	# 	temp_mock_amps_dict=None, sb_conversion_dict=None, sed_cirrus=None, bands=None:

	# 	"""
	# 	This function does the heavy lifting for parsing input data products and setting up variables in pcat_data class. At some point, template section should be cleaned up.
	# 	This is an in place operation. 

	# 	Parameters
	# 	----------

	# 	gdat : Global data object. 

	# 	tail_name (optional) : 
	# 			Default is 'None'.
	# 	show_input_maps (optional) : 'boolean'.
	# 			Default is 'False'.
	# 	temp_mock_amps_dict (optional) :  'dictionary' of floats.
	# 			Default is 'None'.
	# 	sb_conversion_dict (optional) : 'dictionary' of floats.
	# 			Default is 'None'. 

	# 	"""

	# 	if bands is None:
	# 		print('bands is None, using config.spire_bands:', config.spire_bands)
	# 		bands = config.spire_bands

	# 	# these conversion factors are necessary to convert the SPIRE data from MJy/sr to peak-normalized Jansky/beam. 
	# 	if sb_conversion_dict is None:
	# 		sb_conversion_dict = config.sb_conversion_dict

	# 	if temp_mock_amps_dict is None:
	# 		temp_mock_amps_dict = config.temp_mock_amps_dict

	# 	# if sed_cirrus is None:
	# 	# 	print('No cirrus SED provided, using default values..')
	# 	# 	sed_cirrus = config.sed_cirrus

	# 	# gdat.imszs, gdat.regsizes, gdat.margins, gdat.bounds,\
	# 	# 	 gdat.x_max_pivot_list, gdat.y_max_pivot_list, gdat.imszs_orig = [[] for x in range(7)]

	# 	for b, band in enumerate(self.gdat.bands):
	# 		image, uncertainty_map, mask, file_name = load_in_map(self.gdat, band, astrom=self.fast_astrom, show_input_maps=self.gdat.show_input_maps, image_extnames=self.gdat.image_extnames)
	# 		imshp = image.shape 
	# 		self.gdat.imszs_orig[b,:] = np.array([imshp[0], imshp[1]])
	# 		self.gdat.bounds[b,:,:] = np.array([[0, imshp[0]], [0, imshp[1]])

	# 		# gdat.imszs_orig.append(np.array(imshp))
	# 		# bounds = [[0, image.shape[0]], [0, image.shape[1]]
	# 		# gdat.bounds.append(bounds)

	# 		# this part updates the fast astrometry information if cropping the images
	# 		# if bounds is not None:
	# 		# 	big_dim = np.maximum(find_nearest_mod(bounds[0,1]-bounds[0,0]+1, gdat.nregion), find_nearest_mod(bounds[1,1]-bounds[1,0]+1, gdat.nregion))
	# 		# 	self.fast_astrom.dims[b] = (big_dim, big_dim)

	# 		template_list = [] 

	# 		print('gdat.template_name is ', self.gdat.template_filename)
	# 		if self.gdat.n_templates > 0:
	# 			for t, template_name in enumerate(self.gdat.template_names):

	# 				verbprint(self.gdat.verbtype, 'template name is ', template_name, verbthresh=1)
	# 				if self.gdat.band_dict[band] in config.template_bands_dict[template_name]:
	# 					verbprint(self.gdat.verbtype, 'Band, template band, lambda: '+str(self.gdat.band_dict[band])+', '+str(config.template_bands_dict[template_name])+', '+str(self.gdat.lam_dict[self.gdat.band_dict[band]]), verbthresh=1)

	# 					if self.gdat.template_filename is not None and template_name=='sze':
	# 						temp_name = self.gdat.template_filename[template_name]

	# 						for loopband in config.spire_band_names:
	# 							if loopband in temp_name:
	# 								template_file_name = temp_name.replace(loopband,  'P'+str(self.gdat.band_dict[band])+'W')

	# 						print('Template file name for band '+str(self.gdat.band_dict[band])+' is '+template_file_name)
	# 						template = fits.open(template_file_name)[1].data 
	# 					else:
	# 						template = fits.open(file_name)[template_name].data

	# 					if show_input_maps:
	# 						plot_single_map(template, title=template_name)

	# 					if self.gdat.inject_sz_frac is not None and template_name=='sze':								
	# 						template_inject = self.gdat.inject_sz_frac*template*temp_mock_amps_dict[self.gdat.band_dict[band]]*sb_conversion_dict[self.gdat.band_dict[band]]
	# 						image += template_inject
	# 						if show_input_maps:
	# 							ampstr = 'Injected level:  '+str(np.round(self.gdat.inject_sz_frac*temp_mock_amps_dict[self.gdat.band_dict[band]]*sb_conversion_dict[self.gdat.band_dict[band]], 4))
	# 							plot_multipanel([template_inject, image], [ampstr, 'image + sz'], figsize=(8,4), cmap='Greys')

	# 				else:
	# 					print('no band in config.template_bands_dict')
	# 					template = None

	# 				template_list.append(template)


	# 		if b > 0:
	# 			verbprint(self.gdat.verbtype, 'Moving to band '+str(band)+'..', verbthresh=1)
	# 			self.fast_astrom.fit_astrom_arrays(0, b, bounds0=bounds[0], bounds1=bounds[b])

	# 			x_max_pivot, y_max_pivot = self.fast_astrom.transform_q(np.array([self.gdat.imsz0[0]]), np.array([self.gdat.imsz0[1]]), b-1)
	# 			print('xmaxpivot, ymaxpivot for band ', i, ' are ', x_max_pivot, y_max_pivot)
	# 			self.gdat.x_max_pivot_list.append(x_max_pivot)
	# 			self.gdat.y_max_pivot_list.append(y_max_pivot)
	# 		else:
	# 			self.gdat.x_max_pivot_list.append(big_dim)
	# 			self.gdat.y_max_pivot_list.append(big_dim)

	# 		if self.gdat.noise_thresholds is not None:
	# 			uncertainty_map[uncertainty_map > self.gdat.noise_thresholds[b]] = 0 # this equates to downweighting the pixels

	# 		uncertainty_map = uncertainty_map[bounds[0,0]:bounds[0,1], bounds[1,0]:bounds[1,1]]
	# 		image = image[bounds[0,0]:bounds[0,1], bounds[1,0]:bounds[1,1]]
			
	# 		# uncertainty_map = uncertainty_map[bounds[0,0]:bounds[0,1]+1, bounds[1,0]:bounds[1,1]+1] # bounds
	# 		# image = image[bounds[0,0]:bounds[0,1]+1, bounds[1,0]:bounds[1,1]+1]

	# 		smaller_dim, larger_dim = np.min(image.shape), np.max(image.shape)

	# 		self.gdat.width = find_nearest_mod(larger_dim, self.gdat.nregion)
	# 		self.gdat.height = self.gdat.width
	# 		image_size = (self.gdat.width, self.gdat.height)

	# 		resized_image = np.zeros(shape=image_size)
	# 		resized_unc = np.zeros(shape=image_size)
	# 		resized_mask = np.zeros(shape=image_size)

	# 		crop_size_x = np.minimum(self.gdat.width, image.shape[0])
	# 		crop_size_y = np.minimum(self.gdat.height, image.shape[1])
			
	# 		resized_image[:image.shape[0], :image.shape[1]] = image[:crop_size_x, :crop_size_y]
	# 		resized_unc[:image.shape[0], :image.shape[1]] = uncertainty_map[:crop_size_x, :crop_size_y]
	# 		resized_mask[:image.shape[0], :image.shape[1]] = mask[:crop_size_x, :crop_size_y]
	# 		resized_template_list = []

	# 		if b > 0 and int(x_max_pivot) < resized_image.shape[0]:
	# 			print('Setting pixels in band '+str(b)+' not in band 0 FOV to zero..')
	# 			resized_image[resized_mask==0] = 0.
	# 			resized_unc[resized_mask==0] = 0.
	# 			plot_single_map(resized_mask, title='resized mask')

	# 		for t, template in enumerate(template_list):
				
	# 			if template is not None:
	# 				resized_template = np.zeros(shape=image_size)

	# 				# template = template[bounds[0,0]:bounds[0,1]+1, bounds[1,0]:bounds[1,1]+1]

	# 				template = template[bounds[0,0]:bounds[0,1], bounds[1,0]:bounds[1,1]] # bounds
	# 				resized_template[:image.shape[0], : image.shape[1]] = template[:crop_size_x, :crop_size_y]

	# 				if show_input_maps:
	# 					plot_multipanel([resized_template, resized_image], ['Resized template, '+self.gdat.template_order[t], 'Image, '+self.gdat.template_order[t]], cmap='Greys')

	# 				if self.gdat.template_order[t] == 'dust' or self.gdat.template_order[t] == 'planck':
	# 					resized_template -= np.mean(resized_template)

	# 					if show_input_maps:
	# 						plot_single_map(resized_template, title=self.gdat.template_order[t]+', '+str(self.gdat.tail_name), lopct=5, hipct=95)

	# 					if self.gdat.inject_dust and template_name=='planck':
	# 						verbprint(self.gdat.verbtype, 'Injecting dust template into image..', verbthresh=1)
	# 						resized_image += resized_template

	# 						if show_input_maps:
	# 							f = plot_multipanel([resized_template, resized_image], ['Injected dust', 'Image + dust'])

	# 				resized_template_list.append(resized_template.astype(np.float32))
	# 			else:
	# 				resized_template_list.append(None)


	# 		if self.gdat.inject_diffuse_comp and self.gdat.diffuse_comp_path is not None:

	# 			diffuse_comp = np.load(self.gdat.diffuse_comp_path)[band] # specific to SPIRE files
	# 			if show_input_maps:
	# 				plot_single_map(diffuse_comp, title='Diffuse component, '+str(cropped_diffuse_comp.shape))

	# 			cropped_diffuse_comp = diffuse_comp[:self.gdat.width, :self.gdat.height]
	# 			if show_input_maps:
	# 				plot_single_map(cropped_diffuse_comp, title='Cropped diffuse component, '+str(cropped_diffuse_comp.shape))

	# 			resized_image += cropped_diffuse_comp
	# 			if show_input_maps:
	# 				plot_single_map(resized_image, title='Resized image with cropped diffuse comp')

	# 			self.injected_diffuse_comp.append(cropped_diffuse_comp.astype(np.float32))

	# 		variance = resized_unc**2
	# 		variance[variance==0.]=np.inf
	# 		weight = 1. / variance

	# 		self.weights.append(weight.astype(np.float32))
	# 		self.uncertainty_maps.append(resized_unc.astype(np.float32))
	# 		resized_image[weight==0] = 0.
	# 		self.data_array.append(resized_image.astype(np.float32)-self.gdat.mean_offsets[b]) # constant offset, will need to change
	# 		self.template_array.append(resized_template_list)

	# 		if i==0:
	# 			self.gdat.imsz0 = image_size

	# 		if show_input_maps:
	# 			plot_single_map(self.data_array[b], title='Data, '+self.gdat.tail_name, lopct=5, hipct=95)
	# 			plot_single_map(self.uncertainty_maps[b], title='Uncertainty map, '+self.gdat.tail_name, lopct=5, hipct=95)


	# 		self.gdat.imszs.append(image_size)
	# 		self.gdat.regsizes.append(image_size[0]/gdat.nregion)
	# 		self.gdat.frac = np.count_nonzero(weight)/float(self.gdat.width*self.gdat.height)
	# 		psf, cf, nc, nbin = get_gaussian_psf_template(pixel_fwhm=self.gdat.psf_fwhms[b]) # variable psf pixel fwhm

	# 		verbprint(self.gdat.verbtype, 'Image maximum is '+str(np.max(self.data_array[0]))+', gdat.frac = '+str(self.gdat.frac)+', sum of PSF is '+str(np.sum(psf)), verbthresh=1)

	# 		self.psfs.append(psf)
	# 		self.cfs.append(cf)
	# 		self.ncs.append(nc)
	# 		self.nbins.append(nbin)
	# 		self.fracs.append(self.gdat.frac)

	# 	# lets confirm that the final processed images respect the sub-region dimensions
	# 	self.gdat.regions_factor = 1./float(self.gdat.nregion**2)
	# 	assert self.gdat.imsz0[0] % self.gdat.regsizes[0] == 0 
	# 	assert self.gdat.imsz0[1] % self.gdat.regsizes[0] == 0 

	# 	pixel_variance = np.median(self.uncertainty_maps[0]**2)

	# 	verbprint(self.gdat.verbtype, 'pixel_variance:'+str(pixel_variance), verbthresh=1)
	# 	verbprint(self.gdat.verbtype, 'self.dat.fracs:'+str(self.fracs), verbthresh=1)

	# 	self.gdat.err_f = np.sqrt(self.gdat.N_eff * pixel_variance)/self.gdat.err_f_divfac



	def load_in_data_new(self, tail_name=None, show_input_maps=False, \
		temp_mock_amps_dict=None, sb_conversion_dict=None, sed_cirrus=None, bands=None):

		"""
		This function does the heavy lifting for parsing input data products and setting up variables in pcat_data class. At some point, template section should be cleaned up.
		This is an in place operation. 

		Parameters
		----------

		gdat : Global data object. 

		tail_name (optional) : 
				Default is 'None'.
		show_input_maps (optional) : 'boolean'.
				Default is 'False'.
		temp_mock_amps_dict (optional) :  'dictionary' of floats.
				Default is 'None'.
		sb_conversion_dict (optional) : 'dictionary' of floats.
				Default is 'None'. 

		"""

		# these conversion factors are necessary to convert the SPIRE data from MJy/sr to peak-normalized Jansky/beam. 
		if sb_conversion_dict is None:
			sb_conversion_dict = config.sb_conversion_dict

		if temp_mock_amps_dict is None:
			temp_mock_amps_dict = config.temp_mock_amps_dict

		for b, band in enumerate(self.gdat.bands):
			image, uncertainty_map, mask, file_name = load_in_map(self.gdat, band, astrom=self.fast_astrom, show_input_maps=self.gdat.show_input_maps, image_extnames=self.gdat.image_extnames)
			imshp = image.shape 
			self.gdat.imszs_orig[b,:] = np.array([imshp[0], imshp[1]])
			self.gdat.bounds[b,:,:] = np.array([[0, imshp[0]], [0, imshp[1]])

			# gdat.imszs_orig.append(np.array(imshp))
			# bounds = [[0, image.shape[0]], [0, image.shape[1]]
			# gdat.bounds.append(bounds)

			# this part updates the fast astrometry information if cropping the images
			# if bounds is not None:
			# 	big_dim = np.maximum(find_nearest_mod(bounds[0,1]-bounds[0,0]+1, gdat.nregion), find_nearest_mod(bounds[1,1]-bounds[1,0]+1, gdat.nregion))
			# 	self.fast_astrom.dims[b] = (big_dim, big_dim)

			template_list = [] 

			print('gdat.template_name is ', self.gdat.template_filename)
			if self.gdat.n_templates > 0:
				for t, template_name in enumerate(self.gdat.template_names):

					verbprint(self.gdat.verbtype, 'template name is ', template_name, verbthresh=1)
					if self.gdat.band_dict[band] in config.template_bands_dict[template_name]:
						verbprint(self.gdat.verbtype, 'Band, template band, lambda: '+str(self.gdat.band_dict[band])+', '+str(config.template_bands_dict[template_name])+', '+str(self.gdat.lam_dict[self.gdat.band_dict[band]]), verbthresh=1)

						if self.gdat.template_filename is not None and template_name=='sze':
							temp_name = self.gdat.template_filename[template_name]

							for loopband in config.spire_band_names:
								if loopband in temp_name:
									template_file_name = temp_name.replace(loopband,  'P'+str(self.gdat.band_dict[band])+'W')

							print('Template file name for band '+str(self.gdat.band_dict[band])+' is '+template_file_name)
							template = fits.open(template_file_name)[1].data 
						else:
							template = fits.open(file_name)[template_name].data

						if show_input_maps:
							plot_single_map(template, title=template_name)

						if self.gdat.inject_sz_frac is not None and template_name=='sze':								
							template_inject = self.gdat.inject_sz_frac*template*temp_mock_amps_dict[self.gdat.band_dict[band]]*sb_conversion_dict[self.gdat.band_dict[band]]
							image += template_inject
							if show_input_maps:
								ampstr = 'Injected level:  '+str(np.round(self.gdat.inject_sz_frac*temp_mock_amps_dict[self.gdat.band_dict[band]]*sb_conversion_dict[self.gdat.band_dict[band]], 4))
								plot_multipanel([template_inject, image], [ampstr, 'image + sz'], figsize=(8,4), cmap='Greys')

					else:
						print('no band in config.template_bands_dict')
						template = None

					template_list.append(template)


			if b > 0:
				verbprint(self.gdat.verbtype, 'Moving to band '+str(band)+'..', verbthresh=1)
				self.fast_astrom.fit_astrom_arrays(0, b, bounds0=bounds[0], bounds1=bounds[b])

				x_max_pivot, y_max_pivot = self.fast_astrom.transform_q(np.array([self.gdat.imsz0[0]]), np.array([self.gdat.imsz0[1]]), b-1)
				print('xmaxpivot, ymaxpivot for band ', i, ' are ', x_max_pivot, y_max_pivot)
				self.gdat.x_max_pivot_list.append(x_max_pivot)
				self.gdat.y_max_pivot_list.append(y_max_pivot)
			else:
				self.gdat.x_max_pivot_list.append(big_dim)
				self.gdat.y_max_pivot_list.append(big_dim)

			if self.gdat.noise_thresholds is not None:
				uncertainty_map[uncertainty_map > self.gdat.noise_thresholds[b]] = 0 # this equates to downweighting the pixels

			uncertainty_map = uncertainty_map[bounds[0,0]:bounds[0,1], bounds[1,0]:bounds[1,1]]
			image = image[bounds[0,0]:bounds[0,1], bounds[1,0]:bounds[1,1]]
			
			# uncertainty_map = uncertainty_map[bounds[0,0]:bounds[0,1]+1, bounds[1,0]:bounds[1,1]+1] # bounds
			# image = image[bounds[0,0]:bounds[0,1]+1, bounds[1,0]:bounds[1,1]+1]

			smaller_dim, larger_dim = np.min(image.shape), np.max(image.shape)

			self.gdat.width = find_nearest_mod(larger_dim, self.gdat.nregion)
			self.gdat.height = self.gdat.width
			image_size = (self.gdat.width, self.gdat.height)

			resized_image = np.zeros(shape=image_size)
			resized_unc = np.zeros(shape=image_size)
			resized_mask = np.zeros(shape=image_size)

			crop_size_x = np.minimum(self.gdat.width, image.shape[0])
			crop_size_y = np.minimum(self.gdat.height, image.shape[1])
			
			resized_image[:image.shape[0], :image.shape[1]] = image[:crop_size_x, :crop_size_y]
			resized_unc[:image.shape[0], :image.shape[1]] = uncertainty_map[:crop_size_x, :crop_size_y]
			resized_mask[:image.shape[0], :image.shape[1]] = mask[:crop_size_x, :crop_size_y]
			resized_template_list = []

			if b > 0 and int(x_max_pivot) < resized_image.shape[0]:
				print('Setting pixels in band '+str(b)+' not in band 0 FOV to zero..')
				resized_image[resized_mask==0] = 0.
				resized_unc[resized_mask==0] = 0.
				plot_single_map(resized_mask, title='resized mask')

			for t, template in enumerate(template_list):
				
				if template is not None:
					resized_template = np.zeros(shape=image_size)

					# template = template[bounds[0,0]:bounds[0,1]+1, bounds[1,0]:bounds[1,1]+1]

					template = template[bounds[0,0]:bounds[0,1], bounds[1,0]:bounds[1,1]] # bounds
					resized_template[:image.shape[0], : image.shape[1]] = template[:crop_size_x, :crop_size_y]

					if show_input_maps:
						plot_multipanel([resized_template, resized_image], ['Resized template, '+self.gdat.template_order[t], 'Image, '+self.gdat.template_order[t]], cmap='Greys')

					if self.gdat.template_order[t] == 'dust' or self.gdat.template_order[t] == 'planck':
						resized_template -= np.mean(resized_template)

						if show_input_maps:
							plot_single_map(resized_template, title=self.gdat.template_order[t]+', '+str(self.gdat.tail_name), lopct=5, hipct=95)

						if self.gdat.inject_dust and template_name=='planck':
							verbprint(self.gdat.verbtype, 'Injecting dust template into image..', verbthresh=1)
							resized_image += resized_template

							if show_input_maps:
								f = plot_multipanel([resized_template, resized_image], ['Injected dust', 'Image + dust'])

					resized_template_list.append(resized_template.astype(np.float32))
				else:
					resized_template_list.append(None)


			# if self.gdat.diffuse_comp_path is not None:
				# diffuse_comp = np.load(self.gdat.diffuse_comp_path)[band] # specific to SPIRE files

			if self.gdat.diffuse_comp_fpaths is not None:
				diffuse_comp = np.load(self.gdat.diffuse_comp_fpaths)[band] # specific to SPIRE files
				cropped_diffuse_comp = diffuse_comp[:self.gdat.width, :self.gdat.height]
				resized_image += cropped_diffuse_comp

				if show_input_maps:
					plot_single_map(diffuse_comp, title='Diffuse component, '+str(cropped_diffuse_comp.shape))
					plot_single_map(cropped_diffuse_comp, title='Cropped diffuse component, '+str(cropped_diffuse_comp.shape))
					plot_single_map(resized_image, title='Resized image with cropped diffuse comp')

				self.injected_diffuse_comp.append(cropped_diffuse_comp.astype(np.float32))

			variance = resized_unc**2
			variance[variance==0.]=np.inf
			weight = 1. / variance

			self.weights.append(weight.astype(np.float32))
			self.uncertainty_maps.append(resized_unc.astype(np.float32))
			resized_image[weight==0] = 0.
			self.data_array.append(resized_image.astype(np.float32)-self.gdat.mean_offsets[b]) # constant offset, will need to change
			self.template_array.append(resized_template_list)

			if i==0:
				self.gdat.imsz0 = image_size

			if show_input_maps:
				plot_single_map(self.data_array[b], title='Data, '+self.gdat.tail_name, lopct=5, hipct=95)
				plot_single_map(self.uncertainty_maps[b], title='Uncertainty map, '+self.gdat.tail_name, lopct=5, hipct=95)


			self.gdat.imszs[b,:,:] = np.array(image_size)
			self.gdat.regsizes[b] = image_size[0]/gdat.nregion


			self.gdat.frac = np.count_nonzero(weight)/float(self.gdat.width*self.gdat.height)
			psf, cf, nc, nbin = get_gaussian_psf_template(pixel_fwhm=self.gdat.psf_fwhms[b]) # variable psf pixel fwhm

			verbprint(self.gdat.verbtype, 'Image maximum is '+str(np.max(self.data_array[0]))+', gdat.frac = '+str(self.gdat.frac)+', sum of PSF is '+str(np.sum(psf)), verbthresh=1)

			self.psfs.append(psf)
			self.cfs.append(cf)
			self.ncs.append(nc)
			self.nbins.append(nbin)
			self.fracs.append(self.gdat.frac)

			assert image_size[0] % self.gdat.regsizes[b] == 0 
			assert image_size[1] % self.gdat.regsizes[b] == 0 

		pixel_variance = np.median(self.uncertainty_maps[0]**2)

		verbprint(self.gdat.verbtype, 'pixel_variance:'+str(pixel_variance), verbthresh=1)
		verbprint(self.gdat.verbtype, 'self.dat.fracs:'+str(self.fracs), verbthresh=1)

		self.gdat.err_f = np.sqrt(self.gdat.N_eff * pixel_variance)/self.gdat.err_f_divfac










