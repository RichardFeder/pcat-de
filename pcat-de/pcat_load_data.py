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

from plotting_fns import *

class objectview(object):
	def __init__(self, d):
		self.__dict__ = d


def grab_map_by_bounds(bounds, orig_map):

	print('bounds is ', bounds)
	bounds = np.array(bounds).astype(int)
	bound_map = orig_map[bounds[0,0]:bounds[0,1], bounds[1,0]:bounds[1,1]]
	return bound_map

def calc_weights(uncertainty_map):
	variance_map = uncertainty_map**2
	variance_map[variance_map==0.]=np.inf
	variance_map[np.isnan(variance_map)] = np.inf
	weight_map = 1. / variance_map

	return variance_map, weight_map

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

	verbprint(gdat.verbtype, 'Band is '+str(gdat.band_dict[band]), verbthresh=1)
	verbprint(gdat.verbtype, 'file path is '+str(file_path), verbthresh=1)

	if astrom is not None:
		verbprint(gdat.verbtype, 'Loading from ', gdat.band_dict[band], verbthresh=1)
		astrom.load_wcs_header_and_dim(file_path, round_up_or_down=gdat.round_up_or_down)


	if file_path is None:
		if gdat.file_path is None:
			file_path = config.data_path+gdat.dataname+'/'+gdat.tail_name+'.fits'
		else:
			file_path = gdat.file_path
		file_path = file_path.replace('PSW', 'P'+str(gdat.band_dict[band])+'W')

	#by loading in the image this way, we can compose maps from several components, e.g. noiseless CIB + noise realization
	
	if file_path is not None:
		data_file = fits.open(file_path)
	elif gdat.im_fpath is not None:
		data_file = fits.open(gdat.im_fpath)

	for e, extname in enumerate(image_extnames):
		comp = np.nan_to_num(data_file[extname].data)
		if e==0:
			image = comp
			extnamestr = extname
			extnameplotstr = extname
		else:
			image += comp
			extnamestr += '_'+extname
			extnameplotstr += '+'+extname

		if gdat.show_input_maps:
			plot_single_map(comp, title=extname, lopct=5, hipct=95, save_bool=gdat.save_input_plots,\
					 save_dir=gdat.input_map_dir, filename=extname+'_band'+str(band)+'.'+gdat.fig_filetype)

			plot_single_map(image, title=extnameplotstr, lopct=5, hipct=95, save_bool=gdat.save_input_plots,\
								 save_dir=gdat.input_map_dir, filename=extnamestr+'_band'+str(band)+'.'+gdat.fig_filetype)

	uncertainty_map = None
	if gdat.use_uncertainty_map:
		if file_path is not None:
			assert gdat.uncertainty_map_extname in [df.name for df in data_file]
			uncertainty_map = np.nan_to_num(data_file[gdat.uncertainty_map_extname].data)
		elif gdat.err_fpath is not None:
			uncertainty_map = np.nan_to_num(fits.open(gdat.err_fpath)[err_hdu_idx].data)

	# if add_noise is True, Gaussian noise realization added to image. 
	# If scalar_noise_sigma specified, this sets RMS which is added, and updates the uncertainty map accordingly.
	# if scalar_noise_sigma not provided, PCAT assumes Gaussian noise realizations should be drawn from provided uncertainty maps, 
	# i.e. this is good for mock observations. 
	if gdat.add_noise:
		verbprint(gdat.verbtype, 'Adding Gaussian noise..')
		if gdat.scalar_noise_sigma is not None:
			if type(gdat.scalar_noise_sigma)==float:
				noise_sig = gdat.scalar_noise_sigma
			else:
				noise_sig = gdat.scalar_noise_sigma[band]

			noise_realization = np.random.normal(0, noise_sig, image.shape)
			
			if gdat.use_uncertainty_map:
				old_unc = uncertainty_map.copy()
				old_variance = old_unc**2
				old_variance[uncertainty_map != 0] += noise_sig**2
				uncertainty_map = np.sqrt(old_variance)
			else:
				uncertainty_map += noise_sig*np.ones((image.shape[0], image.shape[1]))

			if gdat.show_input_maps:
				plot_multipanel([noise_realization, showim, uncertainty_map], ['Noise realization', 'Image + Gaussian noise', 'Uncertainty map'], figsize=(15, 5), \
					lopct=5, hipct=95, save_bool=gdat.save_input_plots,\
								 save_dir=gdat.input_map_dir, filename='noiseim_sigplusnoiseim_uncmap_band'+str(band)+'.'+gdat.fig_filetype)

		else:
			print('Using uncertainty map to generate noise realization, assuming Gaussian distributed..')
			noise_realization = uncertainty_map*np.random.normal(0, 1, uncertainty_map.shape)

			if gdat.show_input_maps:
				plot_single_map(noise_realization, title='Noise realization', lopct=5, hipct=95, save_bool=gdat.save_input_plots,\
								 save_dir=gdat.input_map_dir, filename='noise_realization_band'+str(band)+'.'+gdat.fig_filetype)

		image += noise_realization
	
	# at this point the uncertainty map should be declared, i.e. it shouldn't be None.
	assert uncertainty_map is not None 

	if gdat.use_mask:
		if gdat.mask_file is not None:
			mask_fpath = gdat.mask_file.replace('PSW', 'P'+str(gdat.band_dict[band])+'W')
			mask = fits.open(mask_fpath)[0].data
		else:
			mask = data_file['MASK'].data
	else:
		verbprint(gdat.verbtype, 'Not using mask..', verbthresh=1)
		mask = np.ones_like(image)


	if gdat.show_input_maps:
		plot_multipanel([image, uncertainty_map, mask], ['Data', 'Uncertainty map', 'Mask'], figsize=(12, 4), lopct=[5, 5, None], hipct=[95, 95, None], save_bool=gdat.save_input_plots,\
								 save_dir=gdat.input_map_dir, filename='image_uncmap_mask_multipanel_band'+str(band)+'.'+gdat.fig_filetype)

	return image, uncertainty_map, mask, file_path



def load_param_dict(param_filepath=None, timestr=None, result_dir=None, encoding=None):
	
	"""
	Loads dictionary of configuration parameters from prior run of PCAT.

	Parameters
	----------

	timestr : string
		time string associated with desired PCAT run.

	result_dir : string, optional
		Directory location of PCAT run results.
		Default is None.

	Returns
	-------

	opt : object containing parameter dictionary

	filepath : string
		file path of parameter file


	"""

	if param_filepath is None:
		if result_dir is None:
			result_dir = config.result_basedir
		param_filepath = result_dir
		if timestr is not None:
			param_filepath += timestr

		param_filepath += '/params.txt'


	filen = open(param_filepath,'rb')
	
	if encoding is not None:
		pdict = pickle.load(filen, encoding=encoding)
	else:
		pdict = pickle.load(filen)
	opt = objectview(pdict) 
	
	return opt, param_filepath



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

	def __init__(self, gdat, load_in_data=True, save_input_plots=True, nregion=1):

		self.gdat = gdat

		self.ncs, self.nbins, self.psfs, self.cfs, self.biases,\
			 self.data_array, self.weights, self.masks, self.uncertainty_maps,\
			 	 self.fracs, self.template_array = [[] for x in range(11)]

		self.fast_astrom = wcs_astrometry(self.gdat.auto_resize, nregion=nregion)

		self.gdat.imszs, self.gdat.imszs_orig = [np.zeros((self.gdat.nbands, 2)).astype(np.int) for x in range(2)]
		self.gdat.x_max_pivot_list, self.gdat.y_max_pivot_list, self.gdat.regsizes = [np.zeros((self.gdat.nbands)).astype(int) for x in range(3)]
		self.gdat.bounds = np.zeros((self.gdat.nbands, 2, 2))

		if load_in_data:
			self.load_in_data()

	def square_pad_maps(self, image, uncertainty_map, mask, template_list):
		''' This pads the maps so that they are square, and modifies the associated uncertainty maps and masks. '''

		smaller_dim, larger_dim = np.min(image.shape), np.max(image.shape)
		width = find_nearest_mod(larger_dim, self.gdat.nregion)
		height = width
		image_size = (width, height)

		resized_image, resized_unc, resized_mask = [np.zeros(shape=image_size) for x in range(3)]

		crop_size_x = np.minimum(width, image.shape[0])
		crop_size_y = np.minimum(height, image.shape[1])

		resized_image[:image.shape[0], :image.shape[1]] = image[:crop_size_x, :crop_size_y]
		resized_unc[:image.shape[0], :image.shape[1]] = uncertainty_map[:crop_size_x, :crop_size_y]
		resized_mask[:image.shape[0], :image.shape[1]] = mask[:crop_size_x, :crop_size_y]
		
		resized_templates = []
		for template in template_list:
			resized_template = np.zeros(shape=image_size)
			resized_template[:image.shape[0], : image.shape[1]] = template[:crop_size_x, :crop_size_y]
			resized_templates.append(resized_template)

		return image_size, resized_image, resized_unc, resized_mask, resized_templates

	def load_sig_templates(self, band, template_file_names, sb_scale_facs=None):

		''' Loads signal templates and applies conversion factors if provided. '''
		if sb_scale_facs is None:
			sb_scale_facs = [None for x in range(len(self.gdat.template_names))]

		for t, template_name in enumerate(self.gdat.template_names):

			verbprint(self.gdat.verbtype, 'template name is ', template_name, verbthresh=1)
			# if self.gdat.band_dict[band] in config.template_bands_dict[template_name]:
				
			if self.gdat.band_dict[band] in self.gdat.template_bands_dict[template_name]:
				# verbprint(self.gdat.verbtype, 'Band, template band, lambda: '+str(self.gdat.band_dict[band])+', '+str(config.template_bands_dict[template_name])+', '+str(self.gdat.lam_dict[self.gdat.band_dict[band]]), verbthresh=1)
				verbprint(self.gdat.verbtype, 'Band, template band, lambda: '+str(self.gdat.band_dict[band])+', '+str(self.gdat.template_bands_dict[template_name])+', '+str(self.gdat.lam_dict[self.gdat.band_dict[band]]), verbthresh=1)

				template = fits.open(template_file_names[t])[template_name].data
				if show_input_maps:
					plot_single_map(template, title=template_name, save_bool=self.gdat.save_input_plots,\
								 save_dir=self.gdat.input_map_dir, filename='template_'+str(template_name)+'_band'+str(band)+'.'+self.gdat.fig_filetype)

				if sb_scale_facs[t] is not None:
					template *= sb_scale_facs[t]

			else:
				print('no band in template_bands_dict')
				template = None

			template_list.append(template)

		return template_list


	def load_in_data(self, tail_name=None, show_input_maps=False, \
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
			sb_conversion_dict = self.gdat.sb_conversion_dict

		if temp_mock_amps_dict is None:
			temp_mock_amps_dict = self.gdat.temp_mock_amps_dict


		if self.gdat.show_input_maps:
			self.gdat.input_map_dir = self.gdat.run_dir+'/input_map_plots/'

			if not os.path.isdir(self.gdat.input_map_dir):
				print('Making directory for input data parsing plots..')
				os.makedirs(self.gdat.input_map_dir)

		for b, band in enumerate(self.gdat.bands):
			image, uncertainty_map, mask, file_name = load_in_map(self.gdat, file_path=self.gdat.data_fpaths[b], band=band,\
																	 astrom=self.fast_astrom, show_input_maps=self.gdat.show_input_maps,\
																	image_extnames=self.gdat.image_extnames)
			imshp = image.shape 
			self.gdat.imszs_orig[b,:] = np.array([imshp[0], imshp[1]])
			self.gdat.bounds[b,:,:] = np.array([[0, imshp[0]], [0, imshp[1]]])

			verbprint(self.gdat.verbtype, 'self.gdat.imszs_orig['+str(b)+',:] = '+str(self.gdat.imszs_orig[b,:]), verbthresh=1)
			verbprint(self.gdat.verbtype, 'self.gdat.bounds['+str(b)+',:,:] = '+str(self.gdat.bounds[b,:,:]), verbthresh=1)

			# this part updates the fast astrometry information if cropping the images
			# if bounds is not None:
			# big_dim = np.maximum(find_nearest_mod(self.bounds[0,1]-bounds[0,0]+1, gdat.nregion), find_nearest_mod(bounds[1,1]-bounds[1,0]+1, gdat.nregion))
			# big_dim = np.maximum(find_nearest_mod(self.gdat.bounds[b,0,1]-self.gdat.bounds[b,0,0]+1, self.gdat.nregion), find_nearest_mod(self.gdat.bounds[b,1,1]-self.gdat.bounds[b,1,0]+1, self.gdat.nregion))
			big_dim = np.maximum(find_nearest_mod(self.gdat.bounds[b,0,1]-self.gdat.bounds[b,0,0], self.gdat.nregion), find_nearest_mod(self.gdat.bounds[b,1,1]-self.gdat.bounds[b,1,0], self.gdat.nregion))

			verbprint(self.gdat.verbtype, 'big dim is '+str(big_dim), verbthresh=1)
			big_dim = int(big_dim)

			self.fast_astrom.dims[b] = (big_dim, big_dim)

			template_list = [] 
			if self.gdat.n_templates > 0:
				verbprint(self.gdat.verbtype, 'Loading signal templates..', verbthresh=1)
				template_list = self.load_sig_templates(band, template_file_names, sb_scale_facs=sb_scale_facs)

				for t, template_inject in enumerate(template_list):

					verbprint(self.gdat.verbtype, 'Adding template to image..', verbthresh=1)
					image += template_inject

					if show_input_maps:
						f = plot_multipanel([template_inject, image], [self.gdat.template_names[t], 'image + '+self.gdat.template_names[t]],\
											 figsize=(8,4), cmap='Greys', save_bool=gdat.save_input_plots,\
								 save_dir=gdat.input_map_dir, filename=str(self.gdat.template_names[t])+'_template_inject_image_band'+str(band)+'.'+gdat.fig_filetype)

			if b > 0:
				verbprint(self.gdat.verbtype, 'Moving to band '+str(band)+'..', verbthresh=1)
				verbprint(self.gdat.verbtype, 'Loading astrometry for band '+str(band), verbthresh=1)
				self.fast_astrom.fit_astrom_arrays(0, b, bounds0=self.gdat.bounds[0], bounds1=self.gdat.bounds[b])
				
				x_max_pivot, y_max_pivot = self.fast_astrom.transform_q(np.array([self.gdat.imszs[0,0]]), np.array([self.gdat.imszs[0,1]]), b-1)
				x_max_pivot = int(x_max_pivot)
				y_max_pivot = int(y_max_pivot)
				verbprint(self.gdat.verbtype, 'xmaxpivot, ymaxpivot for band '+str(b)+' are '+str(x_max_pivot)+','+str(y_max_pivot), verbthresh=1)
				self.gdat.x_max_pivot_list[b] = x_max_pivot
				self.gdat.y_max_pivot_list[b] = y_max_pivot
			else:
				self.gdat.x_max_pivot_list[b] = big_dim
				self.gdat.y_max_pivot_list[b] = big_dim

			image = grab_map_by_bounds(self.gdat.bounds[b], image)
			uncertainty_map = grab_map_by_bounds(self.gdat.bounds[b], uncertainty_map)
			for t, template in enumerate(template_list):
				if template is not None:
					template_list[t] = grab_map_by_bounds(self.gdat.bounds[b], template) # bounds
			# uncertainty_map = uncertainty_map[bounds[0,0]:bounds[0,1]+1, bounds[1,0]:bounds[1,1]+1] # bounds
			# image = image[bounds[0,0]:bounds[0,1]+1, bounds[1,0]:bounds[1,1]+1]

			image_size, resized_image, resized_unc, resized_mask, resized_templates = self.square_pad_maps(image, uncertainty_map, mask, template_list)

			width, height = image_size[0], image_size[1]

			if b > 0 and int(x_max_pivot) < resized_image.shape[0]:
				verbprint('Setting pixels in band '+str(b)+' not in band 0 FOV to zero..')
				resized_image[resized_mask==0] = 0.
				resized_unc[resized_mask==0] = 0.

				if show_input_maps:
					plot_multipanel([resized_mask, resized_image, resized_unc], ['resized mask', 'resized_image', 'resized_unc'], \
								save_bool=gdat.save_input_plots,\
								 save_dir=gdat.input_map_dir, filename='resized_mask_resized_image_resized_unc_band'+str(band)+'.'+gdat.fig_filetype)
			
			if show_input_maps:
				for t, resized_template in enumerate(resized_templates):
					plot_multipanel([resized_template, resized_image], ['Resized template, '+self.gdat.template_order[t], 'Resized Image'], cmap='Greys', save_bool=gdat.save_input_plots,\
								 save_dir=gdat.input_map_dir, filename='resized_template_resized_image_band'+str(band)+'.'+gdat.fig_filetype)

			variance, weight = calc_weights(resized_unc)

			if show_input_maps:
				plot_single_map(weight, title='weight map', lopct=5, hipct=95, save_bool=self.gdat.save_input_plots,\
								 save_dir=self.gdat.input_map_dir, filename='weight_map_band'+str(band)+'.'+self.gdat.fig_filetype)

			self.weights.append(weight.astype(np.float32))
			self.uncertainty_maps.append(resized_unc.astype(np.float32))
			resized_image[weight==0] = 0.

			# remove this?
			self.data_array.append(resized_image.astype(np.float32)-self.gdat.mean_offsets[b]) # constant offset, will need to change
			# self.data_array.append(resized_image.astype(float))
			
			self.template_array.append(resized_templates)

			if b==0:
				self.gdat.imsz0 = image_size

			if show_input_maps:
				plot_single_map(self.data_array[b], title='Data, '+self.gdat.tail_name, lopct=5, hipct=95, save_bool=self.gdat.save_input_plots,\
								 save_dir=self.gdat.input_map_dir, filename='image_data_band'+str(band)+'.'+self.gdat.fig_filetype)
				plot_single_map(self.uncertainty_maps[b], title='Uncertainty map, '+self.gdat.tail_name, lopct=5, hipct=95, save_bool=self.gdat.save_input_plots,\
								 save_dir=self.gdat.input_map_dir, filename='uncertainty_map_band'+str(band)+'.'+self.gdat.fig_filetype)

			self.gdat.imszs[b,:] = np.array(image_size)
			self.gdat.regsizes[b] = image_size[0]/self.gdat.nregion
			self.gdat.frac = np.count_nonzero(weight)/float(width*height)
			
			if self.gdat.psf_postage_stamps is not None:
				verbprint(self.gdat.verbtype, 'Using psf postage stamp/s provided..', verbthresh=0)
				verbprint('NOTE this assumes psf_postage_stamps have been upsampled by factor '+str(self.gdat.nbin)+' already..')
				psf = self.gdat.psf_postage_stamps[b].copy()
				nbin = self.gdat.nbin
				nc = nbin**2
				cf = psf_poly_fit(psf, nbin=nbin)


			elif self.gdat.psf_fwhms[b] is not None:
				verbprint(self.gdat.verbtype, 'Using provided PSF FWHMs to generate Gaussian beam..')
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

		verbprint(self.gdat.verbtype, 'self.gdat.imszs is '+str(self.gdat.imszs), verbthresh=1)

		if b==0:
			# if self.gdat.nominal_nsrc is not None:
			# 	print('Using expected number of sources to scale flux kernel width, where nominal Nsrc = ', self.gdat.nominal_nsrc)
			# 	npiximage = image_size[0]*image_size[1]
			# 	# this barely changes anything
			# 	pixratio = (npiximage - (self.gdat.nominal_nsrc-1)*self.gdat.N_eff)/(npiximage - self.gdat.nominal_nsrc*self.gdat.N_eff)
			# 	# print('(A-(n-1)A_PSF)/(A-nA_PSF) = ', pixratio)
			# 	self.gdat.err_f = np.sqrt(self.gdat.N_eff*pixel_variance*pixratio)/self.gdat.err_f_divfac
			# else:
			
			self.gdat.err_f = np.sqrt(self.gdat.N_eff * pixel_variance)/self.gdat.err_f_divfac
			verbprint(self.gdat.verbtype, 'self.gdat.err_f = '+str(self.gdat.err_f)+' while err_f_divfac = '+str(self.gdat.err_f_divfac), verbthresh=1)









