import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fftpack import fft, ifft
import networkx as nx
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
from PIL import Image
import sys
import pandas as pd
from pcat_main import *
# from spire_roc_condensed_cat import *
from diffuse_gen import *
from fourier_bkg_modl import *


def convert_pngs_to_gif(filenames, gifdir='/Users/richardfeder/Documents/multiband_pcat/', name='', duration=1000, loop=0):

	# Create the frames
	frames = []
	for i in range(len(filenames)):
		new_frame = Image.open(gifdir+filenames[i])
		frames.append(new_frame)

	# Save into a GIF file that loops forever
	frames[0].save(gifdir+name+'.gif', format='GIF',
				   append_images=frames[1:],
				   save_all=True,
				   duration=duration, loop=loop)


def condensed_catalog_overlaid_data(images, all_xs, all_ys, all_fs=None, bands=None, show=False, return_fig=True, unit='Jy/beam', plot_src_mask=None, flux_cut_idx=0, flux_cut=None, \
	red_cut = False, red_color_idx_order = [0, 1, 2]):
	
	nbands = len(all_xs)

	f = plt.figure(figsize=(5.*nbands, 5))

	if red_cut:
		red_src_mask = (all_fs[red_color_idx_order[0]]/all_fs[red_color_idx_order[1]] > 1.)*(all_fs[red_color_idx_order[1]]/all_fs[red_color_idx_order[2]] > 1.)

		print(red_src_mask)


	elif flux_cut is not None:
		print(all_fs[flux_cut_idx])
		plot_src_mask = (all_fs[flux_cut_idx] > flux_cut)
		print(plot_src_mask)
		print(np.sum(plot_src_mask))

	for b in range(nbands):
		print('b == ', b)
		plt.subplot(1, nbands, 1+b)
		if bands is not None:
			plt.title(bands[b], fontsize=20)
		plt.imshow(images[b], vmin=np.percentile(images[b], 5), vmax=np.percentile(images[b], 95), cmap='Greys', origin='lower')
		cbar = plt.colorbar(fraction=0.046, pad=0.04)
		cbar.set_label(unit, fontsize=14)
		plt.xlim(0, images[b].shape[0])
		plt.ylim(0, images[b].shape[1])
		plt.xlabel('x [pix]', fontsize=16)
		plt.ylabel('y [pix]', fontsize=16)

		sizes = None
		if all_fs is not None:
			sizes = 1.5e3*all_fs[b]

		if red_cut:
			plt.scatter(all_xs[b][red_src_mask], all_ys[b][red_src_mask], s=sizes[red_src_mask], marker='x', color='r')

		elif plot_src_mask is not None:
			plt.scatter(all_xs[b][plot_src_mask], all_ys[b][plot_src_mask], s=sizes[plot_src_mask], marker='x', color='r')
		else:
			plt.scatter(all_xs[b], all_ys[b], s=sizes, marker='x', color='r')

	plt.tight_layout()
	if show:
		plt.show()

	if return_fig:
		return f

def grab_extent(bins_var1, bins_var2):
    extent = [bins_var1[0], bins_var1[-1], bins_var2[0], bins_var2[-1]]
    return extent

def plot_atcr(listsamp, title):

	""" Plot chain autocorrelation function """
	numbsamp = listsamp.shape[0]
	four = fft(listsamp - np.mean(listsamp, axis=0), axis=0)
	atcr = ifft(four * np.conjugate(four), axis=0).real
	atcr /= np.amax(atcr, 0)

	autocorr = atcr[:int(numbsamp/2), ...]
	indxatcr = np.where(autocorr > 0.2)
	timeatcr = np.argmax(indxatcr[0], axis=0)

	numbsampatcr = autocorr.size

	figr, axis = plt.subplots(figsize=(6,4))
	plt.title(title, fontsize=16)
	axis.plot(np.arange(numbsampatcr), autocorr)
	axis.set_xlabel(r'$\tau$', fontsize=16)
	axis.set_ylabel(r'$\\xi(\tau)$', fontsize=16)
	axis.text(0.8, 0.8, r'$\tau_{exp} = %.3g$' % timeatcr, ha='center', va='center', transform=axis.transAxes, fontsize=16)
	axis.axhline(0., ls='--', alpha=0.5)
	plt.tight_layout()

	return figr

def plot_atcr_multichain(listsamp, title=None, alpha=1.0, show=False):
    """ Plot ensemble of chain autocorrelation functions """

    figr, axis = plt.subplots(figsize=(6,4))
    if title is not None:
        plt.title(title, fontsize=16)
    
    if np.ndim(listsamp) == 1:
        listsamp = [listsamp]
        
    timeatcr_list = np.zeros((len(listsamp),))
    autocorrs = []
    for s, samp in enumerate(listsamp):
        
        numbsamp = samp.shape[0]
        four = fft(samp - np.mean(samp, axis=0), axis=0)
        atcr = ifft(four * np.conjugate(four), axis=0).real
        atcr /= np.amax(atcr, 0)

        autocorr = atcr[:int(numbsamp/2), ...]
        autocorrs.append(autocorr)
        indxatcr = np.where(autocorr > 0.2)
        timeatcr = np.argmax(indxatcr[0], axis=0)
        timeatcr_list[s] = timeatcr
        numbsampatcr = autocorr.size
        axis.plot(np.arange(numbsampatcr), autocorr, alpha=alpha, color='k')

    axis.set_xlabel(r'$\tau$', fontsize=16)
    axis.set_ylabel(r'$\\xi(\tau)$', fontsize=16)

    if len(listsamp) > 1:
        axis.text(0.8, 0.8, r'$\overline{\tau}_{exp} = %.3g$' % np.median(timeatcr_list), ha='center', va='center', transform=axis.transAxes, fontsize=16)
    else:
        axis.text(0.8, 0.8, r'$\tau_{exp} = %.3g$' % timeatcr, ha='center', va='center', transform=axis.transAxes, fontsize=16)
    axis.axhline(0., ls='--', alpha=0.5)
    plt.tight_layout()

    if show:
    	plt.show()

    return figr, np.arange(numbsampatcr), autocorrs



def plot_custom_multiband_frame(obj, resids, models, panels=['data0','model0', 'residual0','model1', 'residual1','residual2'], \
							zoomlims=None, ndeg=None, \
							fourier_bkg=None, bcib_bkg=None, sz=None, frame_dir_path=None,\
							smooth_fac=4, minpct=5, maxpct=95):

	""" 
	This is the primary function for plotting combinations of the data and model during run time. Seeing the model an residuals, for example, 
	can be useful in troubleshooting bugs. 

	This is an in place operation, with an option to save intermediate frames.

	Notes: 
		- Might integrate this more with the model class so not as cumbersome.
		- generalize "sz" to be any of the templates.
		- Include details of panel name convention.


	Parameters
	----------
	
	obj : 
	resids : 
	models : 
	panels : 'list' of 'strings'. Specifies the panels which are made
		Default is ['data0', 'model0', 'residual0','model1', 'residual1','residual2'].
	zoomlims : 'list' of 'list' of 'ints'. optional zoom in limits for maps.
		Default is None.
	ndeg : 
	fourier_bkg : 
	bcib_bkg : 
		Default is None.
	frame_dir_path : 'str'.
		Default is None.
	smooth_fac : 
	minpct : 'float'. Minimum percentile in colorbar stretch.
		Default is 5.
	maxpct : 'float'. Maximum percentile in colorbar stretch.
		Default is 95.
	

	Returns
	-------

	"""
	

	plt.gcf().clear()
	plt.figure(1, figsize=(15, 10))
	plt.clf()

	scatter_sizefac = 300

	for i in range(6):

		plt.subplot(2,3,i+1)

		band_idx = int(panels[i][-1])

		if 'data' in panels[i]:

			title = 'Data'
			if 'minusptsrc' in panels[i] and fourier_bkg is not None:
				plt.imshow(resids[band_idx]+fourier_bkg[band_idx], origin='lower', interpolation='none', cmap='Greys', vmin=np.percentile(resids[band_idx]+fourier_bkg[band_idx], minpct), vmax=np.percentile(resids[band_idx]+fourier_bkg[band_idx], maxpct))
			elif 'minusfbkg' in panels[i] and fourier_bkg is not None:
				plt.imshow(obj.dat.data_array[band_idx]-fourier_bkg[band_idx], origin='lower', interpolation='none', cmap='Greys', vmin=np.percentile(obj.dat.data_array[band_idx]-fourier_bkg[band_idx], minpct), vmax=np.percentile(obj.dat.data_array[band_idx]-fourier_bkg[band_idx], maxpct))
			else:				
				plt.imshow(obj.dat.data_array[band_idx], origin='lower', interpolation='none', cmap='Greys', vmin=np.percentile(obj.dat.data_array[band_idx], minpct), vmax=np.percentile(obj.dat.data_array[band_idx], maxpct))
			if band_idx > 0:
				xp, yp = obj.dat.fast_astrom.transform_q(obj.stars[obj._X, 0:obj.n], obj.stars[obj._Y, 0:obj.n], band_idx-1)
				plt.scatter(xp, yp, marker='x', s=obj.stars[obj._F+1, 0:obj.n]*scatter_sizefac, color='r')
			else:
				plt.scatter(obj.stars[obj._X, 0:obj.n], obj.stars[obj._Y, 0:obj.n], marker='x', s=obj.stars[obj._F, 0:obj.n]*scatter_sizefac, color='r', alpha=0.8)

		elif 'model' in panels[i]:
			title= 'Model'
			plt.imshow(models[band_idx], origin='lower', interpolation='none', cmap='Greys', vmin=np.percentile(models[band_idx], minpct), vmax=np.percentile(models[band_idx], maxpct))

		elif 'injected_diffuse_comp' in panels[i]:
			title = 'Injected cirrus'
			plt.imshow(obj.dat.injected_diffuse_comp[band_idx], origin='lower', interpolation='none', cmap='Greys', vmin=np.percentile(obj.dat.injected_diffuse_comp[band_idx], minpct), vmax=np.percentile(obj.dat.injected_diffuse_comp[band_idx], maxpct))
		

		elif 'fourier_bkg' in panels[i]:
			title='Fourier components'
			fbkg = fourier_bkg[band_idx]
			fbkg[obj.dat.weights[band_idx]==0] = 0.

			plt.imshow(fbkg, origin='lower', interpolation='none', cmap='Greys', vmin = np.percentile(fbkg , minpct), vmax=np.percentile(fbkg, maxpct))
		

		# elif 'bcib' in panels[i]:
		# 	title = 'Binned CIB'
		# 	bcib = bcib_bkg[band_idx]
		# 	bcib[obj.dat.weights[band_idx]==0] = 0.

		# 	plt.imshow(bcib, origin='lower', interpolation='none', cmap='Greys', vmin = np.percentile(bcib, minpct), vmax=np.percentile(bcib, maxpct))
		

		elif 'residual' in panels[i]:
			title= 'Residual'
			if obj.gdat.weighted_residual:
				plt.imshow(resids[band_idx]*np.sqrt(obj.dat.weights[band_idx]), origin='lower', interpolation='none', cmap='Greys', vmin=-5, vmax=5)
			else:
				plt.imshow(resids[band_idx], origin='lower', interpolation='none', cmap='Greys', vmin = np.percentile(resids[band_idx][obj.dat.weights[band_idx] != 0.], minpct), vmax=np.percentile(resids[band_idx][obj.dat.weights[band_idx] != 0.], maxpct))

			if band_idx > 0:
				xp, yp = obj.dat.fast_astrom.transform_q(obj.stars[obj._X, 0:obj.n], obj.stars[obj._Y, 0:obj.n], band_idx-1)
				plt.scatter(xp, yp, marker='x', s=obj.stars[obj._F+1, 0:obj.n]*scatter_sizefac, color='r')
			else:
				plt.scatter(obj.stars[obj._X, 0:obj.n], obj.stars[obj._Y, 0:obj.n], marker='x', s=obj.stars[obj._F, 0:obj.n]*scatter_sizefac, color='r', alpha=0.8)
	

		elif 'sz' in panels[i]:

			title = 'SZ'

			plt.imshow(sz[band_idx], origin='lower', interpolation='none', cmap='Greys')

			if band_idx > 0:
				xp, yp = obj.dat.fast_astrom.transform_q(obj.stars[obj._X, 0:obj.n], obj.stars[obj._Y, 0:obj.n], band_idx-1)
				plt.scatter(xp, yp, marker='x', s=obj.stars[obj._F+1, 0:obj.n]*100, color='r')
			else:
				plt.scatter(obj.stars[obj._X, 0:obj.n], obj.stars[obj._Y, 0:obj.n], marker='x', s=obj.stars[obj._F, 0:obj.n]*scatter_sizefac, color='r')	


		plt.colorbar(fraction=0.046, pad=0.04)
		title += ', band '+str(band_idx)
		if 'zoom' in panels[i]:
			title += ' (zoomed in)'
			plt.xlim(zoomlims[band_idx][0])
			plt.ylim(zoomlims[band_idx][1])
		else:           
			plt.xlim(-0.5, obj.imszs[band_idx][0]-0.5)
			plt.ylim(-0.5, obj.imszs[band_idx][1]-0.5)
		plt.title(title, fontsize=18)		


	if frame_dir_path is not None:
		plt.savefig(frame_dir_path, bbox_inches='tight', dpi=200)
	plt.draw()
	plt.pause(1e-5)



def scotts_rule_bins(samples):
	"""
	Computes binning for a collection of samples using Scott's rule, which minimizes the integrated MSE of the density estimate.

	Parameters
	----------
	samples : 'list' or '~numpy.ndarray' of shape (Nsamples,)

	Returns
	-------
	bins : `~numpy.ndarray' of shape (Nsamples,)
		bin edges

	"""
	n = len(samples)
	bin_width = 3.5*np.std(samples)/n**(1./3.)
	k = np.ceil((np.max(samples)-np.min(samples))/bin_width)
	bins = np.linspace(np.min(samples), np.max(samples), int(k))
	return bins


def plot_bkg_sample_chain(bkg_samples, band='250 micron', title=True, show=False, convert_to_MJy_sr_fac=None, smooth_fac=None):

	""" This function takes a chain of background samples from PCAT and makes a trace plot. """

	if convert_to_MJy_sr_fac is None:
		convert_to_MJy_sr_fac = 1.
		ylabel_unit = ' [Jy beam$^{-1}$]'
	else:
		ylabel_unit = ' [MJy sr$^{-1}$]'

	f = plt.figure()
	if title:
		plt.title('Uniform background level - '+str(band))

	if smooth_fac is not None:
		bkg_samples = np.convolve(bkg_samples, np.ones((smooth_fac,))/smooth_fac, mode='valid')

	plt.plot(np.arange(len(bkg_samples)), bkg_samples/convert_to_MJy_sr_fac, label=band)
	plt.xlabel('Sample index', fontsize=14)
	plt.ylabel('Amplitude'+ylabel_unit, fontsize=14)
	plt.legend()
	
	if show:
		plt.show()

	return f

def plot_template_amplitude_sample_chain(template_samples, band='250 micron', template_name='sze', title=True, show=False, xlabel='Sample index', ylabel='Amplitude',\
									 convert_to_MJy_sr_fac=None, smooth_fac = None):

	""" This function takes a chain of template amplitude samples from PCAT and makes a trace plot. """

	ylabel_unit = None
	if convert_to_MJy_sr_fac is None:
		convert_to_MJy_sr_fac = 1.
		ylabel_unit = ' [Jy beam$^{-1}$]'
	else:
		ylabel_unit = ' [MJy sr$^{-1}$]'

	if template_name=='dust' or template_name == 'planck':
		ylabel_unit = None

	if smooth_fac is not None:
		print('Applying smoothing to sample chain..')
		template_samples = np.convolve(template_samples, np.ones((smooth_fac,))/smooth_fac, mode='valid')

	f = plt.figure()
	if title:
		plt.title(template_name +' template level - '+str(band))

	plt.plot(np.arange(len(template_samples)), template_samples, label=band)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()
	
	if show:
		plt.show()

	return f

def plot_template_median_std(template, template_samples, band='250 micron', template_name='cirrus dust', title=True, show=False, convert_to_MJy_sr_fac=None, \
	minpct=5, maxpct=95, cmap='Greys'):

	""" 
	This function takes a template and chain of template amplitudes samples from PCAT. 
	These are used to compute the median template estimate as well the pixel-wise standard deviation on the template.
	"""

	if convert_to_MJy_sr_fac is None:
		convert_to_MJy_sr_fac = 1.
		xlabel_unit = '[Jy beam$^{-1}$]'
	else:
		xlabel_unit = '[MJy sr$^{-1}$]'

	f = plt.figure(figsize=(10, 5))

	if title:
		plt.suptitle(template_name)

	mean_t, std_t = np.mean(template_samples), np.std(template_samples)

	plt.subplot(1,2,1)
	plt.title('Median', fontsize=18)
	plt.imshow(mean_t*template/convert_to_MJy_sr_fac, origin='lower', cmap=cmap, interpolation=None, vmin=np.percentile(mean_t*template, minpct)/convert_to_MJy_sr_fac, vmax=np.percentile(mean_t*template, maxpct)/convert_to_MJy_sr_fac)
	cbar = plt.colorbar(fraction=0.046, pad=0.04)
	cbar.set_label(xlabel_unit, fontsize=14)
	plt.subplot(1,2,2)
	plt.title('Standard deviation'+xlabel_unit, fontsize=18)
	plt.imshow(std_t*np.abs(template)/convert_to_MJy_sr_fac, origin='lower', cmap=cmap, interpolation=None, vmin=np.percentile(std_t*np.abs(template), minpct)/convert_to_MJy_sr_fac, vmax=np.percentile(std_t*np.abs(template), maxpct)/convert_to_MJy_sr_fac)
	cbar = plt.colorbar(fraction=0.046, pad=0.04)
	cbar.set_label(xlabel_unit, fontsize=14)

	if show:
		plt.show()
	return f


# def plot_bcib_median_std(binned_cib_coeffs, coarse_cib_templates, ref_img=None, title=True, show=False):
# 	""" Plot median/standard deviation across samples for binned CIB model. 

# 	Inputs
# 	------


# 	Returns
# 	-------


# 	"""
# 	cib_nregion = int(np.sqrt(binned_cib_coeffs.shape[-1]))

# 	bcib_modls = []

# 	for i, bcib_coeff_state in enumerate(binned_cib_coeffs):
# 		bcib_modls.append(np.sum([bcib_coeff_state[j]*coarse_cib_templates[j] for j in range(binned_cib_coeffs.shape[-1])], axis=0))
# 	bcib_modls = np.array(bcib_modls)

# 	mean_bcib_temp = np.median(bcib_modls, axis=0)
# 	std_bcib_temp = np.std(bcib_modls, axis=0)

# 	xlabel_unit = 'Jy/beam'

# 	if ref_img is not None:
# 		f = plt.figure(figsize=(15, 5))

# 		plt.subplot(1,3,1)
# 		plt.title('Data', fontsize=14)
# 		plt.imshow(ref_img, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(ref_img, 5), vmax=np.percentile(ref_img, 99))
# 		cb = plt.colorbar(orientation='vertical', pad=0.04, fraction=0.046)
# 		cb.set_label(xlabel_unit)
# 		plt.subplot(1,3,2)
# 		plt.title('Median binned CIB model', fontsize=14)
# 		plt.imshow(mean_bcib_temp, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(mean_bcib_temp, 5), vmax=np.percentile(mean_bcib_temp, 99))
# 		cb = plt.colorbar(orientation='vertical', pad=0.04, fraction=0.046)
# 		cb.set_label(xlabel_unit)		
# 		plt.subplot(1,3,3)
# 		plt.title('Data - median binned CIB model', fontsize=14)
# 		plt.imshow(ref_img - mean_bcib_temp, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(ref_img, 5), vmax=np.percentile(ref_img, 97))
# 		cb = plt.colorbar(orientation='vertical', pad=0.04, fraction=0.046)
# 		cb.set_label(xlabel_unit)

# 	else:
# 		f = plt.figure(figsize=(11, 5))

# 		plt.subplot(1,2,1)
# 		plt.title('Median'+xlabel_unit)
# 		plt.imshow(mean_bcib_temp, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(mean_bcib_temp, 5), vmax=np.percentile(mean_bcib_temp, 95))
# 		plt.colorbar(pad=0.04)
# 		plt.subplot(1,2,2)
# 		plt.title('Standard deviation'+xlabel_unit)
# 		plt.imshow(std_bcib_temp, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(std_bcib_temp, 5), vmax=np.percentile(std_bcib_temp, 95))
# 		plt.colorbar(pad=0.04)

# 	plt.tight_layout()
# 	if show:
# 		plt.show()
	
# 	return f


# fourier comps

def plot_fc_median_std(fourier_coeffs, imsz, ref_img=None, bkg_samples=None, fourier_templates=None, title=True, show=False, convert_to_MJy_sr_fac=None, psf_fwhm=None):
	
	if convert_to_MJy_sr_fac is None:
		convert_to_MJy_sr_fac = 1.
		xlabel_unit = ' [Jy/beam]'
	else:
		xlabel_unit = ' [MJy/sr]'

	n_terms = fourier_coeffs.shape[-2]

	all_temps = np.zeros((fourier_coeffs.shape[0], imsz[0], imsz[1]))
	if fourier_templates is None:
		fourier_templates = make_fourier_templates(imsz[0], imsz[1], n_terms, psf_fwhm=psf_fwhm)

	for i, fourier_coeff_state in enumerate(fourier_coeffs):
		all_temps[i] = generate_template(fourier_coeff_state, n_terms, fourier_templates=fourier_templates, N=imsz[0], M=imsz[1])
		if bkg_samples is not None:
			all_temps[i] += bkg_samples[i]

	mean_fc_temp = np.median(all_temps, axis=0)
	std_fc_temp = np.std(all_temps, axis=0)

	if ref_img is not None:
		f = plt.figure(figsize=(15, 5))

		plt.subplot(1,3,1)
		plt.title('Data', fontsize=14)
		plt.imshow(ref_img/convert_to_MJy_sr_fac, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(ref_img, 5)/convert_to_MJy_sr_fac, vmax=np.percentile(ref_img, 99)/convert_to_MJy_sr_fac)
		cb = plt.colorbar(orientation='vertical', pad=0.04, fraction=0.046)
		cb.set_label(xlabel_unit)
		plt.subplot(1,3,2)
		plt.title('Median background model', fontsize=14)
		plt.imshow(mean_fc_temp/convert_to_MJy_sr_fac, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(mean_fc_temp, 5)/convert_to_MJy_sr_fac, vmax=np.percentile(mean_fc_temp, 99)/convert_to_MJy_sr_fac)
		cb = plt.colorbar(orientation='vertical', pad=0.04, fraction=0.046)
		cb.set_label(xlabel_unit)		
		plt.subplot(1,3,3)
		plt.title('Data - median background model', fontsize=14)
		plt.imshow((ref_img - mean_fc_temp)/convert_to_MJy_sr_fac, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(ref_img, 5)/convert_to_MJy_sr_fac, vmax=np.percentile(ref_img, 97)/convert_to_MJy_sr_fac)
		cb = plt.colorbar(orientation='vertical', pad=0.04, fraction=0.046)
		cb.set_label(xlabel_unit)

	else:

		plt.subplot(1,2,1)
		plt.title('Median'+xlabel_unit)
		plt.imshow(mean_fc_temp/convert_to_MJy_sr_fac, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(mean_fc_temp, 5)/convert_to_MJy_sr_fac, vmax=np.percentile(mean_fc_temp, 95)/convert_to_MJy_sr_fac)
		plt.colorbar(pad=0.04)
		plt.subplot(1,2,2)
		plt.title('Standard deviation'+xlabel_unit)
		plt.imshow(std_fc_temp/convert_to_MJy_sr_fac, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(std_fc_temp, 5)/convert_to_MJy_sr_fac, vmax=np.percentile(std_fc_temp, 95)/convert_to_MJy_sr_fac)
		plt.colorbar(pad=0.04)

	plt.tight_layout()
	if show:
		plt.show()
	
	return f


def plot_flux_vs_fluxerr(fluxes, flux_errs, show=False, alpha=0.1, snr_levels = [2., 5., 10., 20., 50.], xlim=[1, 1e3], ylim=[0.1, 2e2]):

	""" This takes a list of fluxes and flux uncertainties and plots them against each other, along with selection of SNR levels for reference. """
	
	f = plt.figure()
	plt.title('Flux errors', fontsize=16)
	plt.scatter(fluxes, flux_errs, alpha=alpha, color='k', label='Condensed catalog')
	plt.xlabel('F [mJy]', fontsize=16)
	plt.ylabel('$\\sigma_F$ [mJy]')

	xspace = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 100)
	plt.xscale('log')
	plt.xlim(xlim)
	plt.yscale('log')
	plt.ylim(ylim)
	for s, snr in enumerate(snr_levels):
		plt.plot(xspace, xspace/snr, label='SNR = '+str(np.round(snr)), color='C'+str(s), linestyle='dashed')

	plt.legend(fontsize=14)
	plt.tight_layout()
	if show:
		plt.show()

	return f


def plot_degradation_factor_vs_flux(fluxes, deg_fac, show=False, deg_fac_mode='Flux', alpha=0.1, xlim=[1, 1e3], ylim=[0.5, 60]):

	"""

	alpha : float, optional
		scatter point transparency variable. Default is 0.1

	"""
	f = plt.figure()
	plt.title(deg_fac_mode+' degradation factor', fontsize=16)
	plt.scatter(fluxes, deg_fac, alpha=alpha, color='k', label='Condensed catalog')
	plt.xlabel('F [mJy]', fontsize=16)
	if deg_fac_mode=='Flux':
		plt.ylabel('DF = $\\sigma_F^{obs.}/\\sigma_F^{opt.}$', fontsize=16)
	elif deg_fac_mode=='Position':
		plt.ylabel('DF = $\\sigma_x^{obs.}/\\sigma_x^{opt.}$', fontsize=16)

	plt.xscale('log')
	plt.xlim(xlim)
	plt.yscale('log')
	plt.ylim(ylim)
	plt.axhline(1., linestyle='dashed', color='k', label='Optimal '+deg_fac_mode.lower()+' error \n (instrument noise only)')
	plt.legend(fontsize=14, loc=1)
	plt.tight_layout()
	if show:
		plt.show()

	return f


def plot_bcib_sample_chains(binned_cib_coeffs, show=False):
	f = plt.figure(figsize=(8,6))

	xvals = np.arange(binned_cib_coeffs.shape[0])
	for cib_coeff_idx in range(binned_cib_coeffs.shape[1]):
		plt.plot(xvals, binned_cib_coeffs[:,cib_coeff_idx], alpha=0.5, linewidth=2, color='k')
	plt.xlabel('Thinned samples', fontsize=18)
	plt.ylabel('Binned CIB template amplitude [Jy/beam]', fontsize=18)
	plt.tight_layout()
	if show:
		plt.show()
	return f


def plot_fourier_coeffs_sample_chains(fourier_coeffs, show=False):
	
	norm = matplotlib.colors.Normalize(vmin=0, vmax=np.sqrt(2)*fourier_coeffs.shape[1])
	colormap = matplotlib.cm.ScalarMappable(norm=norm, cmap='jet')

	ravel_ks = [np.sqrt(i**2+j**2) for i in range(fourier_coeffs.shape[1]) for j in range(fourier_coeffs.shape[2])]

	colormap.set_array(ravel_ks/(np.sqrt(2)*fourier_coeffs.shape[1]))

	f, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
	ax1, ax2, ax3, ax4 = axes.flatten()
	xvals = np.arange(fourier_coeffs.shape[0])

	for i in range(fourier_coeffs.shape[1]):
		for j in range(fourier_coeffs.shape[2]):
			ax1.plot(xvals, fourier_coeffs[:,i,j,0], alpha=0.5, linewidth=2, color=plt.cm.jet(np.sqrt(i**2+j**2)/(np.sqrt(2)*fourier_coeffs.shape[1])))
	ax1.set_xlabel('Thinned (post burn-in) samples')
	ax1.set_ylabel('$B_{ij,1}$', fontsize=18)

	for i in range(fourier_coeffs.shape[1]):
		for j in range(fourier_coeffs.shape[2]):
			ax2.plot(xvals, fourier_coeffs[:,i,j,1], alpha=0.5, linewidth=2, color=plt.cm.jet(np.sqrt(i**2+j**2)/(np.sqrt(2)*fourier_coeffs.shape[1])))
	ax2.set_xlabel('Thinned (post burn-in) samples')
	ax2.set_ylabel('$B_{ij,2}$', fontsize=18)

	for i in range(fourier_coeffs.shape[1]):
		for j in range(fourier_coeffs.shape[2]):
			ax3.plot(xvals, fourier_coeffs[:,i,j,2], alpha=0.5, linewidth=2, color=plt.cm.jet(np.sqrt(i**2+j**2)/(np.sqrt(2)*fourier_coeffs.shape[1])))
	ax3.set_xlabel('Thinned (post burn-in) samples')
	ax3.set_ylabel('$B_{ij,3}$', fontsize=18)

	for i in range(fourier_coeffs.shape[1]):
		for j in range(fourier_coeffs.shape[2]):
			ax4.plot(xvals, fourier_coeffs[:,i,j,3], alpha=0.5, linewidth=2, color=plt.cm.jet(np.sqrt(i**2+j**2)/(np.sqrt(2)*fourier_coeffs.shape[1])))
	ax4.set_xlabel('Thinned (post burn-in) samples')
	ax4.set_ylabel('$B_{ij,4}$', fontsize=18)

	f.colorbar(colormap, orientation='vertical', ax=ax1).set_label('$|k|$', fontsize=14)
	f.colorbar(colormap, orientation='vertical', ax=ax2).set_label('$|k|$', fontsize=14)
	f.colorbar(colormap, orientation='vertical', ax=ax3).set_label('$|k|$', fontsize=14)
	f.colorbar(colormap, orientation='vertical', ax=ax4).set_label('$|k|$', fontsize=14)

	plt.tight_layout()

	if show:
		plt.show()


	return f

def plot_mp_fit(temp_A_hat, n_terms, A_hat, data):
    plt.figure(figsize=(10,10))
    plt.suptitle('Moore-Penrose inverse, $N_{FC}$='+str(n_terms), fontsize=20, y=1.02)
    plt.subplot(2,2,1)
    plt.title('Background estimate', fontsize=18)
    plt.imshow(temp_A_hat, origin='lower', cmap='Greys', vmax=np.percentile(temp_A_hat, 99), vmin=np.percentile(temp_A_hat, 1))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(2,2,2)
    plt.hist(np.abs(A_hat), bins=np.logspace(-5, 1, 30))
    plt.xscale('log')
    plt.xlabel('Absolute value of Fourier coefficients', fontsize=14)
    plt.ylabel('N')
    plt.subplot(2,2,3)
    plt.title('Image', fontsize=18)
    plt.imshow(data, origin='lower', cmap='Greys', vmax=np.percentile(temp_A_hat, 95), vmin=np.percentile(temp_A_hat, 5))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(2,2,4)
    plt.title('Image - Background estimate', fontsize=18)
    plt.imshow(data-temp_A_hat, origin='lower', cmap='Greys', vmax=np.percentile(data-temp_A_hat, 95), vmin=np.percentile(data-temp_A_hat, 5))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def plot_posterior_fc_power_spectrum(fourier_coeffs, N, pixsize=6., show=False):

	n_terms = fourier_coeffs.shape[1]
	mesha, meshb = np.meshgrid(np.arange(1, n_terms+1) , np.arange(1, n_terms+1))
	kmags = np.sqrt(mesha**2+meshb**2)
	ps_bins = np.logspace(0, np.log10(n_terms+2), 6)

	twod_power_spectrum_realiz = []
	oned_ps_realiz = []
	kbin_masks = []
	for i in range(len(ps_bins)-1):
		kbinmask = (kmags >= ps_bins[i])*(kmags < ps_bins[i+1])
		kbin_masks.append(kbinmask)

	for i, fourier_coeff_state in enumerate(fourier_coeffs):
		av_2d_ps = np.mean(fourier_coeff_state**2, axis=2)
		oned_ps_realiz.append(np.array([np.mean(av_2d_ps[mask]) for mask in kbin_masks]))
		twod_power_spectrum_realiz.append(av_2d_ps)

	power_spectrum_realiz = np.array(twod_power_spectrum_realiz)
	oned_ps_realiz = np.array(oned_ps_realiz)

	f = plt.figure(figsize=(5, 5))
	# plt.subplot(1,2,1)

	fov_in_rad = N*(pixsize/3600.)*(np.pi/180.)

	plt.errorbar(2*np.pi/(fov_in_rad/np.sqrt(ps_bins[1:]*ps_bins[:-1])), np.median(oned_ps_realiz,axis=0), yerr=np.std(oned_ps_realiz, axis=0), color='k', capsize=5)
	plt.xlabel('$2\\pi/\\theta$ [rad$^{-1}$]', fontsize=16)
	plt.ylabel('$C_{\\ell}$', fontsize=16)
	plt.yscale('log')
	plt.xscale('log')
	plt.tick_params(labelsize=14)
	plt.tight_layout()
	if show:
		plt.show()


	return f


def plot_posterior_bkg_amplitude(bkg_samples, band='250 micron', title=True, show=False, xlabel='Amplitude', convert_to_MJy_sr_fac=None):
	
	if convert_to_MJy_sr_fac is None:
		convert_to_MJy_sr_fac = 1.
		xlabel_unit = ' [Jy/beam]'
	else:
		xlabel_unit = ' [MJy/sr]'

	f = plt.figure()
	if title:
		plt.title('Uniform background level - '+str(band))
		
	if len(bkg_samples)>50:
		binz = scotts_rule_bins(bkg_samples/convert_to_MJy_sr_fac)
	else:
		binz = 10

	plt.hist(np.array(bkg_samples)/convert_to_MJy_sr_fac, label=band, histtype='step', bins=binz)
	plt.xlabel(xlabel+xlabel_unit, fontsize=14)
	plt.ylabel('$N_{samp}$', fontsize=14)
	plt.tick_params(labelsize=14)

	if show:
		plt.show()

	return f    

def plot_posterior_template_amplitude(template_samples, band='250 micron', template_name='sze', title=True, show=False, xlabel='Amplitude', convert_to_MJy_sr_fac=None, \
									mock_truth=None, xlabel_unit=None):

	# print('template samples is ', template_samples)
	if convert_to_MJy_sr_fac is None:
		convert_to_MJy_sr_fac = 1.

		if xlabel_unit is None:
			xlabel_unit = ' [Jy/beam]'
	else:
		if xlabel_unit is None:
			xlabel_unit = ' [MJy/sr]'

	if template_name=='dust' or template_name == 'planck':
		xlabel_unit = ''

	f = plt.figure()
	if title:
		plt.title(template_name +' template level - '+str(band))

	if not np.isnan(template_samples).any() and np.std(template_samples) != 0:
		binz = 10
		plt.hist(np.array(template_samples)/convert_to_MJy_sr_fac, label=band, histtype='step', bins=binz)
	if mock_truth is not None:
		plt.axvline(mock_truth, linestyle='dashdot', color='r', label='Mock truth')
		plt.legend()
	plt.xlabel(xlabel+xlabel_unit, fontsize=14)
	plt.ylabel('$N_{samp}$', fontsize=14)
	plt.tick_params(labelsize=14)

	if show:
		plt.show()

	return f


def plot_posterior_flux_dist(logSv, raw_number_counts, band='250 micron', title=True, show=False, minpct=16, maxpct=84):

	mean_number_cts = np.mean(raw_number_counts, axis=0)
	lower = np.percentile(raw_number_counts, minpct, axis=0)
	upper = np.percentile(raw_number_counts, maxpct, axis=0)
	f = plt.figure()
	if title:
		plt.title('Posterior Flux Density Distribution - ' +str(band))

	plt.errorbar(logSv+3, mean_number_cts, yerr=np.array([np.abs(mean_number_cts-lower), np.abs(upper - mean_number_cts)]),fmt='.', label='Posterior')
	plt.legend()
	plt.yscale('log', nonposy='clip')

	plt.xlabel('$S_{\\nu}$ - ' + str(band) + ' [mJy]')
	plt.ylim(5e-1, 5e2)

	if show:
		plt.show()

	return f



def plot_residual_map(resid, mode='median', band='S', titlefontsize=14, smooth=True, smooth_sigma=3, \
					show=False, plot_refcat=False, convert_to_MJy_sr_fac=None, minpct=5, maxpct=95):


	# TODO - overplot reference catalog on image


	if convert_to_MJy_sr_fac is None:
		convert_to_MJy_sr_fac = 1.
		title_unit = '[Jy/beam]'
	else:
		title_unit = '[MJy/sr]'

	if mode=='median':
		title_mode = 'Median residual'
	elif mode=='last':
		title_mode = 'Last residual'

	if smooth:
		f = plt.figure(figsize=(10, 5))
	else:
		f = plt.figure(figsize=(8,8))
	
	if smooth:
		plt.subplot(1,2,1)

	plt.title(title_mode+' -- '+band+' '+title_unit, fontsize=titlefontsize)
	resid_im = resid/convert_to_MJy_sr_fac
	plt.imshow(resid_im, cmap='Greys', interpolation=None, vmin=np.percentile(resid_im, minpct), vmax=np.percentile(resid_im, maxpct), origin='lower')
	plt.colorbar(fraction=0.046, pad=0.04)

	if smooth:
		plt.subplot(1,2,2)
		smooth_resid = gaussian_filter(resid, sigma=smooth_sigma)/convert_to_MJy_sr_fac
		plt.title('Smoothed Residual'+title_unit, fontsize=titlefontsize)
		plt.imshow(smooth_resid, interpolation=None, cmap='Greys', vmin=np.percentile(smooth_resid, minpct), vmax=np.percentile(smooth_resid, maxpct), origin='lower')
		plt.colorbar(fraction=0.046, pad=0.04)

    plt.tick_params(labelsize=14)

	if show:
		plt.show()


	return f

def plot_residual_1pt_function(resid, mode='median', band='S', show=False, binmin=-0.02, binmax=0.02, nbin=50, convert_to_MJy_sr_fac=None, \
								title=None):


	if convert_to_MJy_sr_fac is None:
		convert_to_MJy_sr_fac = 1.
		xlabel_unit = '[Jy/beam]'
	else:
		xlabel_unit = '[MJy/sr]'

	if len(resid.shape) > 1:
		median_resid_rav = resid.ravel() 
	else:
		median_resid_rav = resid

	median_resid_rav /= convert_to_MJy_sr_fac


	if title is None and mode=='median':
		title = 'Median residual'
	
	f = plt.figure()
	plt.title(title+' 1pt function -- '+band)
	plt.hist(median_resid_rav, bins=nbin, histtype='step')
	plt.axvline(np.median(median_resid_rav), label='Median='+str(np.round(np.median(median_resid_rav), 5))+'\n $\\sigma=$'+str(np.round(np.std(median_resid_rav), 5)))
	plt.legend(frameon=False)
	plt.ylabel('$N_{pix}$', fontsize=14)
	plt.xlabel('Data - Model '+xlabel_unit, fontsize=14)
    plt.tick_params(labelsize=14)

	if show:
		plt.show()

	return f


def plot_multiband_chi_squared(chi2_list, sample_number, band_list, show=False, ndof_list=None):

	burn_in = sample_number[0]

	f = plt.figure()

	for b, band in enumerate(band_list):
		if ndof_list is not None:
			red_chi2= np.array(chi2_list[b])/np.array(ndof_list[b]).astype(np.float32)
			plt.plot(sample_number, red_chi2[burn_in:], label=band_list[b])
			plt.ylabel('Reduced chi-squared', fontsize=14)

		else:	
			plt.plot(sample_number, chi2_list[b][burn_in:], label=band_list[b])
			plt.ylabel('Chi-squared', fontsize=14)

	plt.xlabel('Sample index', fontsize=14)
    plt.tick_params(labelsize=14)

	plt.legend()

	if show:
		plt.show()

	return f

def plot_chi_squared(chi2, sample_number, band='S', show=False, ndof=None):

	burn_in = sample_number[0]
	f = plt.figure()
	if ndof is not None:
		chi2 /= ndof

	plt.plot(sample_number, chi2[burn_in:], label=band)
	plt.axhline(np.min(chi2[burn_in:]), linestyle='dashed',alpha=0.5, label=str(np.min(chi2[burn_in:]))+' (' + str(band) + ')')
	plt.xlabel('Sample index', fontsize=14)
	plt.ylabel('Chi-Squared', fontsize=14)
	plt.tick_params(labelsize=14)

	plt.legend()
	
	if show:
		plt.show()

	return f

def plot_comp_resources(timestats, nsamp, labels=['Proposal', 'Likelihood', 'Implement'], show=False):
	time_array = np.zeros(3, dtype=np.float32)
	
	for samp in range(nsamp):
		time_array += np.array([timestats[samp][2][0], timestats[samp][3][0], timestats[samp][4][0]])
	
	f = plt.figure()
	plt.title('Computational Resources', fontsize=16)
	plt.pie(time_array, labels=labels, autopct='%1.1f%%', shadow=True)
	
	if show:
		plt.show()
	
	return f

def plot_acceptance_fractions(accept_stats, proposal_types=['All', 'Move', 'Birth/Death', 'Merge/Split', 'Templates', 'Fourier Comps'], show=False, skip_idxs=None):

	f = plt.figure()
	
	samp_range = np.arange(accept_stats.shape[0])
	for x in range(len(proposal_types)):

		if skip_idxs is not None:
			if x in skip_idxs:
				continue
		accept_stats[:,x][np.isnan(accept_stats[:,x])] = 0.

		accept_stat_chain = accept_stats[:,x]

		plt.plot(np.arange(len(accept_stat_chain)), accept_stat_chain, label=proposal_types[x])
	plt.legend()
	plt.xlabel('Sample index', fontsize=14)
	plt.ylabel('Acceptance fraction', fontsize=14)
	plt.tick_params(labelsize=14)

	if show:
		plt.show()

	return f

def trace_plot(chains, titlestr=None, i0=0, ylabel=None, titlefontsize=18, show=True, return_fig=True):
    
    f = plt.figure(figsize=(6, 5))
    if titlestr is not None:
        plt.title(titlestr, fontsize=titlefontsize)
    # for chain in temp_chains_full[0]:

    for chain in chains:

        plt.plot(np.arange(i0, len(chain)), chain[i0:])
    
    plt.xlabel('$i_{samp}$', fontsize=18)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=18)
    plt.tick_params(labelsize=14)
    
    if show:
        plt.show()
    
    if return_fig:
        return f


def plot_src_number_posterior(nsrc_fov, show=False, title=False, nsrc_truth=None, fmin=4.0, units='mJy'):

	f = plt.figure()
	
	if title:
		plt.title('Posterior Source Number Histogram')
	
	plt.hist(nsrc_fov, histtype='step', label='Posterior', color='b', bins=15)
	plt.axvline(np.median(nsrc_fov), label='Median=' + str(np.median(nsrc_fov)), color='b', linestyle='dashed')
	if nsrc_truth is not None:
		plt.axvline(nsrc_truth, label='N (F > '+str(fmin)+' mJy) = '+str(nsrc_truth), linestyle='dashed', color='k', linewidth=1.5)
	plt.xlabel('$N_{src}$', fontsize=16)
	plt.ylabel('Number of samples', fontsize=16)
	plt.legend()
		
	if show:
		plt.show()
	
	return f


def plot_src_number_trace(nsrc_fov, show=False, title=False):

	f = plt.figure()
	
	if title:
		plt.title('Source number trace plot (post burn-in)')
	
	plt.plot(np.arange(len(nsrc_fov)), nsrc_fov)

	plt.xlabel('Sample index', fontsize=16)
	plt.ylabel('$N_{src}$', fontsize=16)
	plt.legend()
	
	if show:
		plt.show()

	return f


def plot_grap():

	"""
	Makes plot of probabilistic graphical model for SPIRE
	"""
		
	figr, axis = plt.subplots(figsize=(6, 6))

	grap = nx.DiGraph()   
	grap.add_edges_from([('muS', 'svec'), ('sigS', 'svec'), ('alpha', 'f0'), ('beta', 'nsrc')])
	grap.add_edges_from([('back', 'modl'), ('xvec', 'modl'), ('f0', 'modl'), ('svec', 'modl'), ('PSF', 'modl'), ('ASZ', 'modl')])
	grap.add_edges_from([('modl', 'data')])
	listcolr = ['black' for i in range(7)]
	
	labl = {}

	nameelem = r'\rm{pts}'


	labl['beta'] = r'$\beta$'
	labl['alpha'] = r'$\alpha$'
	labl['muS'] = r'$\vec{\mu}_S$'
	labl['sigS'] = r'$\vec{\sigma}_S$'
	labl['xvec'] = r'$\vec{x}$'
	labl['f0'] = r'$F_0$'
	labl['svec'] = r'$\vec{s}$'
	labl['PSF'] = r'PSF'
	labl['modl'] = r'$M_D$'
	labl['data'] = r'$D$'
	labl['back'] = r'$\vec{A_{sky}}$'
	labl['nsrc'] = r'$N_{src}$'
	labl['ASZ'] = r'$\vec{A_{SZ}}$'
	
	
	posi = nx.circular_layout(grap)
	posi['alpha'] = np.array([-0.025, 0.15])
	posi['muS'] = np.array([0.025, 0.15])
	posi['sigS'] = np.array([0.075, 0.15])
	posi['beta'] = np.array([0.12, 0.15])

	posi['xvec'] = np.array([-0.075, 0.05])
	posi['f0'] = np.array([-0.025, 0.05])
	posi['svec'] = np.array([0.025, 0.05])
	posi['PSF'] = np.array([0.07, 0.05])
	posi['back'] = np.array([-0.125, 0.05])
	
	posi['modl'] = np.array([-0.05, -0.05])
	posi['data'] = np.array([-0.05, -0.1])
	posi['nsrc'] = np.array([0.08, 0.01])
	
	posi['ASZ'] = np.array([-0.175, 0.05])


	rect = patches.Rectangle((-0.10,0.105),0.2,-0.11,linewidth=2, facecolor='none', edgecolor='k')

	axis.add_patch(rect)

	size = 1000
	nx.draw(grap, posi, labels=labl, ax=axis, edgelist=grap.edges())
	nx.draw_networkx_nodes(grap, posi, ax=axis, labels=labl, nodelist=['nsrc'], node_color='white', node_size=500)

	nx.draw_networkx_nodes(grap, posi, ax=axis, labels=labl, nodelist=['modl'], node_color='xkcd:sky blue', node_size=1000)
	nx.draw_networkx_nodes(grap, posi, ax=axis, labels=labl, nodelist=['beta'], node_shape='d', node_color='y', node_size=size)
	
	nx.draw_networkx_nodes(grap, posi, ax=axis, labels=labl, nodelist=['data'],  node_color='grey', node_shape='s', node_size=size)
	nx.draw_networkx_nodes(grap, posi, ax=axis, labels=labl, nodelist=['xvec', 'f0', 'svec'], node_color='orange', node_size=size)
	nx.draw_networkx_nodes(grap, posi, ax=axis, labels=labl, nodelist=['PSF'], node_shape='d', node_color='orange', node_size=size)
	
	nx.draw_networkx_nodes(grap, posi, ax=axis, labels=labl, nodelist=['back'], node_color='orange', node_size=size)
	
	nx.draw_networkx_nodes(grap, posi, ax=axis, labels=labl, nodelist=['ASZ'], node_color='violet', node_size=size)

	nx.draw_networkx_nodes(grap, posi, ax=axis, labels=labl, nodelist=['alpha', 'muS', 'sigS'], node_shape='d', node_color='y', node_size=size)

	
	plt.tight_layout()
	plt.show()
	
	return figr

def grab_atcr(timestr, paramstr='template_amplitudes', band=0, result_dir=None, nsamp=500, template_idx=0, return_fig=True):
	
	band_dict = dict({0:'250 micron', 1:'350 micron', 2:'500 micron'})
	lam_dict = dict({0:250, 1:350, 2:500})

	if result_dir is None:
		result_dir = '/Users/richardfeder/Documents/multiband_pcat/spire_results/'
		print('Result directory assumed to be '+result_dir)
		
	chain = np.load(result_dir+str(timestr)+'/chain.npz')
	if paramstr=='template_amplitudes':
		listsamp = chain[paramstr][-nsamp:, band, template_idx]
	else:
		listsamp = chain[paramstr][-nsamp:, band]
		
	f = plot_atcr(listsamp, title=paramstr+', '+band_dict[band])

	if return_fig:
		return f


def plot_single_map(image, show=True, title=None, return_fig=False, lopct=None, hipct=None, cmap=None):

	f = plt.figure()
	if title is not None:
		plt.title(title, fontsize=14)

	vmin, vmax = None, None
	if lopct is not None:
		vmin = np.nanpercentile(image, lopct)
	if hipct is not None:
		vmax = np.nanpercentile(image, hipct)
	plt.imshow(image, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
	plt.colorbar()
	if show:
		plt.show()
	if return_fig:
		return f


def plot_multipanel(image_list, str_list, xlabel='x', ylabel='y', return_fig = True, show=True, lopct=None, hipct=None, figsize=None, cmap=None, cbar_label=None, \
	suptitle=None):


	if lopct is not None:
		if type(lopct)==float:
			lopct = [lopct for i in image_list]
	if hipct is not None:
		if type(hipct)==float:
			hipct = [hipct for i in image_list]

	npanel = len(image_list)

	if npanel < 4:

		nx, ny = 1, npanel

	elif npanel==4:
		nx, ny = 2, 2

	elif npanel<6:
		nx, ny = 2, 3

	elif npanel >= 6:
		return None 

	if figsize is None:
		figsize = (4*ny, 4*nx)
	f = plt.figure(figsize=figsize)

	if suptitle is not None:
		plt.suptitle(suptitle, fontsize=20)

	for i, image in enumerate(image_list):


		vmin, vmax = None, None
		if lopct is not None:
			vmin = np.nanpercentile(image, lopct[i])
		if hipct is not None:
			vmax = np.nanpercentile(image, hipct[i])

		plt.subplot(nx, ny, i+1)
		plt.title(str_list[i], fontsize=16)
		plt.imshow(image, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
		cbar = plt.colorbar(fraction=0.046, pad=0.04)

		if cbar_label is not None:
			cbar.set_label(cbar_label)

		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.tick_params(labelsize=14)
	plt.tight_layout()
	if show:
		plt.show()

	if return_fig:
		return f






