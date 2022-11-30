from __future__ import print_function
import numpy as np
import numpy.ctypeslib as npct
import ctypes
from ctypes import c_int, c_double
# in order for visual=True to work, interactive backend should be loaded before importing pyplot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import os
import sys
import config
import warnings
import scipy.stats as stats
from scipy.ndimage import gaussian_filter


from image_eval import psf_poly_fit, image_model_eval
from fast_astrom import *
import pickle
from pcat_load_data import *
from pcat_utils import *
from plotting_fns import *
from fourier_bkg_modl import *
from diffuse_gen import *


np.seterr(divide='ignore', invalid='ignore')

class objectview(object):
	def __init__(self, d):
		self.__dict__ = d

class gdatstrt(object):
	""" Initializes global data object used throughout PCAT. """
	def __init__(self):
		pass
	
	def __setattr__(self, attr, valu):
		super(gdatstrt, self).__setattr__(attr, valu)


from priors import *
# def icdf_dpow(unit, minm, maxm, brek, sloplowr, slopuppr):
# def pdfn_dpow(xdat, minm, maxm, brek, sloplowr, slopuppr):# 
#  def compute_Fstat_alph(imszs, nbands, nominal_nsrc):


from pcat_utils import *
# def add_directory(dirpath):
# def create_directories(gdat):
# def verbprint(verbose, text, file=None, verbthresh=0):
# def initialize_c(gdat, libmmult, cblas=False):
# def save_params(directory, gdat):


def fluxes_to_color(flux1, flux2):
	""" Convert two flux densities to a color """
	return 2.5*np.log10(flux1/flux2)

def get_band_weights(band_idxs):
	weights = []
	for idx in band_idxs:
		if idx is None:
			weights.append(0.)
		else:
			weights.append(1.)
	weights /= np.sum(weights)

	return weights 

def neighbours(x,y,neigh,i,generate=False):
	"""
	Neighbours function is used in merge proposal, where you have some source and you want to choose a nearby
	source with some probability to merge. 
	"""

	neighx = np.abs(x - x[i])
	neighy = np.abs(y - y[i])
	adjacency = np.exp(-(neighx*neighx + neighy*neighy)/(2.*neigh*neigh))
	adjacency[i] = 0.
	neighbours = np.sum(adjacency)
	if generate:
		if neighbours:
			j = np.random.choice(adjacency.size, p=adjacency.flatten()/float(neighbours))
		else:
			j = -1
		return neighbours, j
	else:
		return neighbours

def get_region(x, offsetx, regsize):
	return (np.floor(x + offsetx).astype(np.int) / regsize).astype(np.int)

def idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize):
	match_x = (get_region(x[0:n], offsetx, regsize) % 2) == parity_x
	match_y = (get_region(y[0:n], offsety, regsize) % 2) == parity_y
	return np.flatnonzero(np.logical_and(match_x, match_y))

class Proposal:
	""" 
	This class contains the information related to PCAT proposals, including data structures and functions to draw from the 
	various proposal disributions. 
	"""
	_X = 0
	_Y = 1
	_F = 2

	def __init__(self, gdat):
		self.idx_move = None
		self.do_birth = False
		self.idx_kill = None
		self.factor = None
		
		self.goodmove = False
		self.change_bkg_bool = False
		self.change_template_amp_bool = False
		self.change_fourier_comp_bool = False
		self.change_binned_cib_bool = False
		self.perturb_band_idx = None
		
		self.dback = np.zeros(gdat.nbands, dtype=np.float32)
		self.dtemplate = None

		if gdat.float_fourier_comps:
			self.dfc = np.zeros((gdat.n_fourier_terms, gdat.n_fourier_terms, 4))
			self.dfc_rel_amps = np.zeros(gdat.nbands, dtype=np.float32)
			self.fc_rel_amp_bool = False

		self.xphon = np.array([], dtype=np.float32)
		self.yphon = np.array([], dtype=np.float32)
		self.fphon = []
		self.modl_eval_colors = []
		for x in range(gdat.nbands):
			self.fphon.append(np.array([], dtype=np.float32))
		self.gdat = gdat
	
	def set_factor(self, factor):
		self.factor = factor

	def in_bounds(self, catalogue):
		""" Checks that each element of the catalog is within the bounds of the image. """
		return np.logical_and(np.logical_and(catalogue[self._X,:] > 0, catalogue[self._X,:] < (self.gdat.imsz0[0] -1)), \
				np.logical_and(catalogue[self._Y,:] > 0, catalogue[self._Y,:] < self.gdat.imsz0[1] - 1))

	def assert_types(self):
		assert self.xphon.dtype == np.float32
		assert self.yphon.dtype == np.float32
		assert self.fphon[0].dtype == np.float32

	def __add_phonions_stars(self, stars, remove=False):
		fluxmult = -1 if remove else 1

		self.xphon = np.append(self.xphon, stars[self._X,:])
		self.yphon = np.append(self.yphon, stars[self._Y,:])

		for b in range(self.gdat.nbands):
			self.fphon[b] = np.append(self.fphon[b], np.array(fluxmult*stars[self._F+b,:], dtype=np.float32))
		self.assert_types()

	def add_move_stars(self, idx_move, stars0, starsp):
		""" 
		Once a move proposal is confirmed to be valid, this function loads proposed state to class and green lights the calculation of 
		the delta log-likelihood.

		This is an in place operation.

		Parameters
		----------
		idx_move : 'list' of 'ints'. List of sources being perturbed 
		stars0 : 'np.array' of 'floats'. initial point sources from idx_move
		starsp : 'np.array' of 'floats'. proposed point sources
		
		"""
		self.idx_move = idx_move
		self.stars0 = stars0
		self.starsp = starsp
		self.goodmove = True
		inbounds = self.in_bounds(starsp)
		if np.sum(~inbounds)>0:
			starsp[:,~inbounds] = stars0[:,~inbounds]
		self.__add_phonions_stars(stars0, remove=True)
		self.__add_phonions_stars(starsp)
		
	def add_birth_stars(self, starsb):
		""" 
		Same as add_move_stars but for birth proposals 
		
		Parameters
		----------
		starsb : 'np.array' of 'floats'. Point sources being added

		"""
		self.do_birth = True
		self.starsb = starsb
		self.goodmove = True
		if starsb.ndim == 3:
			starsb = starsb.reshape((starsb.shape[0], starsb.shape[1]*starsb.shape[2]))
		self.__add_phonions_stars(starsb)

	def add_death_stars(self, idx_kill, starsk):
		""" 
		Same as add_move_stars but for death proposals 
		
		Parameters
		----------
		idx_kill : 'list' of 'ints'. Indices of point sources removed in proposal.
		starsk : 'np.array' of 'floats'. Point sources being removed

		"""
		self.idx_kill = idx_kill
		self.starsk = starsk
		self.goodmove = True
		if starsk.ndim == 3:
			starsk = starsk.reshape((starsk.shape[0], starsk.shape[1]*starsk.shape[2]))
		self.__add_phonions_stars(starsk, remove=True)

	def change_bkg(self, perturb_band_idx=None):
		self.goodmove = True
		self.change_bkg_bool = True
		if perturb_band_idx is not None:
			self.perturb_band_idx = perturb_band_idx

	def change_template_amplitude(self, perturb_band_idx=None):
		self.goodmove = True
		self.change_template_amp_bool = True
		if perturb_band_idx is not None:
			self.perturb_band_idx = perturb_band_idx

	def change_fourier_comp(self):
		self.goodmove = True
		self.change_fourier_comp_bool = True

	def get_ref_xy(self):
		if self.idx_move is not None:
			return self.stars0[self._X,:], self.stars0[self._Y,:]
		elif self.do_birth:
			bx, by = self.starsb[[self._X,self._Y],:]
			refx = bx if bx.ndim == 1 else bx[:,0]
			refy = by if by.ndim == 1 else by[:,0]
			return refx, refy
		elif self.idx_kill is not None:
			xk, yk = self.starsk[[self._X,self._Y],:]
			refx = xk if xk.ndim == 1 else xk[:,0]
			refy = yk if yk.ndim == 1 else yk[:,0]
			return refx, refy
		elif self.change_bkg_bool:
			return self.stars0[self._X,:], self.stars0[self._Y,:]


class Model:

	_X = 0
	_Y = 1
	_F = 2

	k = 2.5/np.log(10)

	color_mus, color_sigs = [], []
	
	""" the init function sets all of the data structures used for the catalog, 
	randomly initializes catalog source values drawing from catalog priors  """
	def __init__(self, gdat, dat, libmmult=None, newsrc_minmax_range=500):

		self.dat = dat
		self.gdat = gdat
		self.libmmult = libmmult
		self.newsrc_minmax_range = newsrc_minmax_range
		self.err_f = gdat.err_f
		self.pixel_per_beam = [2*np.pi*(psf_pixel_fwhm/2.355)**2 for psf_pixel_fwhm in self.gdat.psf_fwhms] # variable pixel fwhm
		self.imsz0 = gdat.imsz0 # this is just for first band, where proposals are first made
		self.imszs = gdat.imszs # this is list of image sizes for all bands, not just first one
		self.kickrange = gdat.kickrange
		self.margins = np.zeros(gdat.nbands).astype(np.int)
		self.max_nsrc = gdat.max_nsrc
		self.bkg = np.array(gdat.bkg_level)

		# now part of gdat.sample_delay_byprop
		# self.sample_delays = [self.gdat.movestar_sample_delay, self.gdat.birth_death_sample_delay, self.gdat.merge_split_sample_delay, self.gdat.bkg_sample_delay, \
		# 				self.gdat.temp_sample_delay, self.gdat.fc_sample_delay, self.gdat.binned_cib_sample_delay]

		self.run_moveweights = np.array([0. for x in range(len(self.gdat.moveweight_byprop.keys()))]) # fourier comp, movestar. weights are specified in lion __init__()
		print('run moveweights is ', self.run_moveweights)
		self.run_movetypes = [self.gdat.print_movetypes[move] for move in self.gdat.print_movetypes.keys()]
		self.temp_amplitude_sigs = config.temp_amplitude_sigs

		if self.gdat.sz_amp_sig is not None:
			self.temp_amplitude_sigs['sze'] = self.gdat.sz_amp_sig
		if self.gdat.fc_amp_sig is not None:
			self.temp_amplitude_sigs['fc'] = self.gdat.fc_amp_sig
		# if self.gdat.binned_cib_amp_sig is not None:
		# 	self.temp_amplitude_sigs['binned_cib'] = self.gdat.binned_cib_amp_sig # binned cib
		
		# this is for perturbing the relative amplitudes of a fixed fourier comp model across bands
		self.fourier_amp_sig = gdat.fourier_amp_sig
		self.template_amplitudes = np.zeros((self.gdat.n_templates, gdat.nbands))
		self.init_template_amplitude_dicts = self.gdat.init_template_amplitude_dicts
		self.dtemplate = np.zeros_like(self.template_amplitudes)

		for i, key in enumerate(self.gdat.template_order):
			for b, band in enumerate(gdat.bands):
				self.template_amplitudes[i][b] = self.init_template_amplitude_dicts[key][gdat.band_dict[band]]
		
		# if self.gdat.float_cib_templates:
		# 	self.coarse_cib_templates = self.gdat.coarse_cib_templates
		# 	self.dbcc = np.zeros((gdat.cib_nregion**2))
		# 	self.binned_cib_coeffs = np.zeros_like(self.dbcc)

		if self.gdat.float_fourier_comps:
			if self.gdat.init_fourier_coeffs is not None:
				self.fourier_coeffs = self.gdat.init_fourier_coeffs.copy()
			self.fourier_templates = self.gdat.fc_templates
			
			if self.gdat.bkg_moore_penrose_inv:

				verbprint(self.gdat.verbtype, 'Computing Moore Penrose inverse of Fourier components..')
				self.dat.data_array[0] -= np.nanmean(self.dat.data_array[0]) # this is done to isolate the fluctuation component
				self.bkg[0] = 0.

				_, ravel_temps, bt_siginv_b, bt_siginv_b_inv, mp_coeffs, temp_A_hat, nanmask = compute_marginalized_templates(self.gdat.MP_order, self.dat.data_array[0], self.dat.uncertainty_maps[0],\
														  ridge_fac=self.gdat.ridge_fac, ridge_fac_alpha=self.gdat.ridge_fac_alpha, return_temp_A_hat=False, \
														  fourier_templates = self.fourier_templates[0], show=False)

				# save precomputed cov matrices for fast evaluation later
				self.bt_siginv_b_inv = bt_siginv_b_inv 
				self.bt_siginv_b = bt_siginv_b
				self.ravel_temps = ravel_temps
				# use best fit coefficients to initialize Fourier component model
				self.fourier_coeffs[:self.gdat.MP_order, :self.gdat.MP_order, :] = mp_coeffs.copy()

			self.n_fourier_terms = self.gdat.n_fourier_terms
			self.dfc = np.zeros((self.n_fourier_terms, self.n_fourier_terms, 4))
			self.dfc_rel_amps = np.zeros((gdat.nbands))
			self.fc_rel_amps = self.gdat.fc_rel_amps
		else:
			self.fc_rel_amps = None
			self.fourier_coeffs = None
		
		if self.gdat.nsrc_init is not None:
			self.n = self.gdat.nsrc_init
		else:
			self.n = np.random.randint(gdat.max_nsrc)+1

		self.nbands = gdat.nbands
		self.nloop = gdat.nloop
		self.nominal_nsrc = gdat.nominal_nsrc
		self.nregion = gdat.nregion
		self.offsetxs = np.zeros(self.nbands).astype(np.int)
		self.offsetys = np.zeros(self.nbands).astype(np.int)
		
		self.penalty = (2.+gdat.nbands)*0.5*gdat.alph
		print('Parsimony prior is set to dlogP = '+str(self.penalty)+' per source..')

		self.regions_factor = gdat.regions_factor
		self.regsizes = np.array(gdat.regsizes).astype(np.int)
		
		self.stars = np.zeros((2+gdat.nbands,gdat.max_nsrc), dtype=np.float32)
		self.stars[:,0:self.n] = np.random.uniform(size=(2+gdat.nbands,self.n))
		self.stars[self._X,0:self.n] *= gdat.imsz0[0]-1
		self.stars[self._Y,0:self.n] *= gdat.imsz0[1]-1

		self.truealpha = gdat.truealpha

		# additional parameters for double power law
		self.alpha_1 = gdat.alpha_1
		self.alpha_2 = gdat.alpha_2
		self.pivot_dpl = gdat.pivot_dpl
		self.trueminf = gdat.trueminf
		self.verbtype = gdat.verbtype

		self.bkg_prop_sigs = np.array([self.gdat.bkg_sig_fac[b]*np.nanmedian(self.dat.uncertainty_maps[b][self.dat.uncertainty_maps[b]>0])/np.sqrt(self.dat.fracs[b]*self.imszs[b][0]*self.imszs[b][1]) for b in range(gdat.nbands)])

		if gdat.bkg_prior_mus is not None:
			self.bkg_prior_mus = gdat.bkg_prior_mus
		else:
			self.bkg_prior_mus = self.bkg.copy()
		self.bkg_prior_sig = gdat.bkg_prior_sig
		self.dback = np.zeros_like(self.bkg)

		for b in range(self.nbands-1):

			col_string = self.gdat.band_dict[self.gdat.bands[0]]+'-'+self.gdat.band_dict[self.gdat.bands[b+1]]
			self.color_mus.append(self.gdat.color_mus[col_string])
			self.color_sigs.append(self.gdat.color_sigs[col_string])
			
		print('Mean colors (prior, in magnitudes): ', self.color_mus)
		print('Color prior widths (in magnitudes): ', self.color_sigs)

		# unless previous model state provided to PCAT (for example, from a previous run), draw fluxes from specified flux+color priors.
		if gdat.load_state_timestr is None:
			for b in range(gdat.nbands):
				if b==0:
					self.draw_fluxes()
				else:
					new_colors = np.random.normal(loc=self.color_mus[b-1], scale=self.color_sigs[b-1], size=self.n)
					self.stars[self._F+b,0:self.n] = self.stars[self._F,0:self.n]*10**(0.4*new_colors)
		else:
			self.load_previous_model()

	def draw_fluxes(self, flux_prior_type=None):

		"""

		Parameters
		----------
		flux_prior_type : str
			{'single_power_law', 'double_power_law'}

		This is an in place operation.

		"""
		if flux_prior_type is None:
			flux_prior_type = self.gdat.flux_prior_type

		if flux_prior_type=='double_power_law':
			verbprint(self.gdat.verbtype, 'Using double power law flux prior: a1, a2 = '+str(self.alpha_1), +', '+str(self.alpha_2), verbthresh=1)
			self.stars[self._F+b,0:self.n] = icdf_dpow(self.stars[self._F+b,0:self.n],\
														 self.trueminf, self.trueminf*self.newsrc_minmax_range, \
														 self.pivot_dpl, self.alpha_1, self.alpha_2)
		elif flux_prior_type=='single_power_law':
			verbprint(self.gdat.verbtype, 'Using single power law flux prior: alpha = '+str(self.truealpha), verbthresh=1)
			self.stars[self._F+b,0:self.n] **= -1./(self.truealpha - 1.)
			self.stars[self._F+b,0:self.n] *= self.trueminf


	def load_previous_model(self):
		""" 
		
		This function parses previously saved PCAT run and instantiates current run with loaded model parameters.

		"""

		catpath = self.gdat.result_path+'/'+self.gdat.load_state_timestr+'/final_state.npz'
		catload = np.load(catpath)
		gdat_previous, _, _ = load_param_dict(self.gdat.load_state_timestr, result_path=self.gdat.result_path)
		previous_cat = catload['cat']

		self.n = np.count_nonzero(previous_cat[self._F,:])
		if self.gdat.float_background:
			for b in range(gdat_previous.nbands):
				self.bkg[b] = catload['bkg'][b]
		
		if self.gdat.float_templates:
			print('self template amplitudes is ', self.template_amplitudes)
			if gdat_previous.nbands == gdat.nbands:
				self.template_amplitudes=catload['templates']
			else:
				for t in range(self.gdat.n_templates):
					for b in range(gdat_previous.nbands):
						self.template_amplitudes[t, b] = catload['templates'][t,b]

		if self.gdat.float_fourier_comps:
			self.gdat.fourier_coeffs = catload['fourier_coeffs']

		if gdat_previous.nbands == gdat.nbands:
			print('Same number of bands, setting catalog equal to previous catalog..')
			self.stars = previous_cat
		else:
			print('Different number of bands between previous and current run, drawing colors for remaining bands..')
			self.stars[self._X,:] = previous_cat[self._X,:]
			self.stars[self._Y,:] = previous_cat[self._Y,:]
			for b in range(gdat.nbands):
				if gdat_previous.nbands > b:
					self.stars[self._F+b,:] = previous_cat[self._F+b,:]
				else:
					print('Drawing colors on band '+str(b)+'..')
					new_colors = np.random.normal(loc=self.color_mus[b-1], scale=self.color_sigs[b-1], size=self.n)
					self.stars[self._F+b,0:self.n] = self.stars[self._F,0:self.n]*10**(0.4*new_colors)


	def update_moveweights(self, j):
		""" 
		During the burn in stage of sampling, this function gets used to update the proposals PCAT draws from with the specified weights. 
		"""
		# moveweight_idx_dict = config.moveweight_idx_dict
		# moveweight_idx_dict = dict({'movestar':0, 'birth_death':1, 'merge_split':2, 'bkg':3, 'template':4, 'fourier_comp':5, 'binned_cib':6}) # binned cib
		
		# running_[..] refers to the weights across the full run, so if proposals are delayed this accounts for that.
		running_prop_dict = dict({})

		for p, prop_name in enumerate(prop_names):

			prop_dict[prop_name] = dict({})
			prop_dict[prop_name]['sample_delays'] = self.gdat.sample_delays[prop_name]
			prop_dict[prop_name]['moveweight'] = self.gdat.moveweights[prop_name]
			prop_dict[prop_name]['proposal_bools'] = self.gdat.prop_bools[prop_name]
		# proposal_bools = dict({0:True, 1:True, 2:True, 3:self.gdat.float_background, 4:self.gdat.float_templates, 5:self.gdat.float_fourier_comps, 6:self.gdat.float_cib_templates})

		proposal_bool_dict = dict({'movestar':True, 'birth_death':True, 'merge_split':True, 'bkg':self.gdat.float_background, 'template':self.gdat.float_templates, 'fc':self.gdat.float_fourier_comps, 'bcib':self.gdat.float_cib_templates})
		# key_list, val_list = list(.keys()), list(moveweight_idx_dict.values())
		for k, movekey in enumerate(proposal_bool_dict.keys()):
			if j == self.sample_delay_byprop[movekey] and proposal_bool_dict[movekey]
				self.run_moveweights[movekey] = self.gdat.moveweight_byprop[movekey]
				self.run_moveweights[np.isnan(self.run_moveweights)] = 0.
				print('Proposal weights: ', self.moveweights, file=self.gdat.flog)

		# for moveidx, sample_delay in enumerate(self.sample_delays):
		# 	if j == sample_delay and proposal_bools[moveidx]:
		# 		print('Including '+str(key_list[val_list.index(moveidx)])+' proposals')
		# 		self.moveweights[moveidx] = moveweight_dict[moveidx]
		# 		self.moveweights[np.isnan(self.moveweights)] = 0.
		# 		print('Proposal weights: ', self.moveweights, file=self.gdat.flog)

	def normalize_weights(self, weights):
		""" 
		This is used when updating proposal weights during burn-in.
		"""
		return weights / np.sum(weights)
   
	def print_sample_status(self, dts, accept, outbounds, chi2, movetype, bkg_perturb_band_idxs=None, temp_perturb_band_idxs=None):  
		"""
		This function prints out some information at the end of each thinned sample, 
		namely acceptance fractions for the different proposals and some time performance statistics as well. 

		Parameters
		----------
		
		dts : 
		accept, outbounds: array_like
		chi2 :
		movetype :
		bkg_perturb_band_idxs : 
		temp_perturb_band_idxs : 
	

		"""  
		fmtstr = '\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f'
		print('Background '+str(np.round(self.bkg, 5)) + ', N_star '+str(self.n)+' chi^2 '+str(list(np.round(chi2, 2))))
		print('Reduced chi^2 ', [np.round(chi2[b]/(self.dat.fracs[b]*self.dat.data_array[b].shape[0]*self.dat.data_array[b].shape[1]), 2) for b in range(self.gdat.nbands)])

		dts *= 1000
		accept_fracs = []
		timestat_array = np.zeros((6, 1+len(self.moveweights)), dtype=np.float32)
		statlabels = ['Acceptance', 'Out of Bounds', 'Proposal (s)', 'Likelihood (s)', 'Implement (s)', 'Coordinates (s)']
		statarrays = [accept, outbounds, dts[0,:], dts[1,:], dts[2,:], dts[3,:]]

		if bkg_perturb_band_idxs is not None:
			per_band_bkg_acpt = []
			bkg_all_acpts = np.array(statarrays[0][movetype==3])
			for b in range(self.gdat.nbands):
				per_band_bkg_acpt.append(np.mean(bkg_all_acpts[bkg_perturb_band_idxs==b]))
			print('Per band background accept : ', np.round(per_band_bkg_acpt, 3))

		if temp_perturb_band_idxs is not None:
			per_band_temp_acpt = []
			temp_all_acpts = np.array(statarrays[0][movetype==4])
			for b in range(self.gdat.nbands):
				per_band_temp_acpt.append(np.mean(temp_all_acpts[temp_perturb_band_idxs==b]))
			print('Per band SZ accept : ', np.round(per_band_temp_acpt, 3))


		for j in range(len(statlabels)):
			timestat_array[j][0] = np.sum(statarrays[j])/1000
			if j==0:
				accept_fracs.append(np.sum(statarrays[j])/1000)
			print(statlabels[j]+'\t(all) %0.3f' % (np.sum(statarrays[j])/1000), file=self.gdat.flog)
			for k in range(len(self.run_movetypes)):
				if j==0:
					accept_fracs.append(np.mean(statarrays[j][movetype==k]))
				timestat_array[j][1+k] = np.mean(statarrays[j][movetype==k])
				print('('+self.run_movetypes[k]+') %0.3f' % (np.mean(statarrays[j][movetype == k])), end=' ', file=self.gdat.flog)
			print(file=self.gdat.flog)
			if j == 1:
				print('-'*16, file=self.gdat.flog)
		print('-'*16, file=self.gdat.flog)
		print('Total (s): %0.3f' % (np.sum(statarrays[2:])/1000), file=self.gdat.flog)
		print('='*16, file=self.gdat.flog)

		return timestat_array, accept_fracs


	def pcat_multiband_eval(self, x, y, f, bkg, nc, cf, weights, ref, lib, beam_fac=1., margin_fac=1, \
		dtemplate=None, rtype=None, dfc=None, idxvecs=None, precomp_temps=None, fc_rel_amps=None, \
		perturb_band_idx=None, dbcc = None, bcib_idx = None):
		
		"""
		Wrapper for multiband likelihood evaluation given catalog model parameters.


		Inputs
		------
		x, y : 
		f : 
		bkg : 
		nc : 
		cf : 
		weights : 
		ref : 
		lib : 
		beam_fac : 
			Default is 1.
		margin_fac : 
			Default is 1.
		dtemplate : 
			Default is None.
		rtype : 
			Default is None.
		dfc : 
			Default is None.
		idxvecs : 
			Default is None.
		precomp_temps : 
			Default is None.
		fc_rel_amps :
			Default is None.
		perturb_band_idx : 
			Default is None.
		dbcc : 
			Default is None.
		bcib_idx : 
			Default is None

		Returns
		-------

		dmodels : 
		diff2s : 
		dt_transf : 

		"""

		dmodels = []
		dt_transf = 0
		nb = 0

		for b in range(self.nbands):
			dtemp = None

			if perturb_band_idx is not None:
				if b != perturb_band_idx:
					dmodels.append(None)
					continue

			if dtemplate is not None:
				dtemp = []
				for i, temp in enumerate(self.dat.template_array[b]):
					verbprint(self.gdat.verbtype, 'dtemplate in multiband eval is '+str(dtemplate.shape), verbthresh=1)
					if temp is not None and dtemplate[i][b] != 0.:
						dtemp.append(dtemplate[i][b]*temp)
				if len(dtemp) > 0:
					dtemp = np.sum(np.array(dtemp), axis=0).astype(np.float32)
				else:
					dtemp = None

			if precomp_temps is not None:
				pc_temp = precomp_temps[b]

				if fc_rel_amps is not None:
					pc_temp *= fc_rel_amps[b]

				# if passing fixed fourier comp template, fc_rel_amps should be model + d_rel_amps, if perturbing
				# relative amplitude, fc_rel_amps should be one hot vector with change in one of the bands
				
				if dtemp is None:
					dtemp = pc_temp
				else:
					dtemp += pc_temp


			elif dfc is not None:

				if idxvecs is not None:
					for i in range(self.gdat.n_fc_perturb): # default n_fc_perturb = 1, not clear that perturbing several at a time improves convergence
						pc_temp = self.fourier_templates[b][idxvecs[i][0], idxvecs[i][1], idxvecs[i][2]]*dfc[idxvecs[i][0], idxvecs[i][1], idxvecs[i][2]]
						if dtemp is None:
							dtemp = fc_rel_amps[b]*pc_temp
						else:
							dtemp += fc_rel_amps[b]*pc_temp

				else:
					pc_temp = np.sum([dfc[i,j,k]*self.fourier_templates[b][i,j,k] for i in range(self.n_fourier_terms) for j in range(self.n_fourier_terms) for k in range(4)], axis=0)
					if dtemp is None:
						dtemp = fc_rel_amps[b]*pc_temp
					else:
						dtemp += fc_rel_amps[b]*pc_temp

			# binned cib templates
			# if dbcc is not None:

			# 	if bcib_idx is not None:
			# 		pc_bcib_temp = self.coarse_cib_templates[b][bcib_idx]*dbcc[bcib_idx]
			# 	else:
			# 		pc_bcib_temp = np.dot(dbcc, self.coarse_cib_templates[b])
			# 		plot_single_map(pc_bcib_temp, title='Summed CIB template, b='+str(b))

			# 	if dtemp is None:
			# 		dtemp = pc_bcib_temp 
			# 	else:
			# 		dtemp += pc_bcib_temp

			if b>0:
				t4 = time.time()
				if self.gdat.bands[b] != self.gdat.bands[0]:
					xp, yp = self.dat.fast_astrom.transform_q(x, y, b-1)
				else:
					xp = x
					yp = y
				dt_transf += time.time()-t4
			else:
				xp=x
				yp=y

			dmodel, diff2 = image_model_eval(xp, yp, beam_fac[b]*nc[b]*f[b], bkg[b], self.imszs[b], \
											nc[b], np.array(cf[b]).astype(np.float32()), weights=self.dat.weights[b], \
											ref=ref[b], lib=lib, regsize=self.regsizes[b], \
											margin=self.margins[b]*margin_fac, offsetx=self.offsetxs[b], offsety=self.offsetys[b], template=dtemp)
			
			if nb==0:
				diff2s = diff2
				nb += 1
			else:
				diff2s += diff2

			dmodels.append(dmodel)

		return dmodels, diff2s, dt_transf 


	def run_sampler(self, sample_idx):
		""" 
		Main wrapper function for executing the calculation of a thinned sample in PCAT.
		run_sampler() completes nloop samples, so the function gets called 'nsamp' times in a full run.
	
		Parameters
		----------

		sample_idx : 'int'. Thinned sample index.

		Returns
		-------

		n : 'int'. Number of sources.
		chi2 : 'np.array' of type 'float' with shape (nbands,). Image model chi squared for each band.
		timestat_array : 'np.array' of type 'float'.
		accept_fracs : 'np.array' of type 'float'.
		diff2_list : 
		rtype_array : 'np.array' of type 'float' and shape (nloop,). Proposal types for nloop samples corresponding to thinned sample 'sample_idx'. 
		accept : 'np.array' of type 'bool' and shape (nloop,). Booleans indicate whether or not the proposals were accepted or not.
		resids : 'list' of 'np.arrays' of type 'float' and shape (nbands, dimx, dimy). Residual maps at end of thinned sample.
		models : 'list' of 'np.arrays' of type 'float' and shape (nbands, dimx, dimy). Model images at end of thinned sample.

		"""
		
		t0 = time.time()

		nmov, movetype, accept, outbounds, diff2_list = [np.zeros(self.nloop) for x in range(5)]
		dts = np.zeros((4, self.nloop)) # array to store time spent on different proposals

		if self.nregion > 1:
			self.offsetxs[0] = np.random.randint(self.regsizes[0])
			self.offsetys[0] = np.random.randint(self.regsizes[0])
			self.margins[0] = self.gdat.margin
			
			for b in range(self.gdat.nbands - 1):
				reg_ratio = float(self.imszs[b+1][0])/float(self.imszs[0][0])
				self.offsetxs[b+1] = int(self.offsetxs[0]*reg_ratio)
				self.offsetys[b+1] = int(self.offsetys[0]*reg_ratio)
				self.margins[b+1] = int(self.margins[0]*reg_ratio)
				verbprint(self.gdat.verbtype, str(self.offsetxs[b+1])+', '+str(self.offsetys[b+1])+', '+str(self.margins[b+1]), verbthresh=1)

		else:
			self.offsetxs = np.array([0 for b in range(self.gdat.nbands)])
			self.offsetys = np.array([0 for b in range(self.gdat.nbands)])

		self.nregx = int(self.imsz0[0] / self.regsizes[0] + 1)
		self.nregy = int(self.imsz0[1] / self.regsizes[0] + 1)

		resids = []
		for b in range(self.nbands):
			resid = self.dat.data_array[b].copy() # residual for zero image is data
			verbprint(self.gdat.verbtype, 'resid has shape '+str(resid.shape), verbthresh=1)
			resids.append(resid)

		evalx = self.stars[self._X,0:self.n]
		evaly = self.stars[self._Y,0:self.n]
		evalf = self.stars[self._F:,0:self.n]		
		n_phon = evalx.size

		verbprint(self.gdat.verbtype, 'Beginning of run sampler', verbthresh=1)
		verbprint(self.gdat.verbtype, 'self.n here is '+str(self.n), verbthresh=1)
		verbprint(self.gdat.verbtype, 'n_phon = '+str(n_phon), verbthresh=1)

		if self.gdat.cblas:
			lib = self.libmmult.pcat_model_eval
		else:
			lib = self.libmmult.clib_eval_modl

		dtemplate, fcoeff, running_temp = None, None, None

		if self.gdat.float_templates:
			dtemplate = self.template_amplitudes

		if self.gdat.float_fourier_comps or self.gdat.float_cib_templates:
			running_temp = []
			for b in range(self.nbands):
				running_temp.append(np.zeros(self.imszs[b]))
				if self.gdat.float_fourier_comps:
					running_temp[b] += np.sum([self.fourier_coeffs[i,j,k]*self.fourier_templates[b][i,j,k] for i in range(self.n_fourier_terms) for j in range(self.n_fourier_terms) for k in range(4)], axis=0)

				# if self.gdat.float_cib_templates:
				# 	running_temp[b] += np.sum([self.binned_cib_coeffs[i]*self.coarse_cib_templates[b][i] for i in range(self.gdat.cib_nregion**2)], axis=0)


		models, diff2s, dt_transf = self.pcat_multiband_eval(evalx, evaly, evalf, self.bkg, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, ref=resids, lib=lib, beam_fac=self.pixel_per_beam, \
														 dtemplate=dtemplate, precomp_temps=running_temp, fc_rel_amps=self.fc_rel_amps)

		logL = -0.5*diff2s

		verbprint(self.verbtype, 'logL here is ', np.sum(logL), verbthresh=1)
	   
		for b in range(self.nbands):
			resids[b] -= models[b]

		"""the proposals here are: move_stars (P) which changes the parameters of existing model sources, 
		birth/death (BD) and merge/split (MS). Don't worry about perturb_astrometry. 
		The moveweights array, once normalized, determines the probability of choosing a given proposal. """
		
		movefns = [self.move_stars, self.birth_death_stars, self.merge_split_stars, self.perturb_background, \
						self.perturb_template_amplitude, self.perturb_fourier_comp, self.perturb_binned_cib_coeff] 

		if self.gdat.nregion > 1:
			xparities = np.random.randint(2, size=self.nloop)
			yparities = np.random.randint(2, size=self.nloop)

		rtype_array = np.random.choice(self.moveweights.size, p=self.normalize_weights(self.moveweights), size=self.nloop)

		movetype = rtype_array

		bkg_perturb_band_idxs, temp_perturb_band_idxs = [], [] # used for per-band acceptance fractions

		for i in range(self.nloop):
			t1 = time.time()
			rtype = rtype_array[i]
			
			verbprint(self.verbtype, 'rtype = '+str(rtype), verbthresh=1)

			if self.nregion > 1:
				self.parity_x = xparities[i] # should regions be perturbed randomly or systematically?
				self.parity_y = yparities[i]
			else:
				self.parity_x = 0
				self.parity_y = 0

			#proposal types
			proposal = movefns[rtype]()

			dts[0,i] = time.time() - t1
			
			if proposal.goodmove:
				t2 = time.time()

				if self.gdat.cblas:
					lib = self.libmmult.pcat_model_eval
				else:
					lib = self.libmmult.clib_eval_modl

				dtemplate, fcoeff, bkg, fc_rel_amps = None, None, None, None

				if self.gdat.float_templates:
					dtemplate = self.template_amplitudes+self.dtemplate
				if self.gdat.float_fourier_comps:
					fcoeff = self.fourier_coeffs+self.dfc
					fc_rel_amps=self.fc_rel_amps+self.dfc_rel_amps

				if self.gdat.float_background:
					bkg = self.bkg + self.dback
				else:
					bkg = self.bkg

				margin_fac = 1
				if rtype > 2:
					margin_fac = 0

				# test this
				if rtype>=3: # mean normalization/template proposals
					# recompute model likelihood with margins set to zero, use current values of star parameters and use background level equal to self.bkg (+self.dback up to this point)
					if rtype==3:
						bkg_perturb_band_idxs.append(proposal.perturb_band_idx)
					if rtype==4:
						temp_perturb_band_idxs.append(proposal.perturb_band_idx)

					if rtype==5 or rtype==6: # fourier components or binned cib templates
						perturb_band_idx = None 
					else:
						perturb_band_idx = proposal.perturb_band_idx

					mods, diff2s_nomargin, dt_transf = self.pcat_multiband_eval(self.stars[self._X,0:self.n], self.stars[self._Y,0:self.n], self.stars[self._F:,0:self.n], \
											bkg, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, ref=self.dat.data_array, lib=lib, \
											beam_fac=self.pixel_per_beam, margin_fac=margin_fac, rtype=rtype, dtemplate=dtemplate, precomp_temps=running_temp, fc_rel_amps=fc_rel_amps, \
											perturb_band_idx=perturb_band_idx)

					logL = -0.5*diff2s_nomargin



				if rtype == 3: # background

					dmodels, diff2s, dt_transf = self.pcat_multiband_eval(proposal.xphon, proposal.yphon, proposal.fphon, proposal.dback, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, \
													ref=resids, lib=lib, beam_fac=self.pixel_per_beam, margin_fac=margin_fac, rtype=rtype, perturb_band_idx=proposal.perturb_band_idx)
	
					# plot_multipanel([mods[0], dmodels[0]], figsize=(12, 5), cmap='Greys')

				elif rtype == 4: # template

					dmodels, diff2s, dt_transf = self.pcat_multiband_eval(proposal.xphon, proposal.yphon, proposal.fphon, proposal.dback, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, \
													ref=resids, lib=lib, beam_fac=self.pixel_per_beam, margin_fac=margin_fac, dtemplate=proposal.dtemplate, rtype=rtype, \
													perturb_band_idx=proposal.perturb_band_idx)
	

				elif rtype == 5: # fourier comp

					if proposal.fc_rel_amp_bool:

						dmodels, diff2s, dt_transf = self.pcat_multiband_eval(proposal.xphon, proposal.yphon, proposal.fphon, proposal.dback, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, \
														ref=resids, lib=lib, beam_fac=self.pixel_per_beam, margin_fac=margin_fac, rtype=rtype, precomp_temps=running_temp, fc_rel_amps=proposal.dfc_rel_amps)
		
					else:
						
						# nfcperturb = 1 test
						idxvecs = [[proposal.idxs0[n], proposal.idxs1[n], proposal.idxsk[n]] for n in range(self.gdat.n_fc_perturb)]

						dmodels, diff2s, dt_transf = self.pcat_multiband_eval(proposal.xphon, proposal.yphon, proposal.fphon, proposal.dback, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, \
														ref=resids, lib=lib, beam_fac=self.pixel_per_beam, margin_fac=margin_fac, rtype=rtype, dfc=proposal.dfc, idxvecs=idxvecs, fc_rel_amps=fc_rel_amps)
		
						
				# elif rtype == 6: # binned cib

				# 	mods, diff2s_nomargin, dt_transf = self.pcat_multiband_eval(self.stars[self._X,0:self.n], self.stars[self._Y,0:self.n], self.stars[self._F:,0:self.n], \
				# 																bkg, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, ref=self.dat.data_array, lib=lib, \
				# 																beam_fac=self.pixel_per_beam, margin_fac=margin_fac, dtemplate=dtemplate, rtype=rtype, precomp_temps=running_temp, fc_rel_amps=fc_rel_amps)
										
				# 	logL = -0.5*diff2s_nomargin
	


				# 	dmodels, diff2s, dt_transf = self.pcat_multiband_eval(proposal.xphon, proposal.yphon, proposal.fphon, proposal.dback, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, \
				# 									ref=resids, lib=lib, beam_fac=self.pixel_per_beam, margin_fac=margin_fac, rtype=rtype, dbcc=proposal.dbcc, bcib_idx = proposal.bcib_idx)
	
					# plot_multipanel([mods[0], dmodels[0]], figsize=(12, 5), cmap='Greys')



				else: # movestar, birth/death, merge/split

					dmodels, diff2s, dt_transf = self.pcat_multiband_eval(proposal.xphon, proposal.yphon, proposal.fphon, proposal.dback, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, \
													ref=resids, lib=lib, beam_fac=self.pixel_per_beam, margin_fac=margin_fac, rtype=rtype)
	
				

				plogL = -0.5*diff2s  

				if rtype < 3:
					plogL[(1-self.parity_y)::2,:] = float('-inf') # don't accept off-parity regions
					plogL[:,(1-self.parity_x)::2] = float('-inf')

				
				dlogP = plogL - logL

				# if rtype==3:
					# print('dlogP is ', dlogP)


				assert np.isnan(dlogP).any() == False
				
				dts[1,i] = time.time() - t2
				t3 = time.time()
				
				if rtype < 3:
					refx, refy = proposal.get_ref_xy()

					regionx = get_region(refx, self.offsetxs[0], self.regsizes[0])
					regiony = get_region(refy, self.offsetys[0], self.regsizes[0])

					verbprint(self.verbtype, 'Proposal factor has shape '+str(proposal.factor.shape), verbthresh=1)
					verbprint(self.verbtype, 'Proposal factor = '+str(proposal.factor), verbthresh=1)
					
					if proposal.factor is not None:
						dlogP[regiony, regionx] += proposal.factor
					else:
						print('proposal factor is None')


				if rtype < 3:
					acceptreg = (np.log(np.random.uniform(size=(self.nregy, self.nregx))) < dlogP).astype(np.int32)

					acceptprop = acceptreg[regiony, regionx]
					numaccept = np.count_nonzero(acceptprop)

				else:
					# if background proposal:
					# sum up existing logL from subregions

					total_logL = np.sum(logL)
					total_dlogP = np.sum(dlogP)

					# if rtype==6:
					# 	print('total logL is ', total_logL)
					# 	print('total dlogP is ', total_dlogP)

					if proposal.factor is not None:

						if np.abs(proposal.factor) > 100:
							print('whoaaaaa proposal.factor is ', proposal.factor)
						total_dlogP += proposal.factor

					# compute dlogP over the full image
					# compute acceptance
					accept_or_not = (np.log(np.random.uniform()) < total_dlogP).astype(np.int32)

					if accept_or_not:
						# set all acceptreg for subregions to 1
						acceptreg = np.ones(shape=(self.nregy, self.nregx)).astype(np.int32)

						if total_dlogP < -10.:
							print('the chi squared degraded significantly in this proposal')
						# 	print('delta log likelihood:', total_dlogP-proposal.factor)
						# 	print('proposal.factor:', proposal.factor)
					else:
						acceptreg = np.zeros(shape=(self.nregy, self.nregx)).astype(np.int32)

				
				nb = 0 # index used for perturb_band_idx stuff

				""" for each band compute the delta log likelihood between states, then add these together"""
				for b in range(self.nbands):

					if proposal.perturb_band_idx is not None:
						if b != proposal.perturb_band_idx:
							continue

					dmodel_acpt = np.zeros_like(dmodels[b])
					diff2_acpt = np.zeros_like(diff2s)

					# if self.gdat.coupled_bkg_prop and rtype==3:
						# plot_single_map(dmodels[b])
	

					if self.gdat.cblas:

						self.libmmult.pcat_imag_acpt(self.imszs[b][0], self.imszs[b][1], dmodels[b], dmodel_acpt, acceptreg, self.regsizes[b], self.margins[b], self.offsetxs[b], self.offsetys[b])
						# using this dmodel containing only accepted moves, update logL
						self.libmmult.pcat_like_eval(self.imszs[b][0], self.imszs[b][1], dmodel_acpt, resids[b], self.dat.weights[b], diff2_acpt, self.regsizes[b], self.margins[b], self.offsetxs[b], self.offsetys[b])   
					else:
						
						self.libmmult.clib_updt_modl(self.imszs[b][0], self.imszs[b][1], dmodels[b], dmodel_acpt, acceptreg, self.regsizes[b], self.margins[b], self.offsetxs[b], self.offsetys[b])
						# using this dmodel containing only accepted moves, update logL
						self.libmmult.clib_eval_llik(self.imszs[b][0], self.imszs[b][1], dmodel_acpt, resids[b], self.dat.weights[b], diff2_acpt, self.regsizes[b], self.margins[b], self.offsetxs[b], self.offsetys[b])   

					# if rtype==3 and accept_or_not and total_dlogP < 0:
						# print('total_dlogP:', total_dlogP, 'proposal factor is ', proposal.factor)
						# plot_single_map(dmodel_acpt-dmodels[b])

					resids[b] -= dmodel_acpt
					models[b] += dmodel_acpt

					# if rtype==5:
						# plot_single_map(models[b], title='models['+str(b)+']')
						# plot_single_map(dmodel_acpt, title='dmodels_acpt['+str(b)+']')

					if nb==0:
						diff2_total1 = diff2_acpt
						nb += 1
					else:
						diff2_total1 += diff2_acpt


				logL = -0.5*diff2_total1

				# if rtype==5:
				# 	print('logL here is ', logL)

				#implement accepted moves
				if proposal.idx_move is not None:

					if rtype==3:

						self.stars = proposal.starsp

					elif self.gdat.coupled_profile_temp_prop and rtype==4:

						if accept_or_not: 
							acceptprop = np.zeros((self.stars.shape[1],))
							acceptprop[proposal.idx_move] = 1
							starsp = proposal.starsp.compress(acceptprop, axis=1)
							self.stars[:, proposal.idx_move] = starsp

					elif self.gdat.coupled_fc_prop and rtype==5:

						if accept_or_not: 
							acceptprop = np.zeros((self.stars.shape[1],))
							acceptprop[proposal.idx_move] = 1
							starsp = proposal.starsp.compress(acceptprop, axis=1)
							self.stars[:, proposal.idx_move] = starsp

					else:
						starsp = proposal.starsp.compress(acceptprop, axis=1)
						idx_move_a = proposal.idx_move.compress(acceptprop)
						self.stars[:, idx_move_a] = starsp

				
				if proposal.do_birth:
					starsb = proposal.starsb.compress(acceptprop, axis=1)
					starsb = starsb.reshape((2+self.nbands,-1))
					num_born = starsb.shape[1]
					self.stars[:, self.n:self.n+num_born] = starsb
					self.n += num_born

				if proposal.idx_kill is not None:
					idx_kill_a = proposal.idx_kill.compress(acceptprop, axis=0).flatten()
					num_kill = idx_kill_a.size
				   
					# nstar is correct, not n, because x,y,f are full nstar arrays
					self.stars[:, 0:self.max_nsrc-num_kill] = np.delete(self.stars, idx_kill_a, axis=1)
					self.stars[:, self.max_nsrc-num_kill:] = 0
					self.n -= num_kill

				if proposal.change_bkg_bool:
					if np.sum(acceptreg) > 0:
						self.dback += proposal.dback

				if proposal.change_template_amp_bool:

					if np.sum(acceptreg) > 0:
						if proposal.change_binned_cib_bool:
							self.dbcc += proposal.dbcc
							for b in range(self.nbands):
								running_temp[b] += self.coarse_cib_templates[b][proposal.bcib_idx]*proposal.dbcc[proposal.bcib_idx]
						else:
							self.dtemplate += proposal.dtemplate

				if proposal.change_fourier_comp_bool:

					if np.sum(acceptreg) > 0:
						if proposal.fc_rel_amp_bool:
							self.dfc_rel_amps += proposal.dfc_rel_amps
						else:
							self.dfc += proposal.dfc

							for b in range(self.nbands):

								for n in range(self.gdat.n_fc_perturb):
									# fcmap = self.fourier_templates[b][proposal.idxs0[n], proposal.idxs1[n], proposal.idxsk[n]]*proposal.dfc[proposal.idxs0[n], proposal.idxs1[n], proposal.idxsk[n]]
									# plot_single_map(fcmap, title='n = '+str(n))
									running_temp[b] += self.fourier_templates[b][proposal.idxs0[n], proposal.idxs1[n], proposal.idxsk[n]]*proposal.dfc[proposal.idxs0[n], proposal.idxs1[n], proposal.idxsk[n]]					

				dts[2,i] = time.time() - t3

				if rtype < 3:
					if acceptprop.size > 0:
						accept[i] = np.count_nonzero(acceptprop) / float(acceptprop.size)
					else:
						accept[i] = 0
				else:
					if np.sum(acceptreg)>0:
						accept[i] = 1
					else:
						accept[i] = 0
			
			else:
				verbprint(self.verbtype, 'Out of bounds..', verbthresh=1)
				outbounds[i] = 1

			for b in range(self.nbands):
				diff2_list[i] += np.sum(self.dat.weights[b]*(self.dat.data_array[b]-models[b])*(self.dat.data_array[b]-models[b]))


			verbprint(self.verbtype, 'End of loop '+str(i), verbthresh=1)		
			verbprint(self.verbtype, 'self.n = '+str(self.n), verbthresh=1)					
			verbprint(self.verbtype, 'Diff2 = '+str(diff2_list[i]), verbthresh=1)					
			
		# this is after nloop iterations
		chi2 = np.zeros(self.nbands)
		for b in range(self.nbands):
			chi2[b] = np.sum(self.dat.weights[b]*(self.dat.data_array[b]-models[b])*(self.dat.data_array[b]-models[b]))
			

		print('chi2 is ', chi2)
		print('logL is ', -chi2/2)
		verbprint(self.verbtype, 'End of sample. self.n = '+str(self.n), verbthresh=1)

		# save last of nloop samples to chain and initialize delta(parameters) to zero
		if self.gdat.float_templates:
			self.template_amplitudes += self.dtemplate 
			print('At the end of nloop, self.dtemplate is', self.dtemplate)
			print('so self.template_amplitudes are now ', self.template_amplitudes)
			self.dtemplate = np.zeros_like(self.template_amplitudes)

		if self.gdat.float_fourier_comps:
			self.fourier_coeffs += self.dfc 
			self.fc_rel_amps += self.dfc_rel_amps
			print('At the end of nloop, self.dfc_rel_amps is ', self.dfc_rel_amps)
			print('self.fc_rel_amps is ', self.fc_rel_amps)
			self.dfc = np.zeros_like(self.fourier_coeffs)
			self.dfc_rel_amps = np.zeros_like(self.fc_rel_amps)

		# if self.gdat.float_cib_templates:
		# 	self.binned_cib_coeffs += self.dbcc
		# 	self.dbcc = np.zeros_like(self.binned_cib_coeffs)

		self.bkg += self.dback
		print('At the end of nloop, self.dback is', np.round(self.dback, 4), 'so self.bkg is now ', np.round(self.bkg, 4))
		self.dback = np.zeros_like(self.bkg)

		timestat_array, accept_fracs = self.print_sample_status(dts, accept, outbounds, chi2, movetype, bkg_perturb_band_idxs=np.array(bkg_perturb_band_idxs), temp_perturb_band_idxs=np.array(temp_perturb_band_idxs))

		if self.gdat.visual:
			frame_dir_path = None

			if self.gdat.n_frames > 0:

				if sample_idx%(self.gdat.nsamp // self.gdat.n_frames)==0:
					frame_dir_path = self.gdat.frame_dir+'/sample_'+str(sample_idx)+'_of_'+str(self.gdat.nsamp)+'.png'


			fourier_any_bool = any(['fourier_bkg' in panel_name for panel_name in self.gdat.panel_list])
			bcib_any_bool = any(['bcib' in panel_name for panel_name in self.gdat.panel_list])

			fourier_bkg, bcib_bkg = None, None

			if self.gdat.float_fourier_comps and fourier_any_bool:
				fourier_bkg = [self.fc_rel_amps[b]*running_temp[b] for b in range(self.gdat.nbands)]
			# if self.gdat.float_cib_templates and bcib_any_bool:
			# 	bcib_bkg = [running_temp[b] for b in range(self.gdat.nbands)]

			if sample_idx < 50 or sample_idx%self.gdat.plot_sample_period==0:
				plot_custom_multiband_frame(self, resids, models, panels = self.gdat.panel_list, frame_dir_path = frame_dir_path, fourier_bkg = fourier_bkg, \
					bcib_bkg=bcib_bkg)

		return self.n, chi2, timestat_array, accept_fracs, diff2_list, rtype_array, accept, resids, models


	def idx_parity_stars(self):
		return idx_parity(self.stars[self._X,:], self.stars[self._Y,:], self.n, self.offsetxs[0], self.offsetys[0], self.parity_x, self.parity_y, self.regsizes[0])

	def bounce_off_edges(self, catalogue): # works on both stars and galaxies
		mask = catalogue[self._X,:] < 0
		catalogue[self._X, mask] *= -1
		mask = catalogue[self._X,:] > (self.imsz0[0] - 1)
		catalogue[self._X, mask] *= -1
		catalogue[self._X, mask] += 2*(self.imsz0[0] - 1)
		mask = catalogue[self._Y,:] < 0
		catalogue[self._Y, mask] *= -1
		mask = catalogue[self._Y,:] > (self.imsz0[1] - 1)
		catalogue[self._Y, mask] *= -1
		catalogue[self._Y, mask] += 2*(self.imsz0[1] - 1)
		# these are all in place operations, so no return value

	def in_bounds(self, catalogue):
		return np.logical_and(np.logical_and(catalogue[self._X,:] > 0, catalogue[self._X,:] < (self.imsz0[0] -1)), \
				np.logical_and(catalogue[self._Y,:] > 0, catalogue[self._Y,:] < self.imsz0[1] - 1))


	def perturb_background(self):
		""" 
		Perturb mean background level according to Model.bkg_prop_sigs.

		Returns
		-------
		proposal : Proposal class object.
	
		"""

		proposal = Proposal(self.gdat)
		# I want this proposal to return the original dback + the proposed change. If the proposal gets approved later on
		# then model.dback will be set to the updated state
		bkg_idx = np.random.choice(self.nbands)
		dback = np.random.normal(0., scale=self.bkg_prop_sigs[bkg_idx])

		proposal.dback[bkg_idx] = dback

		# this is to do coupled proposal between mean background normalization and point sources. The idea here is to compensate a change in background level with a
		# change in point source fluxes. All (or maybe select?) sources across the FOV are perturbed by N_eff*dback. 
		factor = None

		if self.gdat.dc_bkg_prior:
			bkg_factor = -(self.bkg[bkg_idx]+self.dback[bkg_idx]+proposal.dback[bkg_idx]- self.bkg_prior_mus[bkg_idx])**2/(2*self.bkg_prior_sig**2)
			bkg_factor += (self.bkg[bkg_idx]+self.dback[bkg_idx]-self.bkg_prior_mus[bkg_idx])**2/(2*self.bkg_prior_sig**2)
			if factor is not None:
				factor += bkg_factor
			else:
				factor = bkg_factor

		proposal.set_factor(factor)
		
		proposal.change_bkg(perturb_band_idx=bkg_idx)

		return proposal


	def perturb_fourier_comp(self): # fourier comp
		""" 
		Proposal to perturb amplitudes of Fourier component templates. The proposal width is determined by Model.temp_amplitude_sigs and can be
		scaled with the power law exponent of the assumed power law spectrum.

		I think for now I will only do the single band version of this, though the multiband is possible in principle.

		Note: At one point in the development I tried perturbing several templates at once, determined by n_fc_perturb. A naive implementation of this did not lead to any substantial improvements
		and so it is set to 1, but code is written in a way that can handle several templates at once.

		"""
		
		proposal = Proposal(self.gdat)
		factor = None

		# set dfc_prob to zero if you only want to perturb the amplitudes
		if np.random.uniform() < self.gdat.dfc_prob:

			# choose a component (or several) nfcperturb = 1 test

			proposal.idxs0 = np.random.randint(0, self.n_fourier_terms, self.gdat.n_fc_perturb)
			proposal.idxs1 = np.random.randint(0, self.n_fourier_terms, self.gdat.n_fc_perturb)
			proposal.idxsk = np.random.randint(0, 4, self.gdat.n_fc_perturb)

			fc_sig_fac = self.temp_amplitude_sigs['fc']		


			coeff_pert = fc_sig_fac*np.random.normal(0, 1, self.gdat.n_fc_perturb)/self.gdat.n_fc_perturb
			proposal.dfc[proposal.idxs0, proposal.idxs1, proposal.idxsk] = coeff_pert

		else:

			proposal.fc_rel_amp_bool=True
			band_weights = get_band_weights(self.gdat.fourier_band_idxs)
			band_idx = int(np.random.choice(self.gdat.fourier_band_idxs, p=band_weights))

			d_amp = np.random.normal(0, scale=self.fourier_amp_sig)
			proposal.dfc_rel_amps[band_idx] = d_amp 

		proposal.change_fourier_comp()

		return proposal

	# def perturb_binned_cib_coeff(self):

	# 	""" binned CIB template proposal """
	# 	proposal = Proposal(self.gdat)
	# 	proposal.dbcc = np.zeros((self.gdat.cib_nregion**2))
	# 	proposal.bcib_idx = np.random.choice(self.gdat.cib_nregion**2)

	# 	factor = None 

	# 	dbcc = np.random.normal(0., scale=self.temp_amplitude_sigs['binned_cib'])
	# 	# print('dbcc is ', dbcc)
	# 	proposal.dbcc[proposal.bcib_idx] = dbcc

	# 	# print('prroposal.dbcc is ', proposal.dbcc)

	# 	old_bcib_amp = self.binned_cib_coeffs[proposal.bcib_idx] +self.dbcc[proposal.bcib_idx]
	# 	new_bcib_amp = old_bcib_amp+dbcc		
	# 	if new_bcib_amp < 0:
	# 		proposal.goodmove = False

	# 		return proposal

	# 	proposal.change_template_amplitude()
	# 	proposal.change_binned_cib_bool = True

	# 	return proposal


	def perturb_template_amplitude(self):

		""" 
		Perturb (non-Fourier component) template amplitudes. These are being kept separate since it delineates between parametric/non-parametric models.
		For example, templates for the SZ effect are based on a parametric model, while cirrus or other diffuse emission can be fit with a non-parametric model, 
		with the note that you could have a full spatial model floated as one template (e.g., a Planck interpolated map of cirrus). 
		
		Returns
		-------

		proposal : Proposal class object.
			
		"""

		proposal = Proposal(self.gdat)
		proposal.dtemplate = np.zeros((self.gdat.n_templates, self.gdat.nbands))

		template_idx = np.random.choice(self.gdat.n_templates) # if multiple templates, choose one to change at a time
		temp_band_idxs = self.gdat.template_band_idxs[template_idx]
		factor = None

		# if self.gdat.temp_prop_df is not None:
		# 	d_amp = self.temp_amplitude_sigs[self.gdat.template_order[template_idx]]*np.random.standard_t(self.gdat.temp_prop_df)
		# else:
		d_amp = np.random.normal(0., scale=self.temp_amplitude_sigs[self.gdat.template_order[template_idx]])

		if self.gdat.delta_cp_bool and self.gdat.template_order[template_idx] != 'sze':
			if self.gdat.template_order[template_idx] == 'planck' or self.gdat.template_order[template_idx]=='dust':
				proposal.dtemplate[template_idx,:] = d_amp

		else:
			band_weights = get_band_weights(temp_band_idxs) # this function returns normalized weights

			# uncomment to institute DELTA FN PRIOR SZE @ 250 micron
			if self.gdat.template_order[template_idx] == 'sze':
				band_weights[0] = 0.
				band_weights /= np.sum(band_weights)

			band_idx = int(np.random.choice(temp_band_idxs, p=band_weights))

			proposal.dtemplate[template_idx, band_idx] = d_amp*self.gdat.temp_prop_sig_fudge_facs[band_idx] # added fudge factor for more efficient sampling
			proposal.perturb_band_idx = band_idx


		# the lines below are implementing a step function prior where the ln(prior) = -np.inf when the amplitude is negative
		if self.gdat.template_order[template_idx] == 'sze' and self.gdat.sz_positivity_prior:
			
			old_temp_amp = self.template_amplitudes[template_idx,band_idx] +self.dtemplate[template_idx, band_idx]
			new_temp_amp = old_temp_amp+proposal.dtemplate[template_idx,band_idx]
			
			if new_temp_amp < 0:
				proposal.goodmove = False
				return proposal

		proposal.change_template_amplitude()

		return proposal


	def flux_proposal(self, f0, nw, trueminf=None):
		if trueminf is None:
			trueminf = self.trueminf
		lindf = np.float32(self.err_f/(self.regions_factor*np.sqrt(self.gdat.nominal_nsrc*(2+self.nbands))))
		logdf = np.float32(0.01/np.sqrt(self.gdat.nominal_nsrc))
		ff = np.log(logdf*logdf*f0 + logdf*np.sqrt(lindf*lindf + logdf*logdf*f0*f0)) / logdf
		ffmin = np.log(logdf*logdf*trueminf + logdf*np.sqrt(lindf*lindf + logdf*logdf*trueminf*trueminf)) / logdf
		dff = np.random.normal(size=nw).astype(np.float32)
		aboveffmin = ff - ffmin
		oob_flux = (-dff > aboveffmin)
		dff[oob_flux] = -2*aboveffmin[oob_flux] - dff[oob_flux]
		pff = ff + dff
		pf = np.exp(-logdf*pff) * (-lindf*lindf*logdf*logdf+np.exp(2*logdf*pff)) / (2*logdf*logdf)
		return pf

	def eval_logp_dpl(self, fluxes):

		""" 
		Evaluate the log-prior of the flux distribution, parameterized with a double power law.

		Parameters
		----------
		fluxes : np.array of type 'float' and length N_src_max
			array of flux densities

		Class variables
		---------------
		self.pivot_dpl : 'float'. Pivot flux density of the double power law
		self.alpha_1/self.alpha_2 : variables of type 'float'.  Two assumed power law coefficients
		
		Returns
		-------

		logp_dpl : np.array of type 'float'. Log priors for each source

		"""

		logp_dpl = np.zeros_like(fluxes)
		logf = np.log(fluxes)
		piv_mask = (fluxes > self.pivot_dpl)

		logfac2 = (self.alpha_2-self.alpha_1)*np.log(self.pivot_dpl)
		logp_dpl[piv_mask] = logfac2-self.alpha_2*logf[piv_mask]
		logp_dpl[~piv_mask] = -self.alpha_1*logf[~piv_mask]

		return logp_dpl

	def compute_flux_prior(self, f0, pf):
		""" 
		Function to compute the delta log prior of the flux distribution between model states.

		Parameters
		----------

		self : 'Model' class object
			The flux distribution type is obtained from the class object.
		f0 : 'np.array'
			Original flux densities
		pf : 'np.array'
			Proposed flux densities

		Returns
		-------

		factor : 'np.array' of length len(f0). 
			Delta log-prior for each source

		"""
		if self.gdat.flux_prior_type=='single_power_law':
			dlogf = np.log(pf/f0)
			factor = -self.truealpha*dlogf
		
		elif self.gdat.flux_prior_type=='double_power_law':
			log_prior_dpow_pf = np.log(pdfn_dpow(pf,  self.trueminf, self.trueminf*self.newsrc_minmax_range, self.pivot_dpl, self.alpha_1, self.alpha_2))
			log_prior_dpow_f0 = np.log(pdfn_dpow(f0,  self.trueminf, self.trueminf*self.newsrc_minmax_range, self.pivot_dpl, self.alpha_1, self.alpha_2))
			factor = log_prior_dpow_pf - log_prior_dpow_f0

		else:
			print("Need a valid flux prior type (either single_power_law or double_power_law")
			factor = None

		return factor

	def compute_color_prior(self, fluxes):
		all_colors = []
		color_factors = []
		for b in range(self.nbands-1):
			colors = fluxes_to_color(fluxes[0], fluxes[b+1])
			colors[np.isnan(colors)] = self.color_mus[b]
			all_colors.append(colors)
			color_factors.append(-(colors - self.color_mus[b])**2/(2*self.color_sigs[b]**2))

		return np.array(color_factors), all_colors

	def move_stars(self): 

		""" 
		Proposal to perturb the positions/fluxes of model sources. This is done simultaneously by drawing a flux proposal and then 
		a position change that depends on the max flux of the source, i.e. max(current flux vs. proposed flux). 

		Returns
		-------
		proposal : 'Proposal' class object.

		"""

		idx_move = self.idx_parity_stars()
		nw = idx_move.size
		stars0 = self.stars.take(idx_move, axis=1)
		starsp = np.empty_like(stars0)
		
		f0 = stars0[self._F:,:]
		pfs = []

		for b in range(self.nbands):
			if b==0:
				pf = self.flux_proposal(f0[b], nw)
			else:
				pf = self.flux_proposal(f0[b], nw, trueminf=0.00001) #place a minor minf to avoid negative fluxes in non-pivot bands
			pfs.append(pf)
 
		if (np.array(pfs)<0).any():
			print('negative flux!')
			print(np.array(pfs)[np.array(pfs)<0])

		verbprint(self.verbtype, 'Average flux difference : '+str(np.average(np.abs(f0[0]-pfs[0]))), verbthresh=1)

		factor = self.compute_flux_prior(f0[0], pfs[0])

		if np.isnan(factor).any():
			verbprint(self.verbtype,'Factor NaN from flux', verbthresh=1)
			verbprint(self.verbtype,'Number of f0 zero elements:'+str(len(f0[0])-np.count_nonzero(np.array(f0[0]))), verbthresh=1)
			verbprint(self.verbtype, 'prior factor = '+str(factor), verbthresh=1)

			factor[np.isnan(factor)]=0

		""" the loop over bands below computes colors and prior factors in color used when sampling the posterior
		come back to this later  """
		modl_eval_colors = []

		color_factors_orig, colors_orig = self.compute_color_prior(f0)
		color_factors_prop, colors_prop = self.compute_color_prior(pfs)
		color_factors = color_factors_prop - color_factors_orig

		for b in range(self.nbands-1):
			modl_eval_colors.append(colors_prop[b])

		assert np.isnan(color_factors).any()==False       

		verbprint(self.verbtype,'Average absolute color factors : '+str(np.average(np.abs(color_factors))), verbthresh=1)
		verbprint(self.verbtype,'Average absolute flux factors : '+str(np.average(np.abs(factor))), verbthresh=1)

		factor = np.array(factor) + np.sum(color_factors, axis=0)
		
		dpos_rms = np.float32(np.sqrt(self.gdat.N_eff/(2*np.pi))*self.err_f/(np.sqrt(self.nominal_nsrc*self.regions_factor*(2+self.nbands))))/(np.maximum(f0[0],pfs[0]))

		verbprint(self.verbtype,'dpos_rms : '+str(dpos_rms), verbthresh=1)
		
		dpos_rms[dpos_rms < 1e-3] = 1e-3
		dx = np.random.normal(size=nw).astype(np.float32)*dpos_rms
		dy = np.random.normal(size=nw).astype(np.float32)*dpos_rms
		starsp[self._X,:] = stars0[self._X,:] + dx
		starsp[self._Y,:] = stars0[self._Y,:] + dy
		
		verbprint(self.verbtype, 'dx : '+str(dx), verbthresh=1)
		verbprint(self.verbtype, 'dy : '+str(dy), verbthresh=1)
		verbprint(self.verbtype, 'Mean absolute dx and dy : '+str(np.mean(np.abs(dx)))+', '+str(np.mean(np.abs(dy))), verbthresh=1)

		for b in range(self.nbands):
			starsp[self._F+b,:] = pfs[b]
			if (pfs[b]<0).any():
				print('Proposal fluxes less than 0')
				print('band', b)
				print(pfs[b])
		self.bounce_off_edges(starsp)

		proposal = Proposal(self.gdat)
		proposal.add_move_stars(idx_move, stars0, starsp)
		
		assert np.isinf(factor).any()==False
		assert np.isnan(factor).any()==False

		proposal.set_factor(factor)
		return proposal


	def birth_death_stars(self):
		"""		
		Returns
		-------
		proposal : 'Proposal' class object. 

		"""
		lifeordeath = np.random.randint(2)
		nbd = (self.nregx * self.nregy) / 4
		proposal = Proposal(self.gdat)
		# birth
		if lifeordeath and self.n < self.max_nsrc: # need room for at least one source
			nbd = int(min(nbd, self.max_nsrc-self.n)) # add nbd sources, or just as many as will fit
			# mildly violates detailed balance when n close to nstar
			# want number of regions in each direction, divided by two, rounded up
			
			mregx = int(((self.imsz0[0] / self.regsizes[0] + 1) + 1) / 2) # assumes that imsz are multiples of regsize
			mregy = int(((self.imsz0[1] / self.regsizes[0] + 1) + 1) / 2)

			starsb = np.empty((2+self.nbands, nbd), dtype=np.float32)
			starsb[self._X,:] = (np.random.randint(mregx, size=nbd)*2 + self.parity_x + np.random.uniform(size=nbd))*self.regsizes[0] - self.offsetxs[0]
			starsb[self._Y,:] = (np.random.randint(mregy, size=nbd)*2 + self.parity_y + np.random.uniform(size=nbd))*self.regsizes[0] - self.offsetys[0]
			
			for b in range(self.nbands):
				if b==0:
					if self.gdat.flux_prior_type=='single_power_law':
						starsb[self._F+b,:] = self.trueminf * np.exp(np.random.exponential(scale=1./(self.truealpha-1.),size=nbd))
					elif self.gdat.flux_prior_type=='double_power_law':
						starsb[self._F+b,:] = icdf_dpow(np.random.uniform(0, 1, nbd), self.trueminf, self.trueminf*self.newsrc_minmax_range,\
													 self.pivot_dpl, self.alpha_1, self.alpha_2)
				else:
					# draw new source colors from color prior
					new_colors = np.random.normal(loc=self.color_mus[b-1], scale=self.color_sigs[b-1], size=nbd)
					
					starsb[self._F+b,:] = starsb[self._F,:]*10**(0.4*new_colors)
			
					if (starsb[self._F+b,:]<0).any():
						print('negative birth star fluxes')
						print('new_colors')
						print(new_colors)
						print('starsb fluxes:')
						print(starsb[self._F+b,:])

			# some sources might be generated outside image
			inbounds = self.in_bounds(starsb)

			starsb = starsb.compress(inbounds, axis=1)
			
			# checking for what is in mask takes on average 50 us with scatter depending on how many sources there are being proposed
			not_in_mask = np.array([self.dat.weights[0][int(starsb[self._Y,k]), int(starsb[self._X, k])] > 0 for k in range(starsb.shape[1])])


			starsb = starsb.compress(not_in_mask, axis=1)
			factor = np.full(starsb.shape[1], -self.penalty)

			proposal.add_birth_stars(starsb)
			proposal.set_factor(factor)
			
			assert np.isnan(factor).any()==False
			assert np.isinf(factor).any()==False

		# death
		# does region based death obey detailed balance?
		elif not lifeordeath and self.n > 0: # need something to kill
			idx_reg = self.idx_parity_stars()
			nbd = int(min(nbd, idx_reg.size)) # kill nbd sources, or however many sources remain
			if nbd > 0:
				idx_kill = np.random.choice(idx_reg, size=nbd, replace=False)
				starsk = self.stars.take(idx_kill, axis=1)
				factor = np.full(nbd, self.penalty)
				proposal.add_death_stars(idx_kill, starsk)
				proposal.set_factor(factor)
				assert np.isnan(factor).any()==False
		return proposal

	def merge_split_stars(self):

		""" 
		PCAT proposal to merge/split model sources. 	
		
		Returns
		-------
		proposal : 'Proposal' class object. 

		"""

		splitsville = np.random.randint(2)
		idx_reg = self.idx_parity_stars()
		fracs, sum_fs = [], []
		idx_bright = idx_reg.take(np.flatnonzero(self.stars[self._F, :].take(idx_reg) > 2*self.trueminf)) # in region!
		bright_n = idx_bright.size
		nms = int((self.nregx * self.nregy) / 4)
		goodmove = False
		proposal = Proposal(self.gdat)
		# split
		if splitsville and self.n > 0 and self.n < self.max_nsrc and bright_n > 0: # need something to split, but don't exceed nstar
			
			nms = min(nms, bright_n, self.max_nsrc-self.n) # need bright source AND room for split source
			dx = (np.random.normal(size=nms)*self.kickrange).astype(np.float32)
			dy = (np.random.normal(size=nms)*self.kickrange).astype(np.float32)
			idx_move = np.random.choice(idx_bright, size=nms, replace=False)
			stars0 = self.stars.take(idx_move, axis=1)

			fminratio = stars0[self._F,:] / self.trueminf
 
			verbprint(self.verbtype, 'stars0 at splitsville start: '+str(stars0), verbthresh=1)
			verbprint(self.verbtype, 'fminratio here is '+str(fminratio), verbthresh=1)
			verbprint(self.verbtype, 'dx = '+str(dx), verbthresh=1)
			verbprint(self.verbtype, 'dy = '+str(dy), verbthresh=1)
			verbprint(self.verbtype, 'idx_move : '+str(idx_move), verbthresh=1)
				
			fracs.append((1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32))
			
			for b in range(self.nbands-1):
				# changed to split similar fluxes
				d_color = np.random.normal(0,self.gdat.split_col_sig)
				# this frac_sim is what source 1 is multiplied by in its remaining bands, so source 2 is multiplied by (1-frac_sim)

				frac_sim = np.exp(d_color/self.k)*fracs[0]/(1-fracs[0]+np.exp(d_color/self.k)*fracs[0])

				if (frac_sim < 0).any():
					print('negative fraction!!!!')
					goodmove = False

				fracs.append(frac_sim)

			starsp = np.empty_like(stars0)
			starsb = np.empty_like(stars0)

			# starsp is for source 1, starsb is for source 2

			starsp[self._X,:] = stars0[self._X,:] - ((1-fracs[0])*dx)
			starsp[self._Y,:] = stars0[self._Y,:] - ((1-fracs[0])*dy)
			starsb[self._X,:] = stars0[self._X,:] + fracs[0]*dx
			starsb[self._Y,:] = stars0[self._Y,:] + fracs[0]*dy


			for b in range(self.nbands):
				
				starsp[self._F+b,:] = stars0[self._F+b,:]*fracs[b]
				starsb[self._F+b,:] = stars0[self._F+b,:]*(1-fracs[b])

			# don't want to think about how to bounce split-merge
			# don't need to check if above fmin, because of how frac is decided
			inbounds = np.logical_and(self.in_bounds(starsp), self.in_bounds(starsb))
			stars0 = stars0.compress(inbounds, axis=1)
			starsp = starsp.compress(inbounds, axis=1)
			starsb = starsb.compress(inbounds, axis=1)
			idx_move = idx_move.compress(inbounds)
			fminratio = fminratio.compress(inbounds)

			for b in range(self.nbands):
				fracs[b] = fracs[b].compress(inbounds)
				sum_fs.append(stars0[self._F+b,:])
			
			nms = idx_move.size

			goodmove = (nms > 0)*((np.array(fracs) > 0).all())
			if goodmove:
				proposal.add_move_stars(idx_move, stars0, starsp)
				proposal.add_birth_stars(starsb)
				# can this go nested in if statement? 
			invpairs = np.empty(nms)
			
			verbprint(self.verbtype, 'splitsville is happening', verbthresh=1)
			verbprint(self.verbtype, 'goodmove: '+str(goodmove), verbthresh=1)
			verbprint(self.verbtype, 'invpairs: '+str(invpairs), verbthresh=1)
			verbprint(self.verbtype, 'nms: '+str(nms), verbthresh=1)
			verbprint(self.verbtype, 'sum_fs: '+str(sum_fs), verbthresh=1)
			verbprint(self.verbtype, 'fminratio is '+str(fminratio), verbthresh=1)

			for k in range(nms):
				xtemp = self.stars[self._X, 0:self.n].copy()
				ytemp = self.stars[self._Y, 0:self.n].copy()
				xtemp[idx_move[k]] = starsp[self._X, k]
				ytemp[idx_move[k]] = starsp[self._Y, k]
				xtemp = np.concatenate([xtemp, starsb[self._X, k:k+1]])
				ytemp = np.concatenate([ytemp, starsb[self._Y, k:k+1]])
				invpairs[k] =  1./neighbours(xtemp, ytemp, self.kickrange, idx_move[k]) #divide by zero
				invpairs[k] += 1./neighbours(xtemp, ytemp, self.kickrange, self.n)
			invpairs *= 0.5

		# merge
		elif not splitsville and idx_reg.size > 1: # need two things to merge!

			nms = int(min(nms, idx_reg.size/2))
			idx_move = np.empty(nms, dtype=np.int)
			idx_kill = np.empty(nms, dtype=np.int)
			choosable = np.zeros(self.max_nsrc, dtype=np.bool)
			choosable[idx_reg] = True
			nchoosable = float(idx_reg.size)
			invpairs = np.empty(nms)
			
			verbprint(self.verbtype, 'Merging two things!!', verbthresh=1)
			verbprint(self.verbtype, 'nms: '+str(nms), verbthresh=1)
			verbprint(self.verbtype, 'idx_move '+str(idx_move), verbthresh=1)
			verbprint(self.verbtype, 'idx_kill '+str(idx_kill), verbthresh=1)
				
			for k in range(nms):
				idx_move[k] = np.random.choice(self.max_nsrc, p=choosable/nchoosable)
				invpairs[k], idx_kill[k] = neighbours(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], self.kickrange, idx_move[k], generate=True)
				if invpairs[k] > 0:
					invpairs[k] = 1./invpairs[k]
				# prevent sources from being involved in multiple proposals
				if not choosable[idx_kill[k]]:
					idx_kill[k] = -1
				if idx_kill[k] != -1:
					invpairs[k] += 1./neighbours(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], self.kickrange, idx_kill[k])
					choosable[idx_move[k]] = False
					choosable[idx_kill[k]] = False
					nchoosable -= 2
			invpairs *= 0.5

			inbounds = (idx_kill != -1)
			idx_move = idx_move.compress(inbounds)
			idx_kill = idx_kill.compress(inbounds)
			invpairs = invpairs.compress(inbounds)
			nms = idx_move.size
			goodmove = nms > 0

			stars0 = self.stars.take(idx_move, axis=1)
			starsk = self.stars.take(idx_kill, axis=1)
			f0 = stars0[self._F:,:]
			fk = starsk[self._F:,:]

			for b in range(self.nbands):
				sum_fs.append(f0[b,:] + fk[b,:])
				fracs.append(f0[b,:] / sum_fs[b])
			
			fminratio = sum_fs[0] / self.trueminf
			
			verbprint(self.verbtype, 'fminratio: '+str(fminratio)+', nms: '+str(nms), verbthresh=1)
			verbprint(self.verbtype, 'sum_fs[0] is '+str(sum_fs[0]), verbthresh=1)
			verbprint(self.verbtype, 'stars0: '+str(stars0), verbthresh=1)
			verbprint(self.verbtype, 'starsk: '+str(starsk), verbthresh=1)
			verbprint(self.verbtype, 'idx_move '+str(idx_move), verbthresh=1)
			verbprint(self.verbtype, 'idx_kill '+str(idx_kill), verbthresh=1)
				
			starsp = np.empty_like(stars0)
			# place merged source at center of flux of previous two sources
			starsp[self._X,:] = fracs[0]*stars0[self._X,:] + (1-fracs[0])*starsk[self._X,:]
			starsp[self._Y,:] = fracs[0]*stars0[self._Y,:] + (1-fracs[0])*starsk[self._Y,:]
			
			for b in range(self.nbands):
				starsp[self._F+b,:] = f0[b] + fk[b]
			
			if goodmove:
				proposal.add_move_stars(idx_move, stars0, starsp)
				proposal.add_death_stars(idx_kill, starsk)
			
			# turn bright_n into an array
			bright_n = bright_n - (f0[0] > 2*self.trueminf) - (fk[0] > 2*self.trueminf) + (starsp[self._F,:] > 2*self.trueminf)
		
		""" The lines below are where we compute the prior factors that go into P(Catalog), which we use along with P(Data|Catalog) in order to sample from the posterior. 
		The variable "factor" has the log prior (log(P(Catalog))), and since the prior is a product of individual priors we add log factors to get the log prior."""
		if goodmove:
			# the first three terms are the ratio of the flux priors, the next two come from the position terms when choosing sources to merge/split, 
			# the two terms after that capture the transition kernel since there are several combinations of sources that could be implemented, 
			# the last term is the Jacobian determinant f, which is the same for the single and multiband cases given the new proposals 
			
			factor = np.log(2*np.pi*self.kickrange*self.kickrange) - np.log(self.imsz0[0]*self.imsz0[1]) \
					+ np.log(bright_n) + np.log(invpairs)+ np.log(1. - 2./fminratio) + np.log(sum_fs[0])
			
			if self.gdat.flux_prior_type=='single_power_law':
				fluxfac = np.log(self.truealpha-1) + (self.truealpha-1)*np.log(self.trueminf)-self.truealpha*np.log(fracs[0]*(1-fracs[0])*sum_fs[0])

			elif self.gdat.flux_prior_type=='double_power_law':
				log_prior_dpow_split1 = np.log(pdfn_dpow(fracs[0]*sum_fs[0],  self.trueminf,\
											 self.trueminf*self.newsrc_minmax_range,\
											  self.pivot_dpl, self.alpha_1, self.alpha_2))
				log_prior_dpow_split2 = np.log(pdfn_dpow((1-fracs[0])*sum_fs[0],  self.trueminf,\
											 self.trueminf*self.newsrc_minmax_range,\
											  self.pivot_dpl, self.alpha_1, self.alpha_2))
				log_prior_dpow_tot = np.log(pdfn_dpow(sum_fs[0],  self.trueminf,\
											 self.trueminf*self.newsrc_minmax_range,\
											  self.pivot_dpl, self.alpha_1, self.alpha_2))

				fluxfac = log_prior_dpow_split1 + log_prior_dpow_split2 - log_prior_dpow_tot
			else:
				fluxfac = 0.

			factor += fluxfac

			for b in range(self.nbands-1):

				stars0_color = fluxes_to_color(stars0[self._F,:], stars0[self._F+b+1,:])
				starsp_color = fluxes_to_color(starsp[self._F,:], starsp[self._F+b+1,:])
				dc = self.k*(np.log(fracs[b+1]/fracs[0]) - np.log((1-fracs[b+1])/(1-fracs[0])))

				# added_fac comes from the transition kernel of splitting colors in the manner that we do
				added_fac = 0.5*np.log(2*np.pi*self.gdat.split_col_sig**2)+(dc**2/(2*self.gdat.split_col_sig**2))
				factor += added_fac
				
				if splitsville:
				
					starsb_color = fluxes_to_color(starsb[self._F,:], starsb[self._F+b+1,:])
					# colfac is ratio of color prior factors i.e. P(s_0)P(s_1)/P(s_merged), where 0 and 1 are original sources 
					color_fac = (stars0_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (starsp_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (starsb_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2)-0.5*np.log(2*np.pi*self.color_sigs[b]**2)
			 
				else:
					starsk_color = fluxes_to_color(starsk[self._F,:], starsk[self._F+b+1,:])
					# same as above but for merging sources
					color_fac = (starsp_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (stars0_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (starsk_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2)-0.5*np.log(2*np.pi*self.color_sigs[b]**2)

				factor += color_fac

			# this will penalize the model with extra parameters
			factor -= self.penalty

			# if we have a merge, we want to use the reciprocal acceptance factor, in this case the negative of log(factor)
			if not splitsville:
				factor *= -1

			proposal.set_factor(factor)
						
			if np.isnan(factor).any():
				verbprint(self.verbtype, 'There was a NaN factor in merge/split!', verbthresh=1)	

			verbprint(self.verbtype, 'kickrange factor: '+str(np.log(2*np.pi*self.kickrange*self.kickrange)), verbthresh=1)
			verbprint(self.verbtype, 'imsz factor: '+str(np.log(2*np.pi*self.kickrange*self.kickrange)), verbthresh=1)
			verbprint(self.verbtype, 'kickrange factor: '+str(np.log(self.imsz0[0]*self.imsz0[1])), verbthresh=1)
			verbprint(self.verbtype, 'fminratio: '+str(fminratio)+', fmin factor: '+str(np.log(1. - 2./fminratio)), verbthresh=1)
			verbprint(self.verbtype, 'factor after colors: '+str(factor), verbthresh=1)


		return proposal



class Samples():
	""" 
	The Samples() class saves the parameter chains and other diagnostic statistics about the MCMC run. 
	"""

	def __init__(self, gdat):

		""" Here all of the data structures storing sample information are instantiated. """

		nsamp_nloop = (gdat.nsamp, gdat.nloop)
		nsamp_nsrc = (gdat.nsamp, gdat.max_nsrc)
		nsamp_nbands = (gdat.nsamp, gdat.nbands)

		self.nsample = np.zeros(gdat.nsamp, dtype=np.int32) # number of sources
		self.tq_times = np.zeros(gdat.nsamp, dtype=np.float32)

		self.timestats = np.zeros((gdat.nsamp, 6, 8), dtype=np.float32) # contains information on computational performance for different parts of algorithm
		self.accept_stats = np.zeros((gdat.nsamp, 8), dtype=np.float32) # acceptance fractions for different types of proposals

		self.diff2_all, self.accept_all, self.rtypes = [np.zeros(nsamp_nloop, dtype=np.float32) for x in range(3)]# saves log likelihoods of models, accepted proposals, proposal types at each step
		# self.accept_all = np.zeros(nsamp_nloop, dtype=np.float32) # accepted proposals
		# self.rtypes = np.zeros(nsamp_nloop, dtype=np.float32) # proposal types at each step

		self.xsample = np.zeros(nsamp_nsrc, dtype=np.float32) # x positions of sample sources
		self.ysample = np.zeros(nsamp_nsrc, dtype=np.float32) # y positions of sample sources
		self.fsample = [np.zeros(nsamp_nsrc, dtype=np.float32) for x in range(gdat.nbands)]
		
		self.fourier_coeffs, self.fc_rel_amps, self.binned_cib_coeffs, self.template_amplitudes, self.bkg_sample  = [None for x in range(5)]

		if gdat.float_background:
			self.bkg_sample = np.zeros(nsamp_nbands) # thinned mean background levels
		
		if gdat.float_templates:
			print('GDAT.ntemplates is ', gdat.n_templates)
			self.template_amplitudes = np.zeros((gdat.nsamp, gdat.n_templates, gdat.nbands)) # amplitudes of templates used in fit 
		
		if gdat.float_fourier_comps:
			self.fourier_coeffs = np.zeros((gdat.nsamp, gdat.n_fourier_terms, gdat.n_fourier_terms, 4)) # amplitudes of Fourier templates
			self.fc_rel_amps = np.zeros(nsamp_nbands) # relative amplitudes of diffuse Fourier component model across observing bands.
			
		# if gdat.float_cib_templates:
		# 	self.binned_cib_coeffs = np.zeros((gdat.nsamp, gdat.cib_nregion**2)) # amplitudes of Fourier templates

		self.colorsample = [[] for x in range(gdat.nbands-1)]
		self.residuals = [np.zeros((gdat.residual_samples, gdat.imszs[b][0], gdat.imszs[b][1])) for b in range(gdat.nbands)]
		self.model_images = [np.zeros((gdat.residual_samples, gdat.imszs[b][0], gdat.imszs[b][1])) for b in range(gdat.nbands)]

		self.chi2sample = np.zeros(nsamp_nbands, dtype=np.int32)
		self.nbands = gdat.nbands
		self.gdat = gdat

	def add_sample(self, j, model, diff2_list, accepts, rtype_array, accept_fracs, chi2_all, statarrays, resids, model_images):
		""" 
		For each thinned sample, adds model parameters to class variables. 
		This is an in place operation.

		Parameters
		----------

		j : 'int'. Index of thinned sample
		model : Class object 'Model'.
		diff2_list : 'list' of arrays.
		accepts :
		rtype_array : 
		accept_fracs : 
		chi2_all : 
		statarrays :
		resids : 
		model_images :

		"""
		self.nsample[j] = model.n
		self.xsample[j,:] = model.stars[Model._X, :]
		self.ysample[j,:] = model.stars[Model._Y, :]
		self.diff2_all[j,:] = diff2_list
		self.accept_all[j,:] = accepts
		self.rtypes[j,:] = rtype_array
		self.accept_stats[j,:] = accept_fracs
		self.chi2sample[j] = chi2_all
		self.timestats[j,:] = statarrays
		self.bkg_sample[j,:] = model.bkg

		if self.gdat.float_templates:
			self.template_amplitudes[j,:,:] = model.template_amplitudes 
		if self.gdat.float_fourier_comps:
			self.fourier_coeffs[j,:,:,:] = model.fourier_coeffs 
			self.fc_rel_amps[j,:] = model.fc_rel_amps
		# if self.gdat.float_cib_templates:
		# 	self.binned_cib_coeffs[j] = model.binned_cib_coeffs

		for b in range(self.nbands):
			self.fsample[b][j,:] = model.stars[Model._F+b,:]
			if self.gdat.nsamp - j < self.gdat.residual_samples+1:
				self.residuals[b][-(self.gdat.nsamp-j),:,:] = resids[b] 
				self.model_images[b][-(self.gdat.nsamp-j),:,:] = model_images[b]

	def save_samples(self, result_path, timestr):

		""" 
		Save chain parameters/metadata with numpy compressed file. 
		This is an in place operation.
		
		TODO rewrite this to be general
		
		Parameters
		----------
		result_path : 'str'. Path to result directory.
		timestr : 'str'. Timestring used to save the run (maybe make it possible to customize name?)

		"""
		# fourier comp, fourier comp colors

		if self.nbands < 3:
			residuals2, model_images2 = None, None
		else:
			residuals2, model_images2 = self.residuals[2], self.model_images[2]
		if self.nbands < 2:
			residuals1, model_images1 = None, None
		else:
			residuals1, model_images1 = self.residuals[1], self.model_images[1]

		residuals0, model_images0 = self.residuals[0], self.model_images[0]

		# for b in range(self.nbands):
		# 	np.savez(result_path+'/'+str(timestr)+'/model_image_samples_band'+str(b)+'.npz', model_images=model_images[b], obs_map=obs_maps[b])
		
		# np.savez(result_path + '/' + str(timestr) + '/proposal_stats.npz', rtypes=self.rtypes, times=self.timestats, accepts=self.accept_all, accept=self.accept_stats)

		# np.savez(result_path + '/' + str(timestr) + '/chain.npz', n=self.nsample, x=self.xsample, y=self.ysample, f=self.fsample, \
		# 	chi2=self.chi2sample, diff2s=self.diff2_all, bkg=self.bkg_sample, template_amplitudes=self.template_amplitudes, \
		# 	fourier_coeffs=self.fourier_coeffs, fc_rel_amps=self.fc_rel_amps, binned_cib_coeffs=self.binned_cib_coeffs)
		
		np.savez(result_path + '/' + str(timestr) + '/chain.npz', n=self.nsample, x=self.xsample, y=self.ysample, f=self.fsample, \
			chi2=self.chi2sample, times=self.timestats, accept=self.accept_stats, diff2s=self.diff2_all, rtypes=self.rtypes, \
			accepts=self.accept_all, residuals0=residuals0, residuals1=residuals1, residuals2=residuals2, model_images0=model_images0,\
			model_images1=model_images1, model_images2=model_images2, bkg=self.bkg_sample, template_amplitudes=self.template_amplitudes, \
			fourier_coeffs=self.fourier_coeffs, fc_rel_amps=self.fc_rel_amps, binned_cib_coeffs=self.binned_cib_coeffs)



class lion():

	""" 
	This is where the main lion() class is initialized and is the starting point for all PCAT runs.
	Below, the collection of configurable parameters in PCAT are presented, separated into relevant variable groups. 
	While there are many tunable variables in the implementation, in practice most of these can remain fixed. 
	As a note, there may be a better way of structuring this, or in storing variable initializations in dedicated parameter files.

	"""

	gdat = gdatstrt()

	def __init__(self, **kwargs):
			
		for attr, valu in locals().items():
			if '__' not in attr and attr != 'gdat' and attr != 'map_object':
				setattr(self.gdat, attr, valu)

		#if specified, use seed for random initialization
		if self.gdat.init_seed is not None:
			np.random.seed(self.gdat.init_seed)

		self.gdat.bands = [b for b in np.array([self.gdat.band0, self.gdat.band1, self.gdat.band2]) if b is not None]
		self.gdat.nbands = len(self.gdat.bands)

		if self.gdat.psf_fwhms is None:
			self.gdat.psf_fwhms = [self.gdat.psf_pixel_fwhm for i in range(self.gdat.nbands)]

		self.gdat.N_eff = 4*np.pi*(self.gdat.psf_fwhms[0]/2.355)**2 # variable psf pixel fwhm, use pivot band fwhm

		# point src delay controls when all point source proposals begin
		if self.gdat.point_src_delay is not None:
			for movestr in ['movestar', 'birth_death', 'merge_split']:
				self.gdat.moveweight_dict[movestr] = self.gdat.point_src_delay

			# self.gdat.movestar_sample_delay = self.gdat.point_src_delay
			# self.gdat.birth_death_sample_delay = self.gdat.point_src_delay
			# self.gdat.merge_split_sample_delay = self.gdat.point_src_delay

		self.gdat.band_dict = config.band_dict # for accessing different wavelength filenames
		self.gdat.lam_dict = config.lam_dict
		self.gdat.pixsize_dict = config.pixsize_dict

		if self.gdat.color_sigs is None:
			self.gdat.color_sigs = config.color_sigs
		if self.gdat.color_mus is None:
			self.gdat.color_mus = config.color_mus

		self.gdat.timestr = time.strftime("%Y%m%d-%H%M%S")

		# unless specified, sets order of Fourier model in linear marginalization to that of full model.
		if self.gdat.MP_order is None:
			self.gdat.MP_order = int(self.gdat.n_fourier_terms)
		
		# power law exponents equal to 1 in double power law will cause a numerical error.
		if self.gdat.alpha_1 == 1.0 and self.flux_prior_type=='double_power_law':
			self.gdat.alpha_1 += 0.01
		if self.gdat.alpha_2 == 1.0 and self.flux_prior_type=='double_power_law':
			self.gdat.alpha_2 += 0.01
		
		if self.gdat.template_names is None:
			self.gdat.n_templates=0 
		else:
			self.gdat.n_templates=len(self.gdat.template_names)

		if self.gdat.mean_offsets is None:
			self.gdat.mean_offsets = np.zeros_like(np.array(self.gdat.bands))

		if type(self.gdat.bkg_sig_fac)==float: # if single number, make bkg_sig_fac an array length nbands where each band has same factor
			sigfacs = [self.gdat.bkg_sig_fac for b in range(self.gdat.nbands)]
			self.gdat.bkg_sig_fac = np.array(sigfacs).copy()

		if self.gdat.temp_prop_sig_fudge_facs is None:
			self.gdat.temp_prop_sig_fudge_facs = [1. for b in range(self.gdat.nbands)]

		template_band_idxs = config.template_band_idxs
		# template_band_idxs = dict({'sze':[0, 1, 2], 'sze':[0,1,2], 'lensing':[0, 1, 2], 'dust':[0, 1, 2], 'planck':[0,1,2], 'cib':[0, 1, 2]})

		# fourier comp colors
		fourier_band_idxs = [0, 1, 2]

		self.gdat.template_order = []
		self.gdat.template_band_idxs = np.zeros(shape=(self.gdat.n_templates, self.gdat.nbands))

		if self.gdat.template_names is not None:
			for i, temp_name in enumerate(self.gdat.template_names):		
				for b, band in enumerate(self.gdat.bands):
					if band in template_band_idxs[temp_name]:
						self.gdat.template_band_idxs[i,b] = band
					else:
						self.gdat.template_band_idxs[i,b] = None
				self.gdat.template_order.append(temp_name)

		self.gdat.regions_factor = 1./float(self.gdat.nregion**2)
		self.data = pcat_data(self.gdat, self.gdat.auto_resize, self.gdat.nregion)
		self.data.load_in_data(show_input_maps=self.gdat.show_input_maps)

		# initialize CIB templates if used
		# if self.gdat.float_cib_templates:
		# 	print('Initializing binned CIB templates..')
		# 	dimxs_resize = [self.gdat.imszs[b][0] for b in range(self.gdat.nbands)]
		# 	dimys_resize = [self.gdat.imszs[b][1] for b in range(self.gdat.nbands)]
		# 	dimxs = [self.gdat.imszs_orig[b][0] for b in range(self.gdat.nbands)]
		# 	dimys = [self.gdat.imszs_orig[b][1] for b in range(self.gdat.nbands)]
		# 	print('dimxs resize is ', dimxs_resize)
		# 	print('while original dimensions are ', dimxs)
		# 	self.gdat.coarse_cib_templates = generate_subregion_cib_templates(dimxs, dimys, self.gdat.cib_nregion, dimxs_resize=dimxs_resize, dimys_resize=dimys_resize)
		# 	if self.gdat.show_input_maps:
		# 		for b in range(self.gdat.nbands):
		# 			plot_single_map(self.gdat.coarse_cib_templates[b][0], title='b = '+str(b))

		# in case of severe crowding, can scale parsimony prior using F statistic 
		if self.gdat.F_statistic_alph:
			alph = compute_Fstat_alph(self.gdat.imszs, self.gdat.nbands, self.gdat.nominal_nsrc)
			self.gdat.alph = alph
			print('Regularization prior (per degree of freedom) computed from the F-statistic with '+str(self.gdat.nominal_nsrc)+' sources is '+str(np.round(alph, 3)))

		if self.gdat.float_fourier_comps:
			print('float_fourier_comps set to True, initializing Fourier component model..')
			# if there are previous fourier components, use those
			if self.gdat.init_fourier_coeffs is not None:
				if self.gdat.n_fourier_terms != self.gdat.init_fourier_coeffs.shape[0]:
					self.gdat.n_fourier_terms = self.gdat.init_fourier_coeffs.shape[0]
			else:
				self.gdat.init_fourier_coeffs = np.zeros((self.gdat.n_fourier_terms, self.gdat.n_fourier_terms, 4))

			self.gdat.fc_templates = multiband_fourier_templates(self.gdat.imszs, self.gdat.n_fourier_terms, psf_fwhms=self.gdat.psf_fwhms, x_max_pivot_list=self.gdat.x_max_pivot_list, scale_fac=None)
			self.gdat.fourier_band_idxs = [None for b in range(self.gdat.nbands)]

			# if no fourier comp amplitudes specified set them all to unity
			if self.gdat.fc_rel_amps is None:
				self.gdat.fc_rel_amps = np.ones(shape=(self.gdat.nbands,))

			for b, band in enumerate(self.gdat.bands):
				if band in fourier_band_idxs:
					self.gdat.fourier_band_idxs[b] = band
				else:
					self.gdat.fourier_band_idxs[b] = None

		if self.gdat.bkg_level is None:
			self.gdat.bkg_level = np.zeros((self.gdat.nbands,))
			for b, band in enumerate(self.gdat.bands):
				median_val = np.median(self.data.data_array[b])
				self.gdat.bkg_level[b] = median_val # background will initially be biased high
				# self.gdat.bkg_level[b] = median_val - 0.003 # subtract by 3 mJy/beam since background level is biased high by sources # specific to SPIRE obs
			
		print('Initial background levels set to ', self.gdat.bkg_level)

		if self.gdat.save_outputs:
			#create directory for results, save config file from run
			frame_dir, newdir, timestr = create_directories(self.gdat)
			self.gdat.timestr = timestr
			self.gdat.frame_dir = frame_dir
			self.gdat.newdir = newdir
			save_params(newdir, self.gdat)

	def initialize_print_log(self):
		if self.gdat.print_log:
			self.gdat.flog = open(self.gdat.result_path+'/'+self.gdat.timestr+'/print_log.txt','w')
		else:
			self.gdat.flog = None		


	def main(self):
		""" 
		Here is where we initialize the C libraries and instantiate the arrays that will store our 
		thinned samples and other stats. We want the MKL routine if possible, then OpenBLAS, then regular C, with that order in priority.
		This is also where the MCMC sampler is initialized and run.

		Returns
		-------

		"""

		self.initialize_print_log()
		
		libmmult = initialize_libmmult(cblas = self.gdat.cblas, openblas = self.gdat.openblas)

		initialize_c(self.gdat, libmmult, cblas=self.gdat.cblas)

		start_time = time.time()
		verbprint(self.gdat.verbtype, 'Initializing Samples class..', verbthresh=1)

		samps = Samples(self.gdat)
		verbprint(self.gdat.verbtype, 'Initializing Model class..', verbthresh=1)

		model = Model(self.gdat, self.data, libmmult)

		verbprint(self.gdat.verbtype, 'Done initializing model..', verbthresh=1)

		if self.gdat.n_marg_updates is None:
			self.gdat.n_marg_updates = 0

		fc_marg_counter = 0
		for j in range(self.gdat.nsamp): # run sampler for gdat.nsamp thinned states
			print('Sample', j, file=self.gdat.flog)

			if self.gdat.bkg_moore_penrose_inv and fc_marg_counter < self.gdat.n_marg_updates and j%self.gdat.fc_marg_period==0 and j > 0:

				print('j = ', j, 'while on update ', fc_marg_counter, 'of ', self.gdat.n_marg_updates, 'updates')
				fc_model = generate_template(model.fourier_coeffs, self.gdat.n_fourier_terms, imsz=self.gdat.imszs[0], fourier_templates=model.fourier_templates[0])

				_, _, _, _, mp_coeffs, temp_A_hat, nanmask = compute_marginalized_templates(self.gdat.MP_order, resids[0]+fc_model, self.data.uncertainty_maps[0],\
										  ridge_fac=self.gdat.ridge_fac, ridge_fac_alpha=self.gdat.ridge_fac_alpha, show=False, \
										  ravel_temps = model.ravel_temps, bt_siginv_b_inv=model.bt_siginv_b_inv, bt_siginv_b=model.bt_siginv_b)

				model.fourier_coeffs[:self.gdat.MP_order, :self.gdat.MP_order, :] = mp_coeffs.copy()

				fc_marg_counter += 1

			
			# once ready to sample, recompute proposal weights
			model.update_moveweights(j)

			_, chi2_all, statarrays,  accept_fracs, diff2_list, rtype_array, accepts, resids, model_images = model.run_sampler(j)
			samps.add_sample(j, model, diff2_list, accepts, rtype_array, accept_fracs, chi2_all, statarrays, resids, model_images)


		if self.gdat.save_outputs:
			print('Saving...', file=self.gdat.flog)

			# save catalog ensemble and other diagnostics
			samps.save_samples(self.gdat.result_path, self.gdat.timestr)

			# save final catalog state
			np.savez(self.gdat.result_path + '/'+str(self.gdat.timestr)+'/final_state.npz', cat=model.stars, bkg=model.bkg, templates=model.template_amplitudes, fourier_coeffs=model.fourier_coeffs)

		if self.gdat.timestr_list_file is not None:
			if os.path.exists(self.gdat.timestr_list_file):
				timestr_list = list(np.load(self.gdat.timestr_list_file)['timestr_list'])
				timestr_list.append(self.gdat.timestr)
			else:
				timestr_list = [self.gdat.timestr]
			np.savez(self.gdat.timestr_list_file, timestr_list=timestr_list)

		# if self.gdat.generate_condensed_catalog:
		# 	xmatch_roc = cross_match_roc(timestr=self.gdat.timestr, nsamp=self.gdat.n_condensed_samp)
		# 	xmatch_roc.load_gdat_params(gdat=self.gdat)
		# 	condensed_cat, seed_cat = xmatch_roc.condense_catalogs(prevalence_cut=self.gdat.prevalence_cut, save_cats=True, make_seed_bool=True,\
		# 															 mask_hwhm=self.gdat.mask_hwhm, search_radius=self.gdat.search_radius, matching_dist=self.gdat.matching_dist)

		if self.gdat.make_post_plots:
			result_plots(gdat = self.gdat, generate_condensed_cat=self.gdat.generate_condensed_catalog, n_condensed_samp=self.gdat.n_condensed_samp, prevalence_cut=self.gdat.prevalence_cut, mask_hwhm=self.gdat.mask_hwhm, condensed_catalog_plots=self.gdat.generate_condensed_catalog)

		dt_total = time.time() - start_time
		print('Full Run Time (s):', np.round(dt_total,3), file=self.gdat.flog)
		print('Time String:', str(self.gdat.timestr), file=self.gdat.flog)

		with open(self.gdat.newdir+'/time_elapsed.txt', 'w') as filet:
			filet.write('time elapsed: '+str(np.round(dt_total,3))+'\n')

		plt.close() # I think this is for result plots
			
		if self.gdat.print_log:
			self.gdat.flog.close()



