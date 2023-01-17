import os
import numpy as np
import numpy.ctypeslib as npct
import ctypes
from ctypes import c_int, c_double
import pickle
import time


def add_directory(dirpath):
	""" 
	Checks if directory path already exists, if not creates new directory. 
	Parameters
	----------
	dirpath : 'str'

	Returns the same dirpath.
	"""
	if not os.path.isdir(dirpath):
		os.makedirs(dirpath)
	return dirpath


def create_directories(gdat, run_dir_name=None):
	""" 
	Makes initial directory structure for new PCAT run.

	Parameters
	----------

	gdat : global data object

	Returns
	-------

	frame_dir_name : `str'
		New PCAT frame directory name

	new_dir_name : `str'
		New PCAT parent directory name

	timestr : `str'
		Time string associated with new PCAT run

	"""
	if run_dir_name is None:
		run_dir_name = gdat.timestr 

	run_dir = gdat.result_basedir+run_dir_name

	timestr = gdat.timestr
	if os.path.isdir(run_dir):
		i = 0
		time.sleep(np.random.uniform(0, 5))
		# while os.path.isdir(gdat.result_basedir+gdat.timestr+'_'+str(i)):
		while os.path.isdir(run_dir+'_'+str(i)):
			time.sleep(np.random.uniform(0, 2))
			i += 1
		
		timestr += '_'+str(i)
		run_dir += '_'+str(i)

	os.makedirs(run_dir)

	frame_dir = run_dir+'/frames/'
	
	if not os.path.isdir(frame_dir) and gdat.n_frames > 0:
		os.makedirs(frame_dir)
	
	print('timestr:', timestr)
	return frame_dir, run_dir, timestr


def initialize_c(gdat, libmmult, cblas=False):

	""" 

	This function initializes the C library needed for the core numerical routines in PCAT. 
	This is an in place operation.
	
	Parameters
	----------
	
	gdat : global data object usued by PCAT
	libmmult : Matrix multiplication library
	cblas (optional) : If True, use CBLAS matrix multiplication routines in model evaluation

	"""

	verbprint(gdat.verbtype, 'initializing c routines and data structs', file=gdat.flog, verbthresh=1)
	# if gdat.verbtype > 1:
	# 	print('initializing c routines and data structs', file=gdat.flog)

	array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
	array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
	array_2d_double = npct.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
	array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")

	if cblas:
		if os.path.getmtime('pcat-lion.c') > os.path.getmtime('pcat-lion.so'):
			warnings.warn('pcat-lion.c modified after compiled pcat-lion.so', Warning)		
				
		libmmult.pcat_model_eval.restype = None
		libmmult.pcat_model_eval.argtypes = [c_int, c_int, c_int, c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_1d_int, array_1d_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]
		libmmult.pcat_imag_acpt.restype = None
		libmmult.pcat_imag_acpt.argtypes = [c_int, c_int, array_2d_float, array_2d_float, array_2d_int, c_int, c_int, c_int, c_int]
		libmmult.pcat_like_eval.restype = None
		libmmult.pcat_like_eval.argtypes = [c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]

	else:
		if os.path.getmtime('blas.c') > os.path.getmtime('blas.so'):
			warnings.warn('blas.c modified after compiled blas.so', Warning)		
		
		libmmult.clib_eval_modl.restype = None
		libmmult.clib_eval_modl.argtypes = [c_int, c_int, c_int, c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_1d_int, array_1d_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]
		libmmult.clib_updt_modl.restype = None
		libmmult.clib_updt_modl.argtypes = [c_int, c_int, array_2d_float, array_2d_float, array_2d_int, c_int, c_int, c_int, c_int]
		libmmult.clib_eval_llik.restype = None
		libmmult.clib_eval_llik.argtypes = [c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]

# make this standalone

def initialize_libmmult(cblas=True, openblas=False):
	""" Initializes matrix multiplication used in PCAT."""
	if cblas:
		print('Using CBLAS routines for Intel processors.. :-) ')

		if sys.version_info[0] == 2:
			libmmult = npct.load_library('pcat-lion', '.')
		else:
			libmmult = npct.load_library('pcat-lion', '.')

	elif openblas:
		print('Using OpenBLAS routines... :-/ ')

		if sys.version_info[0] == 2:
			libmmult = npct.load_library('blas-open', '.')
		else:
			libmmult = npct.load_library('blas-open.so', '.')

	else:
		print('Using BLAS routines..')
		libmmult = ctypes.cdll['./blas.so'] # not sure how stable this is, trying to find a good Python 3 fix to deal with path configuration

	return libmmult


def verbprint(verbose, text, file=None, verbthresh=0):
	""" 
	This function is a wrapped print function that accommodates various levels of verbosity. 
	This is an in place operation.

	Parameters
	----------

	verbose : 'int'. Level of verbosity. If verbthresh is None, verbose=1 will result in a statement being printed, otherwise verbose needs to be greater than verbthresh.
	text : 'str'. Text to print. 
	file (optional) : 'str'. User can specifiy file to write logs to. (I'm not sure if this fully works).
			Default is 'None'.
	verbthresh (optional) : Verbosity threshold. Default is 'None' (meaning the verbosity threshold is >0).

	"""
	if verbthresh is not None:
		if verbose > verbthresh:
			print(text, file=file)
	else:
		if verbose:
			print(text, file=file)

