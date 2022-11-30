import numpy as np
import matplotlib 
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def generate_subregion_cib_templates(dimx, dimy, nregion, cib_rel_amps = [1.0, 1.41, 1.17], \
                                    verbose=False, dimxs_resize=None, dimys_resize=None):
    """ 

    This function generates a template set designed to model the unresolved CIB. For maps divisible by nregion, each template is a 2D tophat, and for non-divisible maps
    the border pixels are weighted by the overlap of each sub-region over the pixel solid angle. 

    Note that this is not used within the initial version of PCAT-DE.

	Parameters
	----------
	
	dimx : 'int', or 'list' of ints. This and dimy specify image dimensions for potentially several maps
	dimy : 'int', or 'list' of ints.
	nregion : 'int'. Number of subregions along each image axis.
	cib_rel_amps : 'list' of floats. 
	verbose : 
	dimxs_resize : 
	dimys_resize : 

	Returns
	-------

	coarse_template_list : 'list' of `np.array' of type 'float', shape (nregion**2, dimx, dimy) for each band.


    Notes
    -----

    I also added dimxs_resize/dimys_resize for PCAT specifically, where the sub-region likelihood evaluation is dealt with by padding maps with non-divisible counts.
    """
    if type(dimx)==int or type(dimx)==float:
        dimx = [dimx]
        dimy = [dimy]
    
    if dimxs_resize is not None:
        if type(dimxs_resize)==int:
            dimxs_resize = [dimxs_resize]
            dimys_resize = [dimys_resize]
    # make sure you can divide properly. not sure how this should be for several bands.. 
    assert dimx[0]%nregion==0
    assert dimy[0]%nregion==0
    
    nbands = len(dimx)
    
    if cib_rel_amps is None:
        cib_rel_amps = np.array([1. for x in range(nbands)])

    subwidths = [dimx[n]//nregion for n in range(nbands)]
    
    # I should make these overlapping for non-integer positions, weighted by decimal contribution to given pixel
    subwidths_exact = [float(dimx[n])/nregion for n in range(nbands)]
    ntemp = nregion**2

    if verbose:
        print('subwidths exact:', subwidths_exact)
        print('ntemp is ', ntemp)
        print('subwidths are ', subwidths)
    
    coarse_template_list = []
    
    for n in range(nbands):
        is_divisible = (subwidths[n]==subwidths_exact[n])
        
        if verbose:
            print(subwidths[n], subwidths_exact[n])
            print('is divisible is ', is_divisible)
        
        init_x = 0.
        running_x = 0.
        
        templates = np.zeros((ntemp, dimx[n], dimy[n]))
        
        if dimxs_resize is not None:
            templates_resize = np.zeros((ntemp, dimxs_resize[n], dimys_resize[n]))
        
        for i in range(nregion):
            init_y = 0.
            running_y = 0.
            
            running_x += subwidths_exact[n]
            x_remainder = running_x - np.floor(running_x)
            init_x_remainder = init_x - np.floor(init_x)
            
            for j in range(nregion):
                                
                running_y += subwidths_exact[n]
                y_remainder = running_y - np.floor(running_y)
                init_y_remainder = init_y - np.floor(init_y)

                if verbose:
                    print(running_x, running_y, int(np.floor(init_x)), int(np.floor(running_x)), int(np.ceil(init_y)), int(np.floor(running_y)))
                
                templates[i*nregion + j, int(np.ceil(init_x)):int(np.floor(running_x)), int(np.ceil(init_y)):int(np.floor(running_y))] = 1.0
                                
                if not is_divisible:
                
                    if np.ceil(running_x) > np.floor(running_x): # right edge

                        templates[i*nregion + j, int(np.floor(running_x)),  int(np.ceil(init_y)):int(np.floor(running_y))] = x_remainder
                        if np.floor(running_x) < dimx[n] and np.floor(running_y) < dimy[n]: # top right corner
                            templates[i*nregion + j, int(np.floor(running_x)), int(np.floor(running_y))] = (y_remainder+x_remainder)/4.

                    if np.ceil(running_y) > np.floor(running_y): # top edge

                        templates[i*nregion + j, int(np.ceil(init_x)):int(np.floor(running_x)), int(np.floor(running_y))] = y_remainder
                        if init_x > 0 and np.floor(running_y) < dimy[n]: # top left corner
                            templates[i*nregion + j, int(np.floor(init_x)), int(np.floor(running_y))] = (y_remainder+np.ceil(init_x)-init_x)/4.

                    if init_x > np.floor(init_x): # left edge

                        templates[i*nregion + j, int(np.floor(init_x)),  int(np.ceil(init_y)):int(np.floor(running_y))] = np.ceil(init_x)-init_x
                        if init_x > 0 and init_y > 0: # bottom left corner
                            templates[i*nregion + j, int(np.floor(init_x)), int(np.floor(init_y))] = (np.ceil(init_x)-init_x+np.ceil(init_y)-init_y)/4.

                    if init_y > np.floor(init_y): # bottom edge

                        templates[i*nregion + j, int(np.ceil(init_x)):int(np.floor(running_x)), int(np.floor(init_y))] = np.ceil(init_y)-init_y
                        if init_y > 0 and np.floor(running_x) < dimx[n]: # bottom right corner
                            templates[i*nregion + j, int(np.floor(running_x)), int(np.floor(init_y))] = (x_remainder+np.ceil(init_y)-init_y)/4.


                init_y = running_y
                
            init_x = running_x

        if dimxs_resize is not None:
            templates_resize[:,:dimx[n], :dimy[n]] = templates.copy()
            coarse_template_list.append(cib_rel_amps[n]*templates_resize)

        else:
            coarse_template_list.append(cib_rel_amps[n]*templates)
        
    return coarse_template_list


def psf_smooth_templates(templates, psf_sigmas=[1.27, 1.27, 1.27]):
    """ 
    Smooth templates on scale of PSF (assumed to have sigma of 1.27 pix in each band. 
    
    Parameters
    ----------
    templates : Set of image templates to smooth.
    psf_sigmas : 
        Default is [1.27, 1.27, 1.27]

    Returns
    -------
    smoothed_ts : Smoothed templates
    
    """
	smoothed_ts = []
	for i, template in enumerate(templates):
		smoothed_ts.append(gaussian_filter(template, sigma=psf_sigmas[i]))
	return smoothed_ts



