import numpy as np
import matplotlib 
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from image_eval import psf_poly_fit, image_model_eval
from scipy.ndimage import gaussian_filter


def compute_marginalized_templates(n_terms, data, error, imsz=None, \
                             psf_fwhm=3., ridge_fac = None, ridge_fac_alpha=None,\
                            show=True, x_max_pivot=None, verbose=False, \
                            bt_siginv_b=None, bt_siginv_b_inv=None, ravel_temps=None, fourier_templates=None, \
                                   return_temp_A_hat=False, compute_nanmask=True):
    
    """
    NOTE -- this only works for single band at the moment. Is there a way to compute the Moore-Penrose inverse for 
    backgrounds observed over several bands with a fixed color prior? 

    also , I think that using the full noise model in the matrix product is necessary when using multiband and multi-region
    evaluations. This might already be handled in the code by zeroing out NaNs.
    
    Ridge factor is inversely proportional to fluctuation power in each mode. 


    Parameters
    ----------

    n_terms : `int`.
    data : 
    error : 
    imsz (optional) : `tuple` of length 2. 
        Default is None.
    bt_siginv_b, bt_siginv_b_inv : `~np.array~` of type `float`
        Default is None.
    ravel_temps : 
    fourier_templates : 
    psf_fwhm : `float`.
        Default is 3. 
    ridge_fac : `float`. 
        Default is None. 
    ridge_fac_alpha (optional) : `float`. 
        Default is None. 
    show : `bool`. 
        Default is True. 
    x_max_pivot : 
        Default is None. 
    verbose : `bool`. 
        Default is False. 

    Returns
    -------

    fourier_templates : Collection of Fourier component templates. 
    ravel_temps : raveled templates 
    bt_siginv_b : matrix product B^T Sigma^-1 B 
    bt_siginv_b_inv : 
    mp_coeffs : Moore-Penrose coefficients 
    temp_A_hat : Best fit linear combination of templates 
    
    """
    
    if imsz is None:
        imsz = error.shape
        
    if verbose:
        print('n_terms is ', n_terms)
        
    if ravel_temps is None:
        if fourier_templates is None:
            fourier_templates, meshx_idx, meshy_idx = make_fourier_templates(imsz[0], imsz[1], n_terms, psf_fwhm=psf_fwhm, x_max_pivot=x_max_pivot, return_idxs=True)
        else:
            meshx_idx, meshy_idx = np.meshgrid(np.arange(n_terms), np.arange(n_terms))

        ravel_temps = ravel_temps_from_ndtemp(fourier_templates, n_terms)
        kx_idx_rav = meshx_idx.ravel()
        ky_idx_rav = meshy_idx.ravel()
    
    im_cut_rav = data.copy().ravel()
    err_cut_rav = error.copy().ravel()

    # print('')
    
    if compute_nanmask:
        # compress system of equations excluding any nan-valued entries
        nanmask = np.logical_or(np.logical_or((err_cut_rav ==0), np.isinf(err_cut_rav)), np.logical_or(np.isnan(err_cut_rav), np.isnan(im_cut_rav)))
        
        # print('ravel temps has shape', ravel_temps.shape)
        if ravel_temps.shape[1]==np.sum(~nanmask):
            print('dont need to compress rravel temps it already is')
        else:
            ravel_temps = ravel_temps.compress(~nanmask, axis=1)
        im_cut_rav = im_cut_rav.compress(~nanmask)
        err_cut_rav = err_cut_rav.compress(~nanmask)

        if verbose:
            print('nan values in ', np.sum(nanmask), ' of ', len(nanmask))
    else:
        nanmask = None

    if bt_siginv_b_inv is None:
        print('computing bt siginv b inv')
        bt_siginv_b = np.dot(ravel_temps, np.dot(np.diag(err_cut_rav**(-2)), ravel_temps.transpose()))
            
        assert ~np.isnan(np.linalg.cond(bt_siginv_b))

        if verbose:
            print('condition number of (B^T S^{-1} B)^{-1}: ', np.linalg.cond(bt_siginv_b))
        
        if ridge_fac is not None:
            if verbose:
                print('adding regularization')

            if ridge_fac_alpha is not None:
                kmag = np.sqrt((kx_idx_rav+1)**2 + (ky_idx_rav+1)**2)/np.sqrt(2.) # sqrt of 2 is for kx=1 & ky=1, since its relative to fundamental mode
                ridge_fac /= kmag**(-ridge_fac_alpha)
                
            lambda_I = np.zeros_like(bt_siginv_b)
            np.fill_diagonal(lambda_I, ridge_fac)
            bt_siginv_b += lambda_I
            
        bt_siginv_b_inv = np.linalg.inv(bt_siginv_b)
    
    siginv_K_rav = im_cut_rav*err_cut_rav**(-2) # Sigma^-1 Y
    bt_siginv_K = np.dot(ravel_temps, siginv_K_rav) # B^T Sigma^-1 Y
    A_hat = np.dot(bt_siginv_b_inv, bt_siginv_K) # (B^T Sigma^-1 B^-1 + Lambda I) B^T Sigma^-1 Y
    mp_coeffs = np.reshape(A_hat, (n_terms, n_terms, 4))
    
    temp_A_hat = None
    if return_temp_A_hat:
        temp_A_hat = generate_template(mp_coeffs, n_terms, fourier_templates=fourier_templates, N=imsz[0], M=imsz[1], x_max_pivot=x_max_pivot)

    if show:
        plot_mp_fit(temp_A_hat, n_terms, A_hat, data)
        
    return fourier_templates, ravel_temps, bt_siginv_b, bt_siginv_b_inv, mp_coeffs, temp_A_hat, nanmask


def plot_mp_fit(temp_A_hat, n_terms, A_hat, data):
    plt.figure(figsize=(10,10))
    plt.suptitle('Moore-Penrose inverse, $N_{FC}$='+str(n_terms), fontsize=20)
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
    plt.imshow(data-temp_A_hat, origin='lower', cmap='Greys', vmax=np.percentile(temp_A_hat, 95), vmin=np.percentile(temp_A_hat, 5))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def ravel_temps_from_ndtemp(templates, n_terms, auxdim=4):
    """
    Ravels Fourier templates for computing cross-template covariance used in Moore-Penrose inverse. 

    Parameters
    ----------
    templates : array
        Fourier component templates
    n_terms : int
        Order of Fourier component model
    auxdim : int, optional
        Default is 4.
    """
    ravel_templates_all = np.reshape(templates, (templates.shape[0], templates.shape[1], auxdim, templates.shape[-2]*templates.shape[-1]))
    
    ravel_temps = ravel_templates_all[:n_terms, :n_terms]
    
    ravel_temps = np.reshape(ravel_temps, (ravel_temps.shape[0]*ravel_temps.shape[1]*ravel_temps.shape[2], ravel_temps.shape[-1]))
    
    return ravel_temps

def multiband_fourier_templates(imszs, n_terms, show_templates=False, psf_fwhms=None, x_max_pivot_list=None, scale_fac=None):
    """
    Given a list of image and beam sizes, produces multiband fourier templates for background modeling.

    Parameters
    ----------

    imszs : list of lists
        List containing image dimensions for each of the three observations

    n_terms : int
        Order of Fourier expansion for templates. the number of templates (currently) scales as 2*n_terms^2

    show_templates : bool, optional
        if True, plots the array of templates. Default is False.

    psf_fwhms : list, optional
        List of beam sizes across observations. If left unspecified, all PSFs assumed to have 3 pixel FWHM. 
        Default is 'None'.
    
    Returns
    -------

    all_templates : list of `numpy.ndarray's
        The set of Fourier templates for each observation.

    """

    all_templates = []

    for b in range(len(imszs)):
        if psf_fwhms is None:
            psf_fwhm = None
        else:
            psf_fwhm = psf_fwhms[b]

        x_max_pivot = None
        if x_max_pivot_list is not None:
            x_max_pivot = x_max_pivot_list[b]

        all_templates.append(make_fourier_templates(imszs[b][0], imszs[b][1], n_terms, show_templates=show_templates, psf_fwhm=psf_fwhm, x_max_pivot=x_max_pivot, scale_fac=scale_fac))
    return all_templates

def make_fourier_templates(N, M, n_terms, show_templates=False, psf_fwhm=None, shift=False, x_max_pivot=None, scale_fac=None, return_idxs=False):
        
    """
    
    Given image dimensions and order of the series expansion, generates a set of 2D fourier templates.

    Parameters
    ----------

    N : int
        length of image
    M : int
        width of image
    n_terms : int
        Order of Fourier expansion for templates. the number of templates (currently) scales as 2*n_terms^2
    
    show_templates : bool, optional
        if True, plots the array of templates. Default is False.
    
    psf_fwhm : float, optional
        Observation PSF full width at half maximum (FWHM). This can be used to pre-convolve templates for background modeling 
        Default is 'None'.

    x_max_pivot : float, optional
        Indicating pixel coordinate for boundary of FOV in each dimension. Default is 'None'.

    return_idxs : bool, optional
        If True, returns mesh grids of Fourier component indices for x and y. 
        Default is False.

    Returns
    -------
    
    templates : `numpy.ndarray' of shape (n_terms, n_terms, 4, N, M)
        Contains 2D Fourier templates for truncated series

    """

    templates = np.zeros((n_terms, n_terms, 4, N, M))
    if scale_fac is None:
        scale_fac = 1.

    x = np.arange(N)
    y = np.arange(M)

    meshkx, meshky = np.meshgrid(np.arange(n_terms), np.arange(n_terms))
    
    meshx, meshy = np.meshgrid(x, y)
        
    xtemps_cos, ytemps_cos, xtemps_sin, ytemps_sin = [np.zeros((n_terms, N, M)) for x in range(4)]

    N_denom = N
    M_denom = M

    if x_max_pivot is not None:
        N_denom = x_max_pivot
        M_denom = x_max_pivot

    for n in range(n_terms):

        # modified series
        if shift:
            xtemps_sin[n] = np.sin((n+1-0.5)*np.pi*meshx/N_denom)
            ytemps_sin[n] = np.sin((n+1-0.5)*np.pi*meshy/M_denom)
        else:
            xtemps_sin[n] = np.sin((n+1)*np.pi*meshx/N_denom)
            ytemps_sin[n] = np.sin((n+1)*np.pi*meshy/M_denom)
        
        xtemps_cos[n] = np.cos((n+1)*np.pi*meshx/N_denom)
        ytemps_cos[n] = np.cos((n+1)*np.pi*meshy/M_denom)
    
    for i in range(n_terms):
        for j in range(n_terms):

            if psf_fwhm is not None: # if beam size given, convolve with PSF assumed to be Gaussian
                templates[i,j,0,:,:] = gaussian_filter(xtemps_sin[i]*ytemps_sin[j], sigma=psf_fwhm/2.355)
                templates[i,j,1,:,:] = gaussian_filter(xtemps_sin[i]*ytemps_cos[j], sigma=psf_fwhm/2.355)
                templates[i,j,2,:,:] = gaussian_filter(xtemps_cos[i]*ytemps_sin[j], sigma=psf_fwhm/2.355)
                templates[i,j,3,:,:] = gaussian_filter(xtemps_cos[i]*ytemps_cos[j], sigma=psf_fwhm/2.355)
            else:
                templates[i,j,0,:,:] = xtemps_sin[i]*ytemps_sin[j]
                templates[i,j,1,:,:] = xtemps_sin[i]*ytemps_cos[j]
                templates[i,j,2,:,:] = xtemps_cos[i]*ytemps_sin[j]
                templates[i,j,3,:,:] = xtemps_cos[i]*ytemps_cos[j]
     
    templates *= scale_fac

    if show_templates:
        for k in range(4):
            counter = 1
            plt.figure(figsize=(8,8))
            for i in range(n_terms):
                for j in range(n_terms):           
                    plt.subplot(n_terms, n_terms, counter)
                    plt.title('i = '+ str(i)+', j = '+str(j))
                    plt.imshow(templates[i,j,k,:,:])
                    counter +=1
            plt.tight_layout()
            plt.show()

    if return_idxs:
        return templates, meshkx, meshky

    return templates


def generate_template(fourier_coeffs, n_terms, fourier_templates=None, imsz=None, N=None, M=None, psf_fwhm=None, x_max_pivot=None):

    """
    Given a set of coefficients and Fourier templates, computes their dot product.

    Parameters
    ----------

    fourier_coeffs : `~numpy.ndarray' of shape (n_terms, n_terms, 2)
        Coefficients of truncated Fourier expansion.

    n_terms : `int`
        Order of Fourier expansion to compute sum over. This is left explicit as an input
        in case one wants the flexibility of calling it for different numbers of terms, even
        if the underlying truncated series has more terms.

    fourier_templates : `~numpy.ndarray' of shape (n_terms, n_terms, 2, N, M), optional
        Contains 2D Fourier templates for truncated series. If left unspecified, a set of Fourier templates is generated
        on the fly. Default is 'None'.

    imsz : `list` of length 2. Dimension of image. Default is 'None'.
    N : `int`, optional
        length of image. Default is 'None'.
    M : `int`, optional
        width of image. Default is 'None.'

    psf_fwhm : `float`, optional
        Observation PSF full width at half maximum (FWHM). This can be used to pre-convolve templates for background modeling 
        Default is 'None'.

    x_max_pivot : `float`, optional
        Because of different image resolution across bands and the use of multiple region proposals, the non pivot band images may cover a larger 
        field of view than the pivot band image. When modeling structured emission across several bands, it is important that the Fourier components
        model a consistent field of view. Extra pixels in the non-pivot bands do not contribute to the log-likelihood, so I think the solution is to 
        compute the Fourier templates where the period is based on the WCS transformations across bands, which can translate coordinates bounding
        the pivot image to coordinates in the non-pivot band images.

        Default is 'None'. 

    Returns
    -------

    sum_temp : `~numpy.ndarray' of shape (N, M)
        The summed template.

    """

    if imsz is None and N is None and M is None:
        if fourier_templates is not None:
            imsz = [fourier_templates.shape[-2], fourier_templates.shape[-1]]
        else:
            print('need to provide input dimensions through either imsz or (N, M)')
            return None

    if imsz is None:
        imsz = [N,M]
        
    if fourier_templates is None:
        fourier_templates = make_fourier_templates(imsz[0], imsz[1], n_terms, psf_fwhm=psf_fwhm, x_max_pivot=x_max_pivot)

    sum_temp = np.sum([fourier_coeffs[i,j,k]*fourier_templates[i,j,k] for i in range(n_terms) for j in range(n_terms) for k in range(fourier_coeffs.shape[-1])], axis=0)
    
    return sum_temp





