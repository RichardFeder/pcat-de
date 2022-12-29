.. PCAT-DE documentation master file, created by
   sphinx-quickstart on Wed Aug  3 17:02:51 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PCAT-DE's documentation!
===================================
This readthedocs page describes the software **PCAT-DE**, or **P**\robabilistic **CAT**\aloging in the presence of **D**\iffuse **E**\mission. PCAT-DE was first used in `Butler & Feder et al. (2021) <https://arxiv.org/abs/2110.13932>`_, while PCAT-DE is tested in more detail in Feder et al. (2022).

The advantage of PCAT-DE is its flexibility: any spatial-spectral template may be used in principle and fit alongside a point source population, where the number of sources is unknown. This allows one to probe the **transdimensional covariance** between a given signal and the union of point source models with varying Nsrc, otherwise referred to as a **metamodel**.

The use applications for PCAT-DE so far include:

- Detection and measurement of point sources in the presence of diffuse galactic cirrus
- Measurement of spatially extended Sunyaev-Zel'dovich effect in the presence of cosmic infrared background (CIB) galaxies and diffuse galactic cirrus

Installation
-------------

To install PCAT-DE you can clone the latest version on `Github <https://github.com/RichardFeder/pcat-de>`_.


Existing work on probabilistic cataloging
-----------------------------------------

This work builds on a long list of existing implementations and extensions based on the framework of probabilistic cataloging. All of these methods build on **transdimensional inference**, in which the number of model components is inferred simultaneously with the properties of the components. This list is almost certaintly non-exhaustive:

- Detection of multiple exoplanet radial velocity signals in the presence of stellar activity `Brewer & Donovan (2015) <https://ui.adsabs.harvard.edu/abs/2015MNRAS.448.3206B/abstract>`_
- Point source photometry in crowded stellar fields  `Brewer et al. (2015) <https://iopscience.iop.org/article/10.1088/0004-6256/146/1/7>`_, 
  `Portillo et al. (2017) <https://iopscience.iop.org/article/10.3847/1538-3881/aa8565/pdf>`_, `Feder et al. (2020) <https://iopscience.iop.org/article/10.3847/1538-3881/ab74cf/meta>`_)

- Inference of dark matter sub-halo mass functions in strong gravitationally lensed systems (`Daylan et al. (2017b) <https://iopscience.iop.org/article/10.3847/1538-4357/aaaa1e/pdf>`_)

- Constraints on number counts of sub-threshold point source populations in Fermi-LAT data (`Daylan et al. (2017a) <https://iopscience.iop.org/article/10.3847/1538-4357/aa679e/meta>`_)
- Measurement of spatially extended Sunyaev-Zel'dovich effect in the presence of cosmic infrared background (CIB) galaxies and diffuse galactic cirrus (`Butler & Feder (2021) <https://arxiv.org/abs/2110.13932>`_).


Implementation details
----------------------

The code is structured as follows. First, the ``pcat_main()`` class is instantiated, using a combination of user-provided parameters and parameters stored in configuration files. Each time PCAT is run, a parameter file is saved in pickled and readable forms (params.npz, params_read.txt) within the results folder.

There are many hyperparameters that can be tuned, which are specified in ``config.py``, however in practice many of these do not need to be modified. They can be broken down into various groups:

Data configuration parameters
+++++++++++++++++++++++++++++

This includes the location of the files (can be input through data_path or combination of ``im_fpath`` and ``err_fpath``) and details of the noise model implementation. These should be the names of FITS files (with ``.fits`` extensions). The data can be fed in as either a single observed map (e.g., ``image_extnames=['SIGNAL']``), or as a sum of several maps, (e.g., ``image_extnames=[{signal_noiseless}, {noise}]``), where ``{signal_noiseless}`` and ``{noise}`` should be customized to the saved FITS image cards. Additional Gaussian noise can be added by setting  ``add_noise`` to True and either specifying a constant noise level (``scalar_noise_sigma``) or using the uncertainty map (``add_noise=True`` and ``use_uncertainty_map=True``). The details of the PSF can be specified in terms of a beam full width at half maximum (FWHM, ``psf_fwhm``) provided in pixel units (this assumes a Gaussian beam), or as a generic PSF postage stamp. When an empirical PSF estimate is available it can be fed into PCAT using the ``psf_postage_stamp`` keyword. If one wants to run PCAT on a masked version of the image, the most straightforward way to do this is to set all pixels in the uncertainty map to zero/inf/NaN. PCAT will have predicted model values for these pixels, however they are zero-weighted in the likelihood evaluation so this can be thought of as a type of inpainting.


Map pre-processing parameters
+++++++++++++++++++++++++++++


PCAT sampler parameters/model hyperparameters
+++++++++++++++++++++++++++++++++++++++++++++

The number of samples is set by ``nsamp`` -- by default the chains are thinned by a factor of ``nloop=1000``, so a run with ``nsamp=4000`` is really :math:`4 \times 10^6` model evaluations. For computational efficiency, it is recommended to set a ``max_nsrc`` for the model. The maximum should be sufficiently far from the bulk of the posterior on Nsrc. Oftentimes if the number of sources is diverging it means something is not correct in the data parsing or the astrometric calibration.
- Hyperparameters describing model for constant background/mean normalization of maps ("Background Parameters"), any fixed spatial templates ("Template Parameters") and the Fourier component templates ("Fourier Component Parameters").
- Run time diagnostics/posterior plot details.
- Optional parameters for computing condensed catalog from posterior samples ("Condensed catalog").

These parameters can be directly modified in the configuration file, or passed as keyword arguments to the lion class instantiation. Model proposals are called many times within the PCAT chains. These are included in the ``Proposal()`` class and are drawn from according to the model components and proposal weights ("moveweights").

Data parsing
++++++++++++

One important (and error prone) step in running PCAT is the proper parsing of maps and other data products to PCAT. Because PCAT builds a generative model for the observed data, it typically needs:

- The observed maps
- A model for the point spread function (PSF) of the telescope optics and the pixel function
- A noise model image for each map
- If running on several maps (e.g., multiband data), a consistent astrometric reference frame across images (along with consistent trimming of maps)


In PCAT-DE, two sets of diagnostics are included to ensure the data products are parsed correctly. To validate the astrometry, PCAT has a test module ``validate_astrometry()`` which projects a grid of points across each of the images. The second shows the data as they are parsed in and is used when ``show_input_maps`` is set to True. To plot these for an individual run in real time you want to set ``matplotlib.colors('tkAGG')``, otherwise they will be saved as files in the results folder (specified by ``config.result_path``). 

Examples
--------

Example scripts can be found in the repository under ``example1.py``. Some code implementing artificial star tests can be found in the script ``artificial_star_test.py``

Posteriors and Diagnostics
++++++++++++++++++++++++++

Verifying the proper convergence of PCAT can be done by inspecting the posteriors and other diagnostics derived from posterior samples.

- The chi squared of the samples and the reduced chi squared statistics.
- Pixel-wise residual maps
- Number of sources. Does the posterior on Nsrc reside well within the range of :math:`[N_{min}, N_{max}]`?
- If running on several maps (e.g., multiband data), a consistent astrometric reference frame across images (along with consistent trimming of maps)
- Acceptance fractions for different proposals. If these are too low, it may suggest the model has not converged. If they are too high, it may suggest the proposal kernels are too narrow, such the delta log posterior between models is close to zero.



