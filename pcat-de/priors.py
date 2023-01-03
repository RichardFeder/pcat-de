import numpy as np

def compute_Fstat_alph(imszs, nbands, nominal_nsrc):
	"""
	Computes expected improvement in log-likelihood per degree of freedom (DOF) in the finite DOF limit through the F-statistic (https://en.wikipedia.org/wiki/F-test).

	Parameters
	----------

	imszs : 'np.array' of 'floats'.
	nbands : 'int'. Number of bands in fit.
	nominal_nsrc : 'int'. Number of expected sources in the fit. This could be adapted at a later point in the chain, but is currently fixed ab initio.

	Returns
	-------

	alph : 'float'. Expected improvement in log-likelihood per degree of freedom.

	"""

	npix = np.sum(np.array([imszs[b][0]*imszs[b][1] for b in range(nbands)]))
	alph = 0.5*(2.+nbands)*npix/(npix - (2.+nbands)*nominal_nsrc)
	alph /= 0.5*(2.+nbands) # regularization prior is normalized relative to limit with infinite data, per degree of freedom

	return alph


def icdf_dpow(unit, minm, maxm, brek, sloplowr, slopuppr):
    
    """
    Inverse CDF for double power law, taken from https://github.com/tdaylan/pcat/blob/master/pcat/main.py

	Parameters
	----------
	
	unit : 'np.array' of type 'float'. Uniform draws from CDF
	minm/maxm : 'floats'. Minimum/maximum bounds on flux density distribution. These parameters are used to normalize the distribution so that prior samples can be drawn.
	brek : 'float'. Pivot value for the flux distribution
	sloplowr/slopuppr : 'floats'. Power law parameters for lower/upper end of the CDF

	Returns
	-------

	para : 'np.array' of type 'float'. Sample of flux densities corresponding to CDF draws (unit).

    """
    
    if np.isscalar(unit):
        unit = np.array([unit])
    
    faca = 1. / (brek**(sloplowr - slopuppr) * (brek**(1. - sloplowr) - minm**(1. - sloplowr)) \
                                / (1. - sloplowr) + (maxm**(1. - slopuppr) - brek**(1. - slopuppr)) / (1. - slopuppr))
    facb = faca * brek**(sloplowr - slopuppr) / (1. - sloplowr)

    para = np.empty_like(unit)
    cdfnbrek = facb * (brek**(1. - sloplowr) - minm**(1. - sloplowr))
    indxlowr = np.where(unit <= cdfnbrek)[0]
    indxuppr = np.where(unit > cdfnbrek)[0]
    if indxlowr.size > 0:
        para[indxlowr] = (unit[indxlowr] / facb + minm**(1. - sloplowr))**(1. / (1. - sloplowr))
    if indxuppr.size > 0:
        para[indxuppr] = ((1. - slopuppr) * (unit[indxuppr] - cdfnbrek) / faca + brek**(1. - slopuppr))**(1. / (1. - slopuppr))
    
    return para

def pdfn_dpow(xdat, minm, maxm, brek, sloplowr, slopuppr):
    
    """
    PDF for double power law, also taken from https://github.com/tdaylan/pcat/blob/master/pcat/main.py

	Parameters
	----------

	xdat : 'np.array' of type 'float'. Values that prior is evaluated for.
	minm/maxm : 'floats'. Minimum/maximum bounds on double power law distribution. These parameters are used to normalize the distribution so that prior samples can be drawn.
	brek : 'float'. Pivot value for the flux distribution
	sloplowr/slopuppr : 'floats'. Power law parameters for lower/upper end of the CDF

	Returns
	-------

	pdfn : 'np.array' of type 'float'. Array of priors for each value

    """

    if np.isscalar(xdat):
        xdat = np.array([xdat])
    
    faca = 1. / (brek**(sloplowr - slopuppr) * (brek**(1. - sloplowr) - minm**(1. - sloplowr)) / \
                                            (1. - sloplowr) + (maxm**(1. - slopuppr) - brek**(1. - slopuppr)) / (1. - slopuppr))
    facb = faca * brek**(sloplowr - slopuppr) / (1. - sloplowr)
    
    pdfn = np.empty_like(xdat)
    indxlowr = np.where(xdat <= brek)[0]
    indxuppr = np.where(xdat > brek)[0]
    if indxlowr.size > 0:
        pdfn[indxlowr] = faca * brek**(sloplowr - slopuppr) * xdat[indxlowr]**(-sloplowr)
    if indxuppr.size > 0:
        pdfn[indxuppr] = faca * xdat[indxuppr]**(-slopuppr)
    
    return pdfn