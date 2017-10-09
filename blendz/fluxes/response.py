import warnings
import numpy as np
from scipy.interpolate import interp1d
from blendz.config import _config
from blendz.fluxes import Templates
from blendz.fluxes import Filters

class Responses(object):
    def __init__(self, templates=None, filters=None, config=None):
        #Warn user is config and either/or templates given that config ignored
        if ((templates is not None and config is not None) or
                (filters is not None and config is not None)):
            warnings.warn("""A configuration object was provided to Responses
                            as well as a Template/Filter object, though these
                            should be mutually exclusive. The configuration
                            provided will be ignored.""")
        #Both templates and filters given, just check if configs are equal
        if (templates is not None) and (filters is not None):
            if templates.config == filters.config:
                self.config = filters.config
                self.templates = templates
                self.filters = filters
            else:
                raise ValueError('Configuration of templates and filters must be the same.')
        #Templates given but filters not, load filters using templates config
        elif (templates is not None) and (filters is None):
            self.config = templates.config
            self.templates = templates
            self.filters = Filters(config=self.config)
        #Filters given but templates not, load templates using filters config
        elif (templates is None) and (filters is not None):
            self.config = filters.config
            self.filters = filters
            self.templates = Templates(config=self.config)
        #Neither given, load both from provided (or default, if None) config
        else:
            if config is None:
                self.config = _config
            else:
                self.config = config
            self.templates = Templates(config=self.config)
            self.filters = Filters(config=self.config)

        self.zGrid = self.config.redshift_grid
        self._calculate_responses()
        self._calculate_interpolators()

    def etau_madau(self, wl,z):
        """
        Madau 1995 extinction for a galaxy spectrum at redshift z
        defined on a wavelenght grid wl

        This function is just a *slightly* tidied version of function in bpz_tools.py
        """
        n = len(wl)
        l = np.array([1216.,1026.,973.,950.])
        xe = 1.+z

        #If all the spectrum is redder than (1+z)*wl_lyman_alfa
        if wl[0]> l[0]*xe:
            return np.ones(n)#np.zeros(n)+1.

        #Madau coefficients
        c = np.array([3.6e-3,1.7e-3,1.2e-3,9.3e-4])
        ll = 912.
        tau = np.zeros(n)#wl*0.
        i1 = np.searchsorted(wl,ll)
        i2 = n-1
        #Lyman series absorption
        for i in range(len(l)):
            i2 = np.searchsorted(wl[i1:i2],l[i]*xe)
            tau[i1:i2] = tau[i1:i2]+c[i]*(wl[i1:i2]/l[i])**3.46

        if ll*xe < wl[0]:
            return np.exp(-tau)

        #Photoelectric absorption
        xe = 1.+z
        i2 = np.searchsorted(wl,ll*xe)
        xc = wl[i1:i2]/ll
        xc3 = xc**3
        tau[i1:i2] = tau[i1:i2]+\
                    (0.25*xc3*(xe**.46-xc**0.46)\
                     +9.4*xc**1.5*(xe**0.18-xc**0.18)\
                     -0.7*xc3*(xc**(-1.32)-xe**(-1.32))\
                     -0.023*(xe**1.68-xc**1.68))

        tau = np.clip(tau, 0, 700)
        return np.exp(-tau)

    def _calculate_interpolators(self):
        self._interpolators = {}
        for T in xrange(self.templates.num_templates):
            self._interpolators[T] = {}
            for F in xrange(self.filters.num_filters):
                self._interpolators[T][F] = interp1d(self.zGrid, self._all_responses[T, F, :],\
                                                     bounds_error=False, fill_value=0.)

    def _calculate_responses(self):
        #self._all_responses = {}
        self._all_responses = np.zeros((self.templates.num_templates, self.filters.num_filters, len(self.zGrid)))
        for F in xrange(self.filters.num_filters):
            for iZ, Z in enumerate(self.zGrid):
                extinction = self.etau_madau(self.filters.wavelength(F), Z)
                for T in xrange(self.templates.num_templates):
                    shiftedTemplate = self.templates.interp(T, self.filters.wavelength(F) / (1+Z) )
                    # TODO:The multiply by lambda in here is a conversion
                    # from flux_nu (in the equation) to flux_lambda (how templates are defined)
                    # Add a setting to choose how the templates are defined.
                    # Normalisation includes factor of c
                    flux_norm = self.filters.norm(F) * 2.99792458e18
                    integrand = shiftedTemplate * self.filters.response(F) * \
                                self.filters.wavelength(F) / flux_norm
                    #integrand_extinct = integrand * self.etau_madau(self.filters.wavelength(F), Z)
                    integrand_extinct = integrand * extinction
                    self._all_responses[T, F, iZ] = np.trapz(integrand_extinct, x=self.filters.wavelength(F))
        self.interp = interp1d(self.zGrid, self._all_responses, bounds_error=False, fill_value=0.)

    def __call__(self, T, F, Z):
        #Using isinstance to catch suitable python and numpy data-types
        #Single template/filter/redshift case
        if isinstance(T, (int, np.integer)) and type(F)==int and isinstance(Z, (float, np.floating)):
            return float(self._interpolators[T][F](Z))
        #Single template/filter, multiple redshifts case
        elif isinstance(T, (int, np.integer)) and type(F)==int and type(Z)==np.ndarray:
            return np.array([self._interpolators[T][F](zz) for zz in Z])
        #Single template/redshift case, all filters
        elif isinstance(T, (int, np.integer)) and (F is None) and isinstance(Z, (float, np.floating)):
            return np.array([self._interpolators[T][ff](Z) for ff in xrange(self.filters.num_filters)])
        #Single template, multiple redshift, all filters case
        elif isinstance(T, (int, np.integer)) and (F is None) and type(Z)==np.ndarray:
            return np.array([[self._interpolators[T][ff](zz) \
                              for ff in xrange(self.filters.num_filters)]\
                              for zz in Z])
        else:
            raise TypeError('Incompatible types for arguments. Require T = int, \
                            F = int or None, Z = float or numpy.ndarray. Instead,\
                            got T={}, F={}, Z={}'.format(type(T), type(F), type(Z)))
