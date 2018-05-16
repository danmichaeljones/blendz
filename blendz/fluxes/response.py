from builtins import *
import warnings
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from blendz import Configuration
from blendz.fluxes import Templates
from blendz.fluxes import Filters

class Responses(object):
    def __init__(self, templates=None, filters=None, config=None, **kwargs):
        #Warn user is config and either/or templates given that config ignored
        if ((templates is not None and config is not None) or
                (filters is not None and config is not None)):
            warnings.warn("""A configuration object was provided to Responses
                            as well as a Template/Filter object, though these
                            should be mutually exclusive. The configuration
                            provided will be ignored.""")
        #Both templates and filters given, merge with default+kwargs
        if (templates is not None) and (filters is not None):
            self.config = Configuration(**kwargs)
            self.config.mergeFromOther(templates.config)
            self.config.mergeFromOther(filters.config)
            self.templates = templates
            self.filters = filters
        #Templates given but filters not, load filters using default+kwargs+templates config
        elif (templates is not None) and (filters is None):
            self.config = Configuration(**kwargs)
            self.config.mergeFromOther(templates.config)
            self.templates = templates
            self.filters = Filters(config=self.config)
        #Filters given but templates not, load templates using default+kwargs+filters config
        elif (templates is None) and (filters is not None):
            self.config = Configuration(**kwargs)
            self.config.mergeFromOther(filters.config)
            self.filters = filters
            self.templates = Templates(config=self.config)
        #Neither given, load both from provided (or default, if None) config
        else:
            self.config = Configuration(**kwargs)
            if config is not None:
                self.config.mergeFromOther(config)

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
        for T in range(self.templates.num_templates):
            self._interpolators[T] = {}
            for F in range(self.filters.num_filters):
                self._interpolators[T][F] = interp1d(self.zGrid, self._all_responses[T, F, :],\
                                                     bounds_error=False, fill_value=0.)

    def _calculate_responses(self):
        #self._all_responses = {}
        self._all_responses = np.zeros((self.templates.num_templates, self.filters.num_filters, len(self.zGrid)))
        tot_its = self.templates.num_templates * self.config.z_len * \
                    self.filters.num_filters
        with tqdm(total=tot_its) as pbar:
            for F in range(self.filters.num_filters):
                for iZ, Z in enumerate(self.zGrid):
                    extinction = self.etau_madau(self.filters.wavelength(F), Z)
                    for T in range(self.templates.num_templates):
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
                        pbar.update()
        self.interp = interp1d(self.zGrid, self._all_responses, bounds_error=False, fill_value=0.)

    def plotFiltersAndTemplates(self, single_plot=True):
        if single_plot:
            plt.figure(figsize=(15, 6*self.templates.num_templates))
        for T in range(self.templates.num_templates):
            if single_plot:
                plt.subplot(self.templates.num_templates, 1, T+1)
            else:
                plt.figure(figsize=(15, 6))
            lamt = self.templates.wavelength(T)
            tmp = self.templates.flux(T)
            tmp_z_mid = self.templates.interp(T, lamt/(1.+(self.config.z_hi*0.5)))
            tmp_z_hi = self.templates.interp(T, lamt/(1.+self.config.z_hi))
            plt.plot(lamt, tmp/np.max(tmp), label=self.templates.name(T) + ' at z=0', color='b')
            plt.plot(lamt, tmp_z_mid/np.max(tmp_z_mid), linestyle='--',
                     label=self.templates.name(T) + ' at z={}'.format((self.config.z_hi*0.5)), color='b')
            plt.plot(lamt, tmp_z_hi/np.max(tmp_z_hi), linestyle=':',
                     label=self.templates.name(T) + ' at z={}'.format(self.config.z_hi), color='b')
            filter_upper_bound = np.zeros(self.filters.num_filters)
            for F in range(self.filters.num_filters):
                lamf = self.filters.wavelength(F)
                flt = self.filters.response(F)
                filter_upper_bound[F] = lamf[np.max(np.where( ~np.isclose(flt, 0.))[0])]
                plt.plot(lamf, flt/np.max(flt), color='0.5')
            plt.legend()
            plt.xlim(0, np.max(filter_upper_bound)*1.2)
        if single_plot:
            plt.subplots_adjust(hspace=0.1)

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
            return np.array([self._interpolators[T][ff](Z) for ff in range(self.filters.num_filters)])
        #Single template, multiple redshift, all filters case
        elif isinstance(T, (int, np.integer)) and (F is None) and type(Z)==np.ndarray:
            return np.array([[self._interpolators[T][ff](zz) \
                              for ff in range(self.filters.num_filters)]\
                              for zz in Z])
        else:
            raise TypeError('Incompatible types for arguments. Require T = int, \
                            F = int or None, Z = float or numpy.ndarray. Instead,\
                            got T={}, F={}, Z={}'.format(type(T), type(F), type(Z)))
