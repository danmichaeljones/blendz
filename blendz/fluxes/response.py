import numpy as np
from scipy.interpolate import interp1d
from blendz.config import _config
from templates import Templates
from filters import Filters

class Responses(object):
    def __init__(self, templates=None, filters=None, ref_band=_config.ref_mag, zGrid=_config.redshift_grid):
        if templates is None:
            #Load default templates
            self.templates = Templates()
        else:
            #From kwarg
            self.templates = templates
        if filters is None:
            #Load default filters
            self.filters = Filters()
        else:
            #From kwarg
            self.filters = filters
        self.ref_band = ref_band
        self.zGrid = zGrid

        self._calculate_responses()

    def _calculate_responses(self):
        self._all_responses = {}
        self._interpolators = {}
        for T in xrange(self.templates.num_templates):
            self._all_responses[T] = {}
            self._interpolators[T] = {}
            for F in xrange(self.filters.num_filters):
                self._all_responses[T][F] = np.zeros(len(self.zGrid))
                for iZ, Z in enumerate(self.zGrid):
                    shiftedTemplate = self.templates.interp(T, self.filters.wavelength(F) / (1+Z) )
                    # TODO:The multiply by lambda in here is a conversion
                    # from flux_nu (in the equation) to flux_lambda (how templates are defined)
                    # Add a setting to choose how the templates are defined.
                    integrand = shiftedTemplate * self.filters.response(F) * \
                                self.filters.wavelength(F) / self.filters.norm(F)
                    self._all_responses[T][F][iZ] = np.trapz(integrand, x=self.filters.wavelength(F))
                    #Commented out colours below now as repsonses should be fluxes, colours done in model
                    #Define interpolators as flux here
                    self._interpolators[T][F] = interp1d(self.zGrid, self._all_responses[T][F],\
                                                         bounds_error=False, fill_value=0.)

    def __call__(self, T, F, Z):
        #Using isinstance for redshift to catch python float and np.float64
        #Single template/filter/redshift case
        if type(T)==int and type(F)==int and isinstance(Z, float):
            return float(self._interpolators[T][F](Z))
        #Single template/filter, multiple redshifts case
        elif type(T)==int and type(F)==int and type(Z)==np.ndarray:
            return np.array([self._interpolators[T][F](zz) for zz in Z])
        #Single template/redshift case, all filters
        elif type(T)==int and (F is None) and isinstance(Z, float):
            return np.array([self._interpolators[T][ff](Z) for ff in xrange(self.filters.num_filters)])
        #Single template, multiple redshift, all filters case
        elif type(T)==int and (F is None) and type(Z)==np.ndarray:
            return np.array([[self._interpolators[T][ff](zz) \
                              for ff in xrange(self.filters.num_filters)]\
                            for zz in Z])
        else:
            raise TypeError('Incompatible types for arguments. Require T = int, \
                            F = int or None, Z = float or numpy.ndarray. Instead,\
                            got T={}, F={}, Z={}'.format(type(T), type(F), type(Z)))
