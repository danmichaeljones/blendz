from builtins import *
import warnings
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from blendz import Configuration
from blendz.fluxes import Templates
from blendz.fluxes import Filters

class Responses(object):
    def __init__(self, templates=None, filters=None, config=None, **kwargs):
        #Warn user is config and either/or templates given that config ignored
        if ((templates is not None and config is not None) or
                (filters is not None and config is not None)):
            warnings.warn('A configuration object was provided to Responses '
                          + 'as well as a Template/Filter object, though these '
                          + 'should be mutually exclusive. The configuration '
                          + 'provided will be ignored.')
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

    def _calculate_interpolators(self):
        self._interpolators = {}
        for T in range(self.templates.num_templates):
            self._interpolators[T] = {}
            for F in range(self.filters.num_filters):
                self._interpolators[T][F] = interp1d(self.zGrid, self._all_responses[T, F, :],\
                                                     bounds_error=False, fill_value=0.)

    def _calculate_responses(self):
        self._all_responses = np.zeros((self.templates.num_templates, self.filters.num_filters, len(self.zGrid)))
        tot_its = self.templates.num_templates * self.config.z_len * \
                    self.filters.num_filters
        with tqdm(total=tot_its) as pbar:
            for F in range(self.filters.num_filters):
                for iZ, Z in enumerate(self.zGrid):
                    for T in range(self.templates.num_templates):
                        shiftedTemplate = self.templates.interp(T, self.filters.wavelength(F) / (1+Z) )
                        flux_norm = self.filters.norm(F) * 2.99792458e18
                        integrand = shiftedTemplate * self.filters.response(F) * \
                                    self.filters.wavelength(F) / flux_norm
                        self._all_responses[T, F, iZ] = np.trapz(integrand, x=self.filters.wavelength(F))
                        pbar.update()
        self.interp = interp1d(self.zGrid, self._all_responses, bounds_error=False, fill_value=0.)
