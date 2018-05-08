from builtins import *
from os.path import join
import warnings
import numpy as np
from scipy.interpolate import interp1d
from blendz import Configuration

class Templates(object):
    def __init__(self, config=None, **kwargs):
        self.config = Configuration(**kwargs)
        if config is not None:
            self.config.mergeFromOther(config)

        self.template_dict = self.config.template_dict
        self.num_templates = len(self.template_dict)
        self.possible_types = sorted(set(tmp['type'] for tmp in self.template_dict.values()))

        self.loadTemplates()
        self._num_type = self._countTypes()
        self._interpolators = self._getInterpolators()

        self.tmp_ind_to_type_ind = []
        for T in range(self.num_templates):
            tmpType = self.templateType(T)
            self.tmp_ind_to_type_ind.append(np.where(np.array(self.possible_types)==tmpType)[0][0])


    def loadTemplates(self):
        self._all_templates = {}
        for T in range(self.num_templates):
            self._all_templates[T] = {}
            self._all_templates[T]['lambda'], self._all_templates[T]['flux'] = \
                        np.loadtxt(self.template_dict[T]['path'], unpack=True)
            tmp_order = np.argsort(self._all_templates[T]['lambda'])
            self._all_templates[T]['lambda'] = self._all_templates[T]['lambda'][tmp_order]
            self._all_templates[T]['flux'] = self._all_templates[T]['flux'][tmp_order]
            self._all_templates[T]['name'] = self.template_dict[T]['name']

    def _countTypes(self):
        type_dict = {}
        for tmpType in self.possible_types:
            type_dict[tmpType] = len([tmp['type'] for tmp in self.template_dict.values()\
                                      if tmp['type']==tmpType])
        return type_dict

    def _getInterpolators(self):
        interpolators = {}
        for T in range(self.num_templates):
            interpolators[T] = interp1d(self.wavelength(T), self.flux(T), bounds_error=False, fill_value=0.)
        return interpolators

    def numType(self, tmpType):
        try:
            return self._num_type[tmpType]
        except KeyError:
            warnings.warn('There are no templates of type {}, returning zero.'.format(tmpType))
            return 0
        except TypeError:
            raise TypeError('A value of type {} cannot be a template-type.'.format(type(tmpType)))

    def templateType(self, T):
        try:
            return self.template_dict[T]['type']
        except (KeyError, TypeError):
            raise ValueError('Template may be an integer [0...{}], but got a {} of value {} instead'.format(self.num_templates-1, type(T), T))

    def wavelength(self, T):
        try:
            return self._all_templates[T]['lambda']
        except (KeyError, TypeError):
            raise ValueError('Template may be an integer [0...{}], but got a {} of value {} instead'.format(self.num_templates-1, type(T), T))

    def flux(self, T):
        try:
            return self._all_templates[T]['flux']
        except (KeyError, TypeError):
            raise ValueError('Template may be an integer [0...{}], but got a {} of value {} instead'.format(self.num_templates-1, type(T), T))

    def name(self, T):
        try:
            return self._all_templates[T]['name']
        except (KeyError, TypeError):
            raise ValueError('Template may be an integer [0...{}], but got a {} of value {} instead'.format(self.num_templates-1, type(T), T))

    def interp(self, T, new_lambda):
        try:
            return self._interpolators[T](new_lambda)
        except (KeyError, TypeError):
            raise ValueError('Template may be an integer [0...{}], but got a {} of value {} instead'.format(self.num_templates-1, type(T), T))
