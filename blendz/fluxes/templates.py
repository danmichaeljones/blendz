from os.path import join
import warnings
import numpy as np
from scipy.interpolate import interp1d
from blendz.config import _config

class Templates(object):
    def __init__(self, config=None):
        if config is None:
            self.config = _config
        else:
            self.config = config
        self.template_dict = self.config.template_dict
        self.num_templates = len(self.template_dict)
        self.possible_types = set(tmp['type'] for tmp in self.template_dict.values())

        self.load_templates()
        self._num_type = self._count_types()
        self._interpolators = self._get_interpolators()

    def load_templates(self):
        self._all_templates = {}
        for T in xrange(self.num_templates):
            self._all_templates[T] = {}
            self._all_templates[T]['lambda'], self._all_templates[T]['flux'] = \
                        np.loadtxt(self.template_dict[T]['path'], unpack=True)
            self._all_templates[T]['name'] = self.template_dict[T]['name']

    def _count_types(self):
        type_dict = {}
        for tmpType in self.possible_types:
            type_dict[tmpType] = len([tmp['type'] for tmp in self.template_dict.values()\
                                      if tmp['type']==tmpType])
        return type_dict

    def _get_interpolators(self):
        interpolators = {}
        for T in xrange(self.num_templates):
            interpolators[T] = interp1d(self.wavelength(T), self.flux(T), bounds_error=False, fill_value='extrapolate')
        return interpolators

    def num_type(self, tmpType):
        try:
            return self._num_type[tmpType]
        except KeyError:
            warnings.warn('There are no templates of type {}, returning zero.'.format(tmpType))
            return 0

    def template_type(self, T):
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

    def interp(self, T, newLambda):
        try:
            return self._interpolators[T](newLambda)
        except (KeyError, TypeError):
            raise ValueError('Template may be an integer [0...{}], but got a {} of value {} instead'.format(self.num_templates-1, type(T), T))
