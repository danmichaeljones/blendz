from os.path import join
import warnings
import numpy as np
from scipy.interpolate import interp1d
from blendz.config import _config

class Templates(object):
    def __init__(self, template_dict=_config.template_dict):
        self.template_dict = template_dict
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
            interpolators[T] = interp1d(self.wavelength(T), self.flux(T))
        return interpolators

    def num_type(self, tmpType):
        try:
            return self._num_type[tmpType]
        except KeyError:
            warnings.warn('There are no templates of type {}, returning zero.'.format(tmpType))
            return 0

    def template_type(self, T):
        if type(T) == str:
            return self.template_dict[T]
        elif type(T) == int:
            return self.template_dict[self.template_names[T]]
        else:
            raise TypeError('The argument to template_type() should either be a string or an integer, got \
                             {} instead.'.format(type(T)))

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
