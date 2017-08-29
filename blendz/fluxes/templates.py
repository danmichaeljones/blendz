import numpy as np
from scipy.interpolate import interp1d
import warnings

class Templates(object):
    def __init__(self, template_path, template_dict, file_extension='.sed'):
        self.template_path = template_path
        self.template_dict = template_dict
        self.file_extension = file_extension
        self.template_names = self.template_dict.keys() #for consistent ordering from non-ordered dict
        self.num_templates = len(self.template_dict)
        self.possible_types = set(tmp for tmp in template_dict.values())

        self._all_templates = {}
        self.load_templates()
        self._num_type = self._count_types()
        self._interpolators = self._get_interpolators()

    def load_templates(self, filenames=None, filepath=None, file_extension=None):
        #Default arguments evaluated at define, self only available at function call, so use None instead
        if filenames is None:
            filenames = self.template_names
        if filepath is None:
            filepath = self.template_path
        if file_extension is None:
            file_extension = self.file_extension

        for T in xrange(len(filenames)):
            self._all_templates[T] = {}
            self._all_templates[T]['lambda'], self._all_templates[T]['flux'] = \
                    np.loadtxt(filepath + filenames[T] + file_extension, unpack=True)

    def _count_types(self):
        type_dict = {}
        for tmpType in self.possible_types:
            type_dict[tmpType] = len([T for T in self.template_dict.values() if T==tmpType])
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
        return self._all_templates[T]['lambda']

    def flux(self, T):
        return self._all_templates[T]['flux']

    def interp(self, T, newLambda):
        return self._interpolators[T](newLambda)
