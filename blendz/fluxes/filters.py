from builtins import *
from os.path import join
import numpy as np
from blendz import Configuration

class Filters(object):
    """Definitions of the photometric filter bands observations have been made with.

    More description of filters would presumably go here.

    Args:
        config (blendz.Configuratio): Description of parameter `config`. Defaults to None.

        kwargs : Description of parameter `kwargs`.

    Returns:
        type: Description of returned object.

    """
    def __init__(self, config=None, **kwargs):
        self.config = Configuration(**kwargs)
        if config is not None:
            self.config.mergeFromOther(config)

        self.filter_path = self.config.filter_path
        self.filter_names = self.config.filters
        self.num_filters = len(self.filter_names)

        self._all_filters = {}
        #load_filters reads filters in from file *and* calculates the normalisations
        self.load_filters()

    def load_filters(self, filenames=None, filepath=None):
        #Default arguments evaluated at define, self only available at function call, so use None instead
        if filenames is None:
            filenames = self.filter_names
        if filepath is None:
            filepath = self.filter_path

        for F in range(len(filenames)):
            #Read from file
            self._all_filters[F] = {}
            self._all_filters[F]['lambda'], self._all_filters[F]['response'] = \
                    np.loadtxt(join(filepath, filenames[F]), unpack=True)
            flt_order = np.argsort(self._all_filters[F]['lambda'])
            self._all_filters[F]['lambda'] = self._all_filters[F]['lambda'][flt_order]
            self._all_filters[F]['response'] = self._all_filters[F]['response'][flt_order]
            #Calculate normalisation
            self._all_filters[F]['norm'] = np.trapz(self._all_filters[F]['response'] / self._all_filters[F]['lambda'],\
                                            x = self._all_filters[F]['lambda'])

    def wavelength(self, F):
        try:
            return self._all_filters[F]['lambda']
        except (KeyError, TypeError):
            raise ValueError('Filter may be an integer [0...{}], but got a {} of value {} instead'.format(self.num_filters-1, type(F), F))

    def response(self, F):
        try:
            return self._all_filters[F]['response']
        except (KeyError, TypeError):
            raise ValueError('Filter may be an integer [0...{}], but got a {} of value {} instead'.format(self.num_filters-1, type(F), F))

    def norm(self, F):
        try:
            return self._all_filters[F]['norm']
        except (KeyError, TypeError):
            raise ValueError('Filter may be an integer [0...{}], but got a {} of value {} instead'.format(self.num_filters-1, type(F), F))
