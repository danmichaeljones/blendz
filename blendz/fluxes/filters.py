import numpy as np

class Filters(object):
    def __init__(self, filter_path, filter_names, file_extension='.res'):
        self.filter_path = filter_path
        self.filter_names = filter_names
        self.file_extension = file_extension
        self.num_filters = len(self.filter_names)

        self._all_filters = {}
        #load_filters reads filters in from file *and* calculates the normalisations
        self.load_filters()

    def load_filters(self, filenames=None, filepath=None, file_extension=None):
        #Default arguments evaluated at define, self only available at function call, so use None instead
        if filenames is None:
            filenames = self.filter_names
        if filepath is None:
            filepath = self.filter_path
        if file_extension is None:
            file_extension = self.file_extension

        for F in xrange(len(filenames)):
            #Read from file
            self._all_filters[F] = {}
            self._all_filters[F]['lambda'], self._all_filters[F]['response'] = \
                    np.loadtxt(filepath + filenames[F] + file_extension, unpack=True)
            #Calculate normalisation
            self._all_filters[F]['norm'] = np.trapz(self._all_filters[F]['response'] / self._all_filters[F]['lambda'],\
                                            x = self._all_filters[F]['lambda'])

    def wavelength(self, F):
        return self._all_filters[F]['lambda']

    def response(self, F):
        return self._all_filters[F]['response']

    def norm(self, F):
        return self._all_filters[F]['norm']
