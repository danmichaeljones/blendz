import warnings
import numpy as np
from blendz.config import _config
from blendz.photometry import Galaxy

#TODO: What are the errors on the colour data? Should just be simple division to
# propagate flux errors, but should actually calculate this rather than guessing
# to make sure it's right.

class Photometry(object):
    def __init__(self, data_path=_config.data_path, zero_point_errors=_config.zero_point_errors):
        self.data_path = data_path
        self.zero_point_errors = zero_point_errors

        self.photo_data = np.loadtxt(self.data_path)
        self.zero_point_frac = 10.**(0.4*self.zero_point_errors) - 1.
        self.num_galaxies = np.shape(self.photo_data)[0]

        self.getGalaxies()
        self.current_galaxy = None

    def getGalaxies(self):
        self.galaxies = []
        for g in xrange(self.num_galaxies):
            data_row = self.photo_data[g, :]
            self.galaxies.append(Galaxy(g, data_row, self.zero_point_frac))

    def iterate(self, start=None, stop=None, step=None):
        out_list = self.galaxies[start:stop:step]
        for gal in out_list:
            self.current_galaxy = gal
            yield gal

    def __iter__(self):
        iterator = self.iterate()
        for g in xrange(self.num_galaxies):
            yield next(iterator)

    def __getitem__(self, key):
        out = self.galaxies[key]
        if isinstance(out, list):
            warnings.warn("""This slice of the photometry returns a list of Galaxy objects, but doesn't
                             update current_galaxy, so this should not be used for iterating over if any
                             methods are called. Instead, you should use the
                             Photometry.iterate(start, stop, step) method for iterating""")

        return out
