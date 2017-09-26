import warnings
import numpy as np
from blendz.config import _config

#TODO: What are the errors on the colour data? Should just be simple division to
# propagate flux errors, but should actually calculate this rather than guessing
# to make sure it's right.

class PhotometryBase(object):
    def __init__(self):
        self.galaxies = []
        self.current_galaxy = None

    def iterate(self, start=None, stop=None, step=None):
        out_list = self.galaxies[start:stop:step]
        for gal in out_list:
            self.current_galaxy = gal
            yield gal
        #Clean up by resetting current_galaxy to None when done
        self.current_galaxy = None

    def __iter__(self):
        iterator = self.iterate()
        for g in xrange(len(self.galaxies)):
            yield next(iterator)
        #Clean up by resetting current_galaxy to None when done
        self.current_galaxy = None

    def __getitem__(self, key):
        out = self.galaxies[key]
        if isinstance(out, list):
            warnings.warn("""This slice of the photometry returns a list of Galaxy objects, but doesn't
                             update current_galaxy, so this should not be used for iterating over if any
                             methods are called. Instead, you should use the
                             iterate(start, stop, step) method for iterating.""")
        return out

    @property
    def num_galaxies(self):
        '''
        Read only attribute returning the number of galaxies contained in the photometry data set.
        '''
        return len(self.galaxies)
