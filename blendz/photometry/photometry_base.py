from builtins import *
import warnings
from contextlib import contextmanager
import numpy as np
from blendz import Configuration

class PhotometryBase(object):
    def __init__(self, config=None, **kwargs):
        self.all_galaxies = []
        self.current_galaxy = None

        self.config = Configuration(**kwargs)
        if config is not None:
            self.config.mergeFromOther(config)

    def iterate(self, start=None, stop=None, step=None):
        out_list = self.all_galaxies[start:stop:step]
        for gal in out_list:
            self.current_galaxy = gal
            yield gal
        #Clean up by resetting current_galaxy to None when done
        self.current_galaxy = None

    def __iter__(self):
        iterator = self.iterate()
        for g in range(len(self.all_galaxies)):
            yield next(iterator)
        #Clean up by resetting current_galaxy to None when done
        self.current_galaxy = None

    def __getitem__(self, key):
        out = self.all_galaxies[key]
        if isinstance(out, list):
            warnings.warn('This slice of the photometry returns a list of '
                          + 'Galaxy objects, but doesn\'t update current_galaxy, '
                          + 'so this should not be used for iterating over if any '
                          + 'methods are called. Instead, you should use the '
                          + 'iterate(start, stop, step) method for iterating.')
        return out

    @contextmanager
    def galaxy(self, index):
        gal = self.all_galaxies[index]
        self.current_galaxy = gal
        yield gal
        self.current_galaxy = None

    @property
    def num_galaxies(self):
        '''
        Read only attribute returning the number of galaxies contained in the photometry data set.
        '''
        return len(self.all_galaxies)
