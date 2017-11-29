import warnings
import numpy as np
from blendz.config import _config
from blendz.photometry import PhotometryBase, Galaxy

#TODO: What are the errors on the colour data? Should just be simple division to
# propagate flux errors, but should actually calculate this rather than guessing
# to make sure it's right.

class Photometry(PhotometryBase):
    def __init__(self, config=None):
        super(Photometry, self).__init__()
        if config is None:
            warnings.warn('USING DEFAULT CONFIG IN PHOTOMETRY, USE THIS FOR TESTING PURPOSES ONLY!')
            self.config = _config
        else:
            self.config = config
        self.data_path = self.config.data_path
        self.zero_point_errors = self.config.zero_point_errors

        self.photo_data = np.loadtxt(self.data_path)
        self.num_to_load = np.shape(self.photo_data)[0]
        self.zero_point_frac = 10.**(0.4*self.zero_point_errors) - 1.

        self.loadGalaxies()

    def loadGalaxies(self):
        for g in xrange(self.num_to_load):
            mag_data = self.photo_data[g, self.config.mag_cols]
            mag_sigma = self.photo_data[g, self.config.sigma_cols]
            self.galaxies.append(Galaxy(mag_data, mag_sigma, self.config, self.zero_point_frac, g))
            if self.config.spec_z_col is not None:
                self.galaxies[g].spec_redshift = self.photo_data[g, self.config.spec_z_col]
