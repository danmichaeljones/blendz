import numpy as np
from blendz.config import _config
from blendz.photometry import PhotometryBase, Galaxy

#TODO: What are the errors on the colour data? Should just be simple division to
# propagate flux errors, but should actually calculate this rather than guessing
# to make sure it's right.

class Photometry(PhotometryBase):
    def __init__(self, data_path=_config.data_path, zero_point_errors=_config.zero_point_errors):
        super(Photometry, self).__init__()

        self.data_path = data_path
        self.zero_point_errors = zero_point_errors

        self.photo_data = np.loadtxt(self.data_path)
        self.num_to_load = np.shape(self.photo_data)[0]
        self.zero_point_frac = 10.**(0.4*self.zero_point_errors) - 1.

        self.loadGalaxies()

    def loadGalaxies(self):
        for g in xrange(self.num_to_load):
            mag_data = self.photo_data[g, _config.mag_cols]
            mag_sigma = self.photo_data[g, _config.sigma_cols]
            self.galaxies.append(Galaxy(mag_data, mag_sigma, _config.ref_band, self.zero_point_frac, g))
