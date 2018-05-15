from builtins import *
import warnings
import numpy as np
from blendz import Configuration
from blendz.photometry import PhotometryBase, Galaxy

class Photometry(PhotometryBase):
    def __init__(self, config=None, **kwargs):
        super(Photometry, self).__init__(config=config, **kwargs)
        self.data_path = self.config.data_path
        self.zero_point_errors = self.config.zero_point_errors
        if self.config.data_is_csv:
            self.photo_data = np.loadtxt(self.data_path,
                skiprows=self.config.skip_data_rows, delimiter=',')
        else:
            self.photo_data = np.loadtxt(self.data_path,
                skiprows=self.config.skip_data_rows)
        self.num_to_load = np.shape(self.photo_data)[0]
        self.zero_point_frac = 10.**(0.4*self.zero_point_errors) - 1.

        self.loadGalaxies()

    def loadGalaxies(self):
        for g in range(self.num_to_load):
            mag_data = self.photo_data[g, self.config.mag_cols]
            mag_sigma = self.photo_data[g, self.config.sigma_cols]
            self.all_galaxies.append(Galaxy(mag_data, mag_sigma, self.config, self.zero_point_frac, g))
            if self.config.spec_z_col is not None:
                self.all_galaxies[g].truth['num_components'] = len(self.config.spec_z_col)
                for c in range(len(self.config.spec_z_col)):
                    self.all_galaxies[g].truth[c] = {}
                    self.all_galaxies[g].truth[c]['redshift'] = self.photo_data[g, self.config.spec_z_col[c]]

            if self.config.magnitude_limit_col is not None:
                self.all_galaxies[g].magnitude_limit = self.photo_data[g, self.config.magnitude_limit_col]
            else:
                self.all_galaxies[g].magnitude_limit = self.config.magnitude_limit
