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
                    cmp_redshift = self.photo_data[g, self.config.spec_z_col[c]]
                    self.all_galaxies[g].truth[c] = {'redshift': cmp_redshift}
                    if not (self.config.z_lo <= cmp_redshift <= self.config.z_hi):
                        warn_str = 'Galaxy {} has a spectroscopic redshift of {} in component {}. '.format(g, cmp_redshift, c) \
                                 + 'This is outside of the prior range ({} -> {}).'.format(self.config.z_lo, self.config.z_hi)
                        warnings.warn(warn_str)

            if self.config.magnitude_limit_col is not None:
                self.all_galaxies[g].magnitude_limit = self.photo_data[g, self.config.magnitude_limit_col]
            else:
                self.all_galaxies[g].magnitude_limit = self.config.magnitude_limit

            #If ref_mag_hi_sigma is not set, use ref_mag_hi instead, but prefer sigma
            if self.config.ref_mag_hi_sigma is None:
                if self.config.ref_mag_hi is not None:
                    self.all_galaxies[g].ref_mag_hi = self.config.ref_mag_hi
                else:
                    raise ValueError('One of ref_mag_hi or ref_mag_hi_sigma must be set in config to load photometry.')
            else:
                ref_flux_hi = self.config.ref_mag_hi_sigma * self.all_galaxies[g].ref_flux_sigma
                ref_mag_hi = -2.5 * np.log10(ref_flux_hi)
                self.all_galaxies[g].ref_mag_hi = ref_mag_hi
