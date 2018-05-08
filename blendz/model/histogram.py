from builtins import *
import numpy as np
from blendz.model import ModelBase


class Histogram(ModelBase):
    def __init__(self, n_mag_bins=15, n_z_bins=15, **kwargs):
        super(HistModel, self).__init__(**kwargs)
        self.n_mag_bins = n_mag_bins
        self.n_z_bins = n_z_bins
        self.z_bins = np.linspace(self.config.z_lo, self.config.z_hi, n_z_bins)
        self.mag_bins = np.linspace(self.config.ref_mag_lo, self.config.ref_mag_hi, n_mag_bins)
        prior_params_full = np.append(self.prior_params, 1.-sum(self.prior_params))
        num_types = len(self.responses.templates.possible_types)
        self.histogram = prior_params_full.reshape((n_z_bins, n_mag_bins,
                                                    num_types))

    def correlationFunction(self, redshifts):
        if len(redshifts)==1:
            return 0.
        elif len(redshifts)==2:
            theta = self.config.angular_resolution
            redshifts = np.sort(redshifts)
            r_2 = self.comovingSeparation(0., redshifts[1])
            delta_r = self.comovingSeparation(redshifts[0], redshifts[1])
            power = 1. - (self.config.gamma/2.)
            one = (self.config.r0**2.) / (power * r_2 * r_2 * theta * theta)
            two = (delta_r**2 + (r_2 * r_2 * theta * theta)) / (self.config.r0**2.)
            three = (delta_r**2) / (self.config.r0**2.)
            return one * ( (two**power) - (three**power) )
        else:
            #Could define this recursively by calling with a
            #len 2 slice of redshifts - assuming bispectrum and above = 0
            raise NotImplementedError('No N>2 yet...')

    def lnPriorCalibrationPrior(self):
        if (sum(self.prior_params) <= 1.) and \
                np.all(self.prior_params >= 0.) and \
                np.all(self.prior_params <= 1.):
            return 0.
        else:
            return -np.inf

    def lnPrior(self, redshift, magnitude):
        iz = np.digitize(redshift, self.z_bins)
        im = np.digitize(magnitude, self.mag_bins)
        return np.log(self.histogram[iz, im, :])
