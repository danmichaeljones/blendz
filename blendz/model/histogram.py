from builtins import *
import numpy as np
from tqdm import tqdm
from blendz.model import ModelBase


class Histogram(ModelBase):
    def __init__(self, n_mag_bins=15, n_z_bins=15, **kwargs):
        super(Histogram, self).__init__(**kwargs)
        self.n_mag_bins = n_mag_bins
        self.n_z_bins = n_z_bins
        self.z_bins = np.linspace(self.config.z_lo, self.config.z_hi, n_z_bins)
        self.mag_bins = np.linspace(self.config.ref_mag_lo, self.config.ref_mag_hi, n_mag_bins)
        self.num_types = len(self.responses.templates.possible_types)

        if self.prior_params is not np.nan:
            if len(self.prior_params) != self.n_z_bins * self.n_mag_bins * self.num_types:
                raise ValueError('Wrong number of prior parameters.')

        if self.prior_params is not np.nan:
            self.histogram = self.prior_params.reshape((self.n_z_bins, self.n_mag_bins, self.num_types))
        else:
            self.histogram = None

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

    def calibrate(self, photometry, num_samps=10000, chain_save_interval=None,
                  chain_save_path=None, config_save_path=None, burn_len=0):

        hist_size = (self.n_z_bins, self.n_mag_bins, self.num_types)
        flat_size = self.n_z_bins * self.n_mag_bins * self.num_types

        #Init with a flat histogram
        new_heights = np.ones(hist_size)
        new_heights /= sum(new_heights)

        height_chain = np.zeros((num_samps, flat_size))

        for s in tqdm(range(num_samps)):

            template_inds = np.zeros(photometry.num_galaxies)
            magnitude_inds = np.zeros(photometry.num_galaxies)
            redshift_inds = np.zeros(photometry.num_galaxies)
            for g in photometry:
                iz = np.digitize(g.truth[0]['redshift'], self.z_bins)
                im = np.digitize(g.ref_mag_data, self.mag_bins)
                prob = new_heights.reshape(hist_size)[iz, im, :]

                template_inds[g.index] = np.random.choice(len(prob), p=prob/sum(prob))
                redshift_inds[g.index] = iz
                magnitude_inds[g.index] = im

            #Convert T/Z/M bin indices into bin counts
            flat_inds = np.ravel_multi_index((np.int_(redshift_inds),
                                              np.int_(magnitude_inds),
                                              np.int_(template_inds)),
                                              hist_size)
            bin_counts = np.bincount(flat_inds, minlength=flat_size)

            #Draw dirichlet given bin counts to give new histogram heights
            new_heights = np.random.dirichlet(bin_counts)
            height_chain[s, :] = new_heights

            if (chain_save_interval is not None) and (chain_save_path is not None):
                if (s % chain_save_interval == 0):
                    np.savetxt(chain_save_path, height_chain)

        if chain_save_path is not None:
            np.savetxt(chain_save_path, height_chain)

        mean_params = np.mean(height_chain[burn_len:, :], axis=0)
        self.prior_params = mean_params
        self.histogram = self.prior_params.reshape((self.n_z_bins, self.n_mag_bins, self.num_types))

        if config_save_path is not None:
            array_str = np.array2string(mean_params, separator=',', max_line_width=np.inf)[1:-1]
            cfg_str = u'[Run]\n\nprior_params = ' + array_str
            with open(config_save_path, 'w') as config_file:
                config_file.write(cfg_str)
