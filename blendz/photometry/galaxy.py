from builtins import *
import numpy as np


class Galaxy(object):
    def __init__(self, mag_data, mag_sigma, config, zero_point_frac, index):
        self.mag_data = mag_data
        self.mag_sigma = mag_sigma

        self.config = config
        self.ref_band = self.config.ref_band
        self.ref_mag_data = self.mag_data[self.ref_band]
        self.ref_mag_sigma = self.mag_sigma[self.ref_band]
        self.zero_point_frac = zero_point_frac
        self.index = index

        self.truth = {}

        self.convertMagnitudes()

    def convertMagnitudes(self):
        self.flux_data = 10.**(-0.4*self.mag_data)
        self.flux_sigma = (10.**(0.4*self.mag_sigma)-1.)*self.flux_data

        # Galaxy not OBSERVED - set flux to zero and error to very high (1e108)
        noObs = np.where(np.isclose(self.mag_data, self.config.no_observe_value, atol=0.01))[0]
        for no in noObs:
            self.flux_data[no] = 0.
            self.flux_sigma[no] = 1e108

        # Galaxy not DETECTED - set flux to zero, error to abs(error)
        noDet = np.where(np.isclose(self.mag_data, self.config.no_detect_value, atol=0.01))[0]
        for nd in noDet:
            self.flux_data[nd] = 0.
            self.flux_sigma[nd] = 10.**(-0.4*self.mag_sigma[nd])

        #Array of bools for bands where galaxy is neither not-observed nor not-detected
        # i.e., the galaxy IS seen in this band
        notSeen = np.array([((b in noObs) or (b in noDet)) for b in range(len(self.mag_data))])
        seen = np.array([not b for b in notSeen])

        if not seen[self.ref_band]:
            print(self.mag_data)
            raise ValueError('Galaxies must be observed in reference band, but galaxy {} had ref-band magnitude of {}, i.e., marked as a non-observation/detection.'.format(self.index, self.ref_mag_data))

        #Add the zero point errors in
        #First, handle the observed objects
        self.flux_sigma = np.where(seen, np.sqrt(self.flux_sigma*self.flux_sigma+(self.zero_point_frac*self.flux_data)**2.), self.flux_sigma)
        #Handle objects not seen differently to observed objects
        self.flux_sigma = np.where(notSeen, np.sqrt(self.flux_sigma*self.flux_sigma+(self.zero_point_frac*(self.flux_sigma/2.))**2), self.flux_sigma)

        #Create attribute for the flux in the reference band
        self.ref_flux_data = self.flux_data[self.config.ref_band]
        self.ref_flux_sigma = self.flux_sigma[self.config.ref_band]

        #Flux data and sigma, with the reference band removed
        self.flux_data_noRef = self.flux_data[self.config.non_ref_bands]
        self.flux_sigma_noRef = self.flux_sigma[self.config.non_ref_bands]
