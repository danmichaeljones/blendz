import numpy as np
from blendz.config import _config

class Galaxy(object):
    def __init__(self, index, data_row, zero_point_frac):
        self.mag_data = data_row[_config.mag_cols]
        self.mag_sigma = data_row[_config.sigma_cols]
        self.ref_mag_data = self.mag_data[_config.ref_band]
        self.ref_mag_sigma = self.mag_sigma[_config.ref_band]
        self.zero_point_frac = zero_point_frac
        self.index = index

        self.convertMagnitudes()

    def convertMagnitudes(self):
        self.flux_data = 10.**(-0.4*self.mag_data)
        self.flux_sigma = (10.**(0.4*self.mag_sigma)-1.)*self.flux_data

        #If the magnitude is -99, the galaxy is not observed in that band,
        #    so set flux to zero and error to very high (1e108)
        noObs = np.where(self.mag_data==-99.)[0]
        for no in noObs:
            self.flux_data[no] = 0.
            self.flux_sigma[no] = 1e108

        #If the magnitude is 99, the galaxy is not detected in that band,
        #    so set flux to zero, error to abs(error)
        noDet = np.where(self.mag_data==99.)[0]
        for nd in noDet:
            self.flux_data[nd] = 0.
            self.flux_sigma[nd] = 10.**(-0.4*self.mag_sigma[nd])

        #Array of indices for bands where galaxy is neither not-observed nor not-detected
        # i.e., the galaxy IS seen in this band
        seen = [(self.mag_data!=99.) * (self.mag_data!=-99.)]
        notSeen = [~((self.mag_data!=99.) * (self.mag_data!=-99.))]
        #Add the zero point errors in
        #First, handle the observed objects
        self.flux_sigma = np.where(seen, np.sqrt(self.flux_sigma*self.flux_sigma+(self.zero_point_frac*self.flux_data)**2.), self.flux_sigma)[0]
        #Handle objects not seen differently to observed objects
        self.flux_sigma = np.where(notSeen, np.sqrt(self.flux_sigma*self.flux_sigma+(self.zero_point_frac*(self.flux_sigma/2.))**2), self.flux_sigma)[0]

        #Calculate colours
        #TODO: Check the colour sigmas!!!
        self.colour_data = self.flux_data / self.flux_data[_config.ref_band]
        self.colour_sigma = self.flux_sigma / self.flux_sigma[_config.ref_band]
