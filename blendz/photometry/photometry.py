import numpy as np
from blendz.config import _config

#TODO: What are the errors on the colour data? Should just be simple division to
# propagate flux errors, but should actually calculate this rather than guessing
# to make sure it's right.

class Photometry(object):
    def __init__(self, data_path=_config.data_path, zero_point_errors=_config.zero_point_errors):
        self.data_path = data_path
        self.zero_point_errors = zero_point_errors

        self.photo_data = np.loadtxt(self.data_path)
        self.zero_point_frac = 10.**(0.4*self.zero_point_errors) - 1.
        self.num_galaxies = np.shape(self.photo_data)[0]

        self.extract_magnitudes()
        #The conversion process also handles the non-observed/detected sources
        self.convert_magnitudes_to_fluxes()
        self.calculate_colours()

    def extract_magnitudes(self):
        self.mag_data = {}
        self.mag_sigma_data = {}
        self.ref_mag_data = {}
        self.ref_mag_sigma_data = {}
        for g in xrange(self.num_galaxies):
            self.mag_data[g] = self.photo_data[g, _config.mag_cols]
            self.mag_sigma_data[g] = self.photo_data[g, _config.sigma_cols]
            self.ref_mag_data[g] = self.mag_data[g][_config.ref_mag]
            self.ref_mag_sigma_data[g] = self.mag_sigma_data[g][_config.ref_mag]

    def convert_magnitudes_to_fluxes(self):
        self.flux_data = {}
        self.flux_sigma_data = {}
        for g in xrange(self.num_galaxies):
            self.flux_data[g] = 10.**(-0.4*self.mag_data[g])
            self.flux_sigma_data[g] = (10.**(0.4*self.mag_sigma_data[g])-1.)*self.flux_data[g]

            #If the magnitude is -99, the galaxy is not observed in that band,
            #    so set flux to zero and error to very high (1e108)
            noObs = np.where(self.mag_data[g]==-99.)[0]
            for no in noObs:
                self.flux_data[g][no] = 0.
                self.flux_sigma_data[g][no] = 1e108

            #If the magnitude is 99, the galaxy is not detected in that band,
            #    so set flux to zero, error to abs(error)
            noDet = np.where(self.mag_data[g]==99.)[0]
            for nd in noDet:
                self.flux_data[g][nd] = 0.
                self.flux_sigma_data[g][nd] = 10.**(-0.4*self.mag_sigma_data[g][nd])#np.abs(sigmaData[g][nd])

            #Array of indices for bands where galaxy is neither not-observed nor not-detected
            # i.e., the galaxy IS seen in this band
            seen = [(self.mag_data[g]!=99.) * (self.mag_data[g]!=-99.)]
            notSeen = [~((self.mag_data[g]!=99.) * (self.mag_data[g]!=-99.))]
            #Add the zero point errors in
            #First, handle the observed objects
            self.flux_sigma_data[g] = np.where(seen, np.sqrt(self.flux_sigma_data[g]*self.flux_sigma_data[g]+(self.zero_point_frac*self.flux_data[g])**2), self.flux_sigma_data[g])[0]
            #Handle objects not seen differently to observed objects
            self.flux_sigma_data[g] = np.where(notSeen, np.sqrt(self.flux_sigma_data[g]*self.flux_sigma_data[g]+(self.zero_point_frac*(self.flux_sigma_data[g]/2.))**2), self.flux_sigma_data[g])[0]

    def calculate_colours(self):
        #TODO: Check the colour sigmas!!!
        self.colour_data = {}
        self.colour_sigma_data = {}
        for g in xrange(self.num_galaxies):
            self.colour_data[g] = self.flux_data[g] / self.flux_data[g][_config.ref_mag]
            self.colour_sigma_data[g] = self.flux_sigma_data[g] / self.flux_sigma_data[g][_config.ref_mag]
