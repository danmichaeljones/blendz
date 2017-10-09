import warnings
import numpy as np
from blendz.config import _config
from blendz.photometry import PhotometryBase, Galaxy

class SimulatedPhotometry(PhotometryBase):
    def __init__(self, num_sims, config=None, num_components=1, max_redshift=6., max_scale=50., max_err_frac=0.1, responses=None, seed=None):
        super(SimulatedPhotometry, self).__init__()

        if responses is not None:
            #Set responses, take its config and warn user if config also provided
            self.responses = responses
            self.config = self.responses.config
            if config is not None:
                warnings.warn("""A configuration object was provided to
                                SimulatedPhotometry as well as a Responses
                                object, though these should be mutually exclusive.
                                The configuration provided will be ignored.""")
        else:
            if config is None:
                self.confif = _config
            else:
                self.config = config
            self.responses = Responses(config=self.config)

        self.num_sims = num_sims
        self.num_components = num_components
        self.max_redshift = max_redshift
        self.max_scale = max_scale
        self.max_err_frac = max_err_frac
        self.rstate = np.random.RandomState(seed) #Default seed is none -> random

        self.zero_point_errors = self.config.zero_point_errors
        self.zero_point_frac = 10.**(0.4*self.zero_point_errors) - 1.

        self.simulateRandomGalaxies(self.num_sims, self.num_components, self.max_redshift, self.max_scale, self.max_err_frac)

    def generateBlendMagnitude(self, num_components, redshifts, scales, template_indices, err_frac):
        true_flux = np.zeros(self.responses.filters.num_filters)
        for c in xrange(num_components):
            true_flux += self.responses(template_indices[c], None, redshifts[c]) * scales[c]
        fracs = np.zeros(num_components)
        for c in xrange(num_components):
            fracs[c] = ((self.responses(template_indices[c], None, redshifts[c]) * scales[c]) / true_flux)[self.config.ref_band]
        rand_err  = (np.random.rand(self.responses.filters.num_filters) * (true_flux * err_frac * 2)) - (true_flux * err_frac)
        obs_flux = true_flux + rand_err
        flux_err = true_flux * err_frac
        obs_mag = np.log10(obs_flux) / (-0.4)
        mag_err = np.log10((flux_err/obs_flux)+1.) / (-0.4)
        return obs_mag, mag_err, fracs

    def randomBlend(self, num_components, max_redshift, max_scale, max_err_frac):
        sim_redshift = np.sort(self.rstate.rand(num_components) * max_redshift)
        sim_scale = self.rstate.rand(num_components) * max_scale
        sim_template = self.rstate.randint(0, self.responses.templates.num_templates, num_components)
        sim_err_frac = self.rstate.rand() * max_err_frac

        obs_mag, mag_err, fracs = self.generateBlendMagnitude(num_components, sim_redshift, sim_scale, sim_template, sim_err_frac)

        truth = {}
        truth['num_components'] = num_components
        for c in xrange(num_components):
            truth[c] = {'redshift': sim_redshift[c], 'scale': sim_scale[c],
            'template': sim_template[c], 'fraction': fracs[c]}

        return obs_mag, mag_err, truth

    def simulateRandomGalaxies(self, num_sims, num_components, max_redshift, max_scale, max_err_frac):
        for g in xrange(num_sims):
            mag_data, mag_sigma, truth = self.randomBlend(num_components, max_redshift, max_scale, max_err_frac)
            new_galaxy = Galaxy(mag_data, mag_sigma, self.config.ref_band, self.zero_point_frac, g)
            new_galaxy.truth = truth
            self.galaxies.append(new_galaxy)

    def simulateGalaxies(self, redshifts, scales, templates, err_frac):
        for g in xrange(len(redshifts)):
            num_components = len(redshifts[g])
            obs_mag, mag_err, fracs = self.generateBlendMagnitude(
                                        num_components, redshifts[g], scales[g],
                                        templates[g], err_frac)
            truth = {}
            truth['num_components'] = num_components
            for c in xrange(num_components):
                truth[c] = {'redshift': redshifts[g][c],
                            'scale': scales[g][c],
                            'fraction': fracs[c],
                            'template': templates[g][c]}
            new_galaxy = Galaxy(obs_mag, mag_err, self.config.ref_band,
                                self.zero_point_frac, g)
            new_galaxy.truth = truth
            self.galaxies.append(new_galaxy)
