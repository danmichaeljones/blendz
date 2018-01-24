from builtins import *
import warnings
import numpy as np
from blendz.config import Configuration
from blendz.photometry import PhotometryBase, Galaxy
from blendz.model import BPZ
from blendz.utilities import incrementCount, Reject

class SimulatedPhotometry(PhotometryBase):
    def __init__(self, num_sims, config=None, num_components=1, max_redshift=None,
                max_err_frac=0.1, model=None, seed=None, random_err=True,
                measurement_component_specification=None, magnitude_bounds=[20., 32], **kwargs):
        super(SimulatedPhotometry, self).__init__()

        if model is not None:
            #Set model, take default+kwargs+model config and warn user if config also provided
            self.config = Configuration(**kwargs)
            self.config.mergeFromOther(self.model.config)
            self.model = model
            self.responses = self.model.responses

            if config is not None:
                warnings.warn("""A configuration object was provided to
                                SimulatedPhotometry as well as a Responses
                                object, though these should be mutually exclusive.
                                The configuration provided will be ignored.""")
        else:
            #Config given, take default+kwargs+given config
            self.config = Configuration(**kwargs)
            if config is not None:
                self.config.mergeFromOther(config)

            self.model = BPZ(config=self.config)
            self.responses = self.model.responses

        self.num_sims = num_sims
        self.num_components = num_components
        self.max_err_frac = max_err_frac
        self.random_err = random_err
        self.num_measurements = self.responses.filters.num_filters
        if max_redshift is None:
            self.max_redshift = self.config.z_hi
        else:
            self.max_redshift = max_redshift
        self.magnitude_bounds = magnitude_bounds

        self.zero_point_errors = self.config.zero_point_errors
        self.zero_point_frac = 10.**(0.4*self.zero_point_errors) - 1.

        #Use an incrementing counter for the seed to make sure its always
        #different between various different function calls
        if seed is None:
            self.sim_seed = incrementCount(np.random.randint(1e9))
        else:
            self.sim_seed = incrementCount(seed)

        self.simulateRandomGalaxies(self.num_sims, self.num_components,
                                    max_redshift=self.max_redshift,
                                    max_err_frac=self.max_err_frac,
                                    measurement_component_specification=measurement_component_specification,
                                    magnitude_bounds=self.magnitude_bounds)

    def generateBlendMagnitude(self, num_components, redshifts, scales, template_indices, err_frac):
        true_flux = np.zeros(self.responses.filters.num_filters)
        for c in range(num_components):
            true_flux += self.responses(template_indices[c], None, redshifts[c]) * scales[c] * self.model.measurement_component_mapping[c, :]
        fracs = np.zeros(num_components)
        for c in range(num_components):
            fracs[c] = ((self.responses(template_indices[c], None, redshifts[c]) * scales[c]) / true_flux)[self.config.ref_band]
        rand_err  = (np.random.rand(self.responses.filters.num_filters) * (true_flux * err_frac * 2)) - (true_flux * err_frac)
        obs_flux = true_flux + rand_err
        flux_err = true_flux * err_frac
        obs_mag = np.log10(obs_flux) / (-0.4)
        mag_err = np.log10((flux_err/obs_flux)+1.) / (-0.4)
        return obs_mag, mag_err, fracs

    def drawBlendFromPrior(self, num_components, max_redshift=None, magnitude_bounds=None):
        '''
        Use the priors defined in the Model to draw a random blend of components.

        We don't use the magnitudes we sample to generate the photometric data
        but the true magnitudes we generate should match the ones drawn here - consistency test
        '''
        if max_redshift is None:
            max_redshift = self.max_redshift
        if magnitude_bounds is None:
            magnitude_bounds = self.magnitude_bounds

        magnitudes = np.zeros(num_components)
        templates = np.zeros(num_components, dtype=int)
        redshifts = np.zeros(num_components)
        true_ref_response = np.zeros(num_components)
        for c in range(num_components):
            #Sample magnitude
            rj = Reject(lambda m: np.exp(self.model.lnMagnitudePrior(m)), magnitude_bounds[0],
                                         magnitude_bounds[1], seed=self.sim_seed.next())
            magnitudes[c] = rj.sample(1)
            #Sample template given magnitude
            rstate = np.random.RandomState(self.sim_seed.next())
            tmp_prior = np.zeros(self.responses.templates.num_templates)
            for t in range(self.responses.templates.num_templates):
                tmp_type_t = self.responses.templates.templateType(t)
                tmp_prior[t] = np.exp(self.model.lnTemplatePrior(tmp_type_t, magnitudes[c]))
            templates[c] = rstate.choice(self.responses.templates.num_templates, p=tmp_prior)
            tmp_type = self.responses.templates.templateType(templates[c])
            #Sample redshift, given template and magnitude
            #Include redshift (c+1)-point-correlation if other redshifts drawn
            if c==0:
                fn = lambda z: np.exp(self.model.lnRedshiftPrior(z, tmp_type, magnitudes[c]))
            else:
                fn = lambda z: np.exp(self.model.lnRedshiftPrior(z, tmp_type, magnitudes[c])) * \
                        (1. + self.model.correlationFunction(np.append(redshifts[:c], z)))
            rj = Reject(fn, 0, max_redshift, seed=self.sim_seed.next())
            redshifts[c] = rj.sample(1)
            #Get flux response at these parameters in the reference band
            true_ref_response[c] = self.responses(templates[c], self.config.ref_band, redshifts[c]) # TODO: Measurment mapping???
        #The magnitudes that have been drawn deterministically set the reference-band flux fraction
        fluxes = 10.**(-0.4*magnitudes)
        total_flux = np.sum(fluxes)
        fractions = fluxes / total_flux
        #The fraction and the template responses together give the component scaling
        scales = fractions / true_ref_response
        scales /= np.sum(scales)
        #The source normalisation is set by the scaled responses
        #and the sum (in linear, not log space) of the drawn magnitudes
        source_normalisation = total_flux / np.sum(scales * true_ref_response)
        #Finally, the component normalisation used to generate the data is the
        #component scaling * source_normalisation, i.e., A*a_alpha
        component_normalisation = scales * source_normalisation
        #Sort component parameters we're returning by redshift
        #order = np.argsort(redshifts)
        #component_normalisation = component_normalisation[order]
        #templates = templates[order]
        #redshifts = redshifts[order]
        return redshifts, component_normalisation, templates, magnitudes



    def randomBlend(self, num_components, max_redshift=None, max_err_frac=None, \
                    magnitude_bounds=None, sort_redshifts=None):
        if max_redshift is None:
            max_redshift = self.max_redshift
        if max_err_frac is None:
            max_err_frac = self.max_err_frac
        if magnitude_bounds is None:
            magnitude_bounds = self.magnitude_bounds
        if sort_redshifts is None:
            sort_redshifts = self.config.sort_redshifts

        np.random.seed(self.sim_seed.next())
        if self.random_err:
            sim_err_frac = np.random.rand() * max_err_frac
        else:
            sim_err_frac = max_err_frac
        sim_redshift, sim_scale, sim_template, sim_magnitude = self.drawBlendFromPrior(num_components, max_redshift=max_redshift, magnitude_bounds=magnitude_bounds)

        obs_mag, mag_err, fracs = self.generateBlendMagnitude(num_components, sim_redshift, sim_scale, sim_template, sim_err_frac)

        #Order component truths before saving, either sort redshift...
        if sort_redshifts:
            order = np.argsort(sim_redshift)
            sim_redshift = sim_redshift[order]
            sim_scale = sim_scale[order]
            sim_template = sim_template[order]
            sim_magnitude = sim_magnitude[order]
            fracs = fracs[order]
        #...or magnitudes
        else:
            order = np.argsort(sim_magnitude)
            sim_redshift = sim_redshift[order]
            sim_scale = sim_scale[order]
            sim_template = sim_template[order]
            sim_magnitude = sim_magnitude[order]
            fracs = fracs[order]

        truth = {}
        truth['num_components'] = num_components
        for c in range(num_components):
            truth[c] = {'redshift': sim_redshift[c], 'scale': sim_scale[c],
            'template': sim_template[c], 'fraction': fracs[c], 'magnitude': sim_magnitude[c]}

        return obs_mag, mag_err, truth

    def simulateRandomGalaxies(self, num_sims, num_components, max_redshift=None, max_err_frac=None, measurement_component_specification=None, magnitude_bounds=None):
        if max_redshift is None:
            max_redshift = self.max_redshift
        if max_err_frac is None:
            max_err_frac = self.max_err_frac
        if magnitude_bounds is None:
            magnitude_bounds = self.magnitude_bounds

        self.model._setMeasurementComponentMapping(measurement_component_specification, num_components)
        for g in range(num_sims):
            mag_data, mag_sigma, truth = self.randomBlend(num_components, max_redshift, max_err_frac, magnitude_bounds=magnitude_bounds)
            new_galaxy = Galaxy(mag_data, mag_sigma, self.config, self.zero_point_frac, g)
            new_galaxy.truth = truth
            self.all_galaxies.append(new_galaxy)

    def simulateGalaxies(self, redshifts, scales, templates, err_frac):
        for g in range(len(redshifts)):
            num_components = len(redshifts[g])
            obs_mag, mag_err, fracs = self.generateBlendMagnitude(
                                        num_components, redshifts[g], scales[g],
                                        templates[g], err_frac)
            truth = {}
            truth['num_components'] = num_components
            for c in range(num_components):
                truth[c] = {'redshift': redshifts[g][c],
                            'scale': scales[g][c],
                            'fraction': fracs[c],
                            'template': templates[g][c]}
            new_galaxy = Galaxy(obs_mag, mag_err, self.config,
                                self.zero_point_frac, g)
            new_galaxy.truth = truth
            self.all_galaxies.append(new_galaxy)
