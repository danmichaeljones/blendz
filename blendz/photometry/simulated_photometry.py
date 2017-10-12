import warnings
import numpy as np
from samplepy import Rejection
from blendz.config import _config
from blendz.photometry import PhotometryBase, Galaxy
from blendz.model import BPZ
from blendz.utilities import incrementCount

class SimulatedPhotometry(PhotometryBase):
    def __init__(self, num_sims, config=None, num_components=1, max_redshift=6.,
                max_scale=50., max_err_frac=0.1, model=None, seed=None,
                measurement_component_specification=None, magnitude_bounds=[20., 32]):
        super(SimulatedPhotometry, self).__init__()

        if model is not None:
            #Set model, take its config and warn user if config also provided
            self.model = model
            self.responses = self.model.responses
            self.config = self.model.config
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
            self.model = BPZ(config=self.config)
            self.responses = self.model.responses

        self.num_sims = num_sims
        self.num_components = num_components
        self.max_redshift = max_redshift
        self.max_scale = max_scale
        self.max_err_frac = max_err_frac
        self.num_measurements = self.responses.filters.num_filters

        self.zero_point_errors = self.config.zero_point_errors
        self.zero_point_frac = 10.**(0.4*self.zero_point_errors) - 1.

        #Use an incrementing counter for the seed to make sure its always
        #different between various different function calls
        if seed is None:
            self.seed = incrementCount(np.random.randint(1e9))
        else:
            self.seed = incrementCount(seed)

        self.simulateRandomGalaxies(self.num_sims, self.num_components,
                                    self.max_redshift, self.max_scale,
                                    self.max_err_frac, measurement_component_specification,
                                    magnitude_bounds=magnitude_bounds)

    def setMeasurementComponentMapping(self, specification, num_components):
        #MAKES MORE SENSE IN PHOTOMETRY????????
        '''
        Construct the measurement-component mapping matrix from the specification.

        If specification is None, it is assumed that all measurements contain
        num_components components. Otherwise, specification should be a list of
        num_measurements tuples, where each tuples contains the (zero-based)
        indices of the components that measurement contains.

        If specification is given, the reference band must contain all components.
        '''
        if specification is None:
            self.measurement_component_mapping = np.ones((num_components, self.num_measurements))
            self.redshifts_exchangeable = True
        else:
            measurement_component_mapping = np.zeros((num_components, self.num_measurements))
            for m in xrange(self.num_measurements):
                measurement_component_mapping[specification[m], m] = 1.

            if np.all(measurement_component_mapping[:, self.config.ref_band] == 1.):
                #Set the mapping
                self.measurement_component_mapping = measurement_component_mapping
                #Set whether the redshifts are exchangable and so need sorting condition
                #Only need to check if there's more than one component
                if num_components > 1:
                    self.redshifts_exchangeable = np.all(measurement_component_mapping[1:, :] ==
                                                         measurement_component_mapping[:-1, :])
                else:
                    self.redshifts_exchangeable = None

            else:
                #TODO: Currently enforcing the ref band to have all components. This is needed
                # to be able to specifiy the fractions (IS IT??). Also ref band is currently used in the priors,
                # though the magnitudes going to the priors either have to be in the reference band
                # *OR* on their own, in which case no separation in necessary (like original BPZ case)
                raise ValueError('The reference band must contain all components.')

    def generateBlendMagnitude(self, num_components, redshifts, scales, template_indices, err_frac):
        true_flux = np.zeros(self.responses.filters.num_filters)
        for c in xrange(num_components):
            true_flux += self.responses(template_indices[c], None, redshifts[c]) * scales[c] * self.measurement_component_mapping[c, :]
        fracs = np.zeros(num_components)
        for c in xrange(num_components):
            fracs[c] = ((self.responses(template_indices[c], None, redshifts[c]) * scales[c]) / true_flux)[self.config.ref_band]
        rand_err  = (np.random.rand(self.responses.filters.num_filters) * (true_flux * err_frac * 2)) - (true_flux * err_frac)
        obs_flux = true_flux + rand_err
        flux_err = true_flux * err_frac
        obs_mag = np.log10(obs_flux) / (-0.4)
        mag_err = np.log10((flux_err/obs_flux)+1.) / (-0.4)
        return obs_mag, mag_err, fracs

    def drawBlendFromPrior(self, num_components, max_redshift, max_scale, max_err_frac, magnitude_bounds=[20., 32.]):
        '''
        Use the priors defined in the Model to draw a random blend of components.

        We don't use the magnitudes we sample to generate the photometric data
        but the true magnitudes we generate should match the ones drawn here - consistency test
        '''
        magnitudes = np.zeros(num_components)
        templates = np.zeros(num_components, dtype=int)
        redshifts = np.zeros(num_components)
        true_ref_response = np.zeros(num_components)
        for c in xrange(num_components):
            #Sample magnitude
            rj = Rejection(lambda m: np.exp(self.model.lnMagnitudePrior(m)), magnitude_bounds)
            magnitudes[c] = rj.sample(1, self.seed.next())
            #Sample template given magnitude
            rstate = np.random.RandomState(self.seed.next())
            tmp_prior = np.zeros(self.responses.templates.num_templates)
            for t in xrange(self.responses.templates.num_templates):
                tmp_type_t = self.responses.templates.template_type(t)
                tmp_prior[t] = np.exp(self.model.lnTemplatePrior(tmp_type_t, magnitudes[c]))
            templates[c] = rstate.choice(self.responses.templates.num_templates, p=tmp_prior)
            tmp_type = self.responses.templates.template_type(templates[c])
            #Sample redshift, given template and magnitude
            #Include redshift (c+1)-point-correlation if other redshifts drawn
            if c==0:
                fn = lambda z: np.exp(self.model.lnRedshiftPrior(z, tmpType, magnitudes[c]))
            else:
                fn = lambda z: np.exp(self.model.lnRedshiftPrior(z, tmpType, magnitudes[c])) *
                        (1. + self.model.correlationFunction(np.append(redshifts[:c], z)))
            rj = Rejection(fn)
            redshifts[c] = rj.sample(1, self.seed.next())
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
        order = np.argsort(redshifts)
        component_normalisation = component_normalisation[order]
        templates = templates[order]
        redshifts = redshifts[order]
        return redshifts, component_normalisation, templates



    def randomBlend(self, num_components, max_redshift, max_scale, max_err_frac, magnitude_bounds=[20., 32.]):
        np.random.seed(self.seed.next())
        sim_err_frac = self.rstate.rand() * max_err_frac
        sim_redshift, sim_scale, sim_template = drawBlendFromPrior(num_components, max_redshift, max_scale, max_err_frac, magnitude_bounds=magnitude_bounds):

        obs_mag, mag_err, fracs = self.generateBlendMagnitude(num_components, sim_redshift, sim_scale, sim_template, sim_err_frac)

        truth = {}
        truth['num_components'] = num_components
        for c in xrange(num_components):
            truth[c] = {'redshift': sim_redshift[c], 'scale': sim_scale[c],
            'template': sim_template[c], 'fraction': fracs[c]}

        return obs_mag, mag_err, truth

    def simulateRandomGalaxies(self, num_sims, num_components, max_redshift, max_scale, max_err_frac, measurement_component_specification=None, magnitude_bounds=[20., 32.]):
        self.setMeasurementComponentMapping(measurement_component_specification, num_components)
        for g in xrange(num_sims):
            mag_data, mag_sigma, truth = self.randomBlend(num_components, max_redshift, max_scale, max_err_frac, magnitude_bounds=magnitude_bounds)
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
