from builtins import *
import warnings
from math import ceil
from itertools import repeat
import numpy as np
import emcee
from blendz.config import Configuration
from blendz.photometry import PhotometryBase, Galaxy
from blendz.model import BPZ
from blendz.utilities import incrementCount

class SimulatedPhotometry(PhotometryBase):
    def __init__(self, num_sims, config=None, num_components=1, max_redshift=None,
                max_err_frac=0.1, model=None, seed=None, random_err=True,
                measurement_component_specification=None, magnitude_bounds=[20., 32], **kwargs):
        super(SimulatedPhotometry, self).__init__(config=config, **kwargs)

        if model is not None:
            #Set model, take default+kwargs+model config and warn user if config also provided
            self.config = Configuration(**kwargs)
            self.config.mergeFromOther(model.config)
            self.model = model
            self.responses = self.model.responses

            if config is not None:
                warnings.warn("""A configuration object was provided to
                                SimulatedPhotometry as well as a Responses
                                object, though these should be mutually exclusive.
                                The configuration provided will be ignored.""")
        else:
            #Config given, take default+kwargs+given config
            #which is loaded in in PhotometryBase
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

        self.simulateRandomGalaxies(self.num_components, self.num_sims,
                                    max_redshift=self.max_redshift,
                                    max_err_frac=self.max_err_frac,
                                    measurement_component_specification=measurement_component_specification,
                                    magnitude_bounds=self.magnitude_bounds)

    def drawParametersFromPrior(self, num_components, num_sims, burn_len=10000, num_walkers = 100):
        '''
        Use mcmc to draw ``num_sims`` sets of source parameters for
        sources of ``num_components`` components.

        Parameters are returned as an array shape (num_sims, 3*num_components)
        where the second axis is ordered [z1, z2... t1, t2... m0_1, m0_2...].
        Because it's a single array, the template indices are floats, not ints,
        though they have been rounded.
        '''
        #Prior boundary parameters
        z_range = self.config.z_hi - self.config.z_lo
        z_shift = self.config.z_lo
        mag_range = self.config.ref_mag_hi - self.config.ref_mag_lo
        mag_shift = self.config.ref_mag_lo
        tmp_range = self.responses.templates.num_templates - 1.
        tmp_shift = 0.
        #Set up emcee sampler
        num_pars = int(3 * num_components)

        all_ranges = [z for z in repeat(z_range, num_components)] + \
                     [t for t in repeat(tmp_range, num_components)] + \
                     [m for m in repeat(mag_range, num_components)]

        all_shifts = [z for z in repeat(z_shift, num_components)] + \
                     [t for t in repeat(tmp_shift, num_components)] + \
                     [m for m in repeat(mag_shift, num_components)]

        start_pos = [(np.random.random(num_pars) * np.array(all_ranges)) + \
                    np.array(all_shifts) for i in range(num_walkers)]

        for w in range(num_walkers):
            if self.config.sort_redshifts:
                #Force sorted redshifts
                start_pos[w][:num_components] = np.sort(start_pos[w][:num_components])
            else:
                #Force sorted magnitudes
                start_pos[w][-num_components:] = np.sort(start_pos[w][-num_components:])

        sampler = emcee.EnsembleSampler(num_walkers, num_pars, self.model._lnTotalPrior)
        #Run burn in
        burn_sample_len = int(ceil(burn_len / float(num_walkers)))
        burn_pos, burn_prob, burn_state = sampler.run_mcmc(start_pos, burn_sample_len)
        #Sample our parameters until we have enough selected sources
        num_selected = 0
        #Sample in batches of num_sims samples, plus a bit more
        #because some are lost due to not making the selection criteria
        main_sample_len = int(ceil(num_sims / float(num_walkers))*1.2)
        params = np.zeros((num_sims, num_pars))
        #Fill up the return params array with samples from the chain
        #that obey the magnitude selection criteria
        while num_selected < num_sims:
            sampler.reset()
            sampler.run_mcmc(burn_pos, main_sample_len)
            chain = sampler.flatchain
            for i in range(np.shape(chain)[0]):
                sample_i = chain[-i, :]
                if num_selected<num_sims and self._fluxDataWithinSelection(sample_i):
                    params[num_selected, :] = sample_i
                    params[num_selected, num_components:2*num_components] = \
                        np.around(params[num_selected, num_components:2*num_components])
                    num_selected += 1

        return params

    def generateObservables(self, params, max_err_frac, min_err_frac=0.):
        '''
        Use array of params shape (num_sims, 3*num_components) to generate
        array of observed fluxes and array of errors, both of shape
        (num_sims, num_filters)
        '''
        num_sims, num_params = np.shape(params)
        num_components = num_params // 3
        out_shape = (num_sims, self.responses.filters.num_filters)
        true_flux = np.zeros(out_shape)
        for g in range(num_sims):
            for c in range(num_components):
                zc = params[g, c]
                tc = int(params[g, num_components + c])
                mc = params[g, (2*num_components) + c]
                resp_c = self.responses(tc, None, zc)
                norm = (10.**(-0.4 * mc)) / resp_c[self.config.ref_band]
                true_flux[g, :] += resp_c * norm * self.model.measurement_component_mapping[c, :]
        err_frac_range = max_err_frac - min_err_frac
        rand_err_frac = (np.random.rand(*out_shape) * err_frac_range) + min_err_frac
        rand_err_sign = (np.random.randint(2, size=out_shape) * 2.) - 1.
        rand_err  = rand_err_frac * rand_err_sign * true_flux
        obs_flux = true_flux + rand_err
        flux_err = true_flux * max_err_frac

        obs_mag = np.log10(obs_flux) / (-0.4)
        mag_err = np.log10((flux_err/obs_flux)+1.) / 0.4
        return obs_mag, mag_err

    def createTruthDicts(self, params):
        '''
        Use array of params shape (num_sims, 3*num_components) to generate
        list of truth dictionaries.
        '''
        num_sims, num_params = np.shape(params)
        num_components = num_params // 3

        all_truths = []
        for g in range(num_sims):
            redshifts = params[g, :num_components]
            templates = params[g, num_components:2*num_components].astype(int)
            magnitudes = params[g, 2*num_components:]
            #Order component truths before saving, either by redshift...
            if self.config.sort_redshifts:
                order = np.argsort(redshifts)
                redshifts = redshifts[order]
                templates = templates[order]
                magnitudes = magnitudes[order]
            #...or magnitudes
            else:
                order = np.argsort(magnitudes)
                redshifts = redshifts[order]
                templates = templates[order]
                magnitudes = magnitudes[order]

            truth = {}
            truth['num_components'] = num_components
            for c in range(num_components):
                truth[c] = {'redshift': redshifts[c],
                            'template': templates[c],
                            'magnitude': magnitudes[c]}
            all_truths.append(truth)
        return all_truths

    def simulateRandomGalaxies(self, num_components, num_sims, max_redshift=None,
                               max_err_frac=None, measurement_component_specification=None,
                               magnitude_bounds=None, burn_len=10000, num_walkers = 100,
                               min_err_frac=0.):
        if max_redshift is None:
            max_redshift = self.max_redshift
        if max_err_frac is None:
            max_err_frac = self.max_err_frac
        if magnitude_bounds is None:
            magnitude_bounds = self.magnitude_bounds

        self.model._setMeasurementComponentMapping(measurement_component_specification, num_components)

        prior_parameters = self.drawParametersFromPrior(num_components, num_sims, burn_len=burn_len, num_walkers = num_walkers)
        all_mag_data, all_mag_sigma = self.generateObservables(prior_parameters, max_err_frac, min_err_frac=min_err_frac)
        all_truths = self.createTruthDicts(prior_parameters)

        for g in range(num_sims):
            new_galaxy = Galaxy(all_mag_data[g, :], all_mag_sigma[g, :], self.config, self.zero_point_frac, g)
            new_galaxy.truth = all_truths[g]
            self.all_galaxies.append(new_galaxy)

    def simulateGalaxies(self, redshifts, scales, templates, err_frac):
        for g in range(len(redshifts)):
            num_components = len(redshifts[g])
            obs_mag, mag_err, fracs = self.generateObservables(
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
