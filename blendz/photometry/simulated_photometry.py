from builtins import *
import warnings
from math import ceil
from itertools import repeat
import numpy as np
import emcee
from tqdm import tqdm
from blendz.config import Configuration
from blendz.photometry import PhotometryBase, Galaxy
from blendz.model import BPZ
from blendz.utilities import incrementCount

class SimulatedPhotometry(PhotometryBase):
    def __init__(self, num_sims, config=None, num_components=1, max_redshift=None,
                model=None, seed=None, signal_to_noise=10.,
                num_walkers=100, burn_len=10000,
                measurement_component_specification=None, magnitude_bounds=None, **kwargs):
        super(SimulatedPhotometry, self).__init__(config=config, **kwargs)

        if model is not None:
            #Set model, take default+kwargs+model config and warn user if config also provided
            self.config = Configuration(**kwargs)
            self.config.mergeFromOther(model.config)
            self.model = model
            self.responses = self.model.responses

            if config is not None:
                warnings.warn('A configuration object was provided to '
                              + 'SimulatedPhotometry as well as a Responses '
                              + 'object, though these should be mutually exclusive. '
                              + 'The configuration provided will be ignored.')
        else:
            #Config given, take default+kwargs+given config
            #which is loaded in in PhotometryBase
            self.model = BPZ(config=self.config, max_ref_mag_hi=self.config.ref_mag_hi)
            self.responses = self.model.responses

        #Config must have magnitude_limit set, NOT magnitude_limit_col for simulations
        # We assume a fixed magnitude cut.
        if self.config.magnitude_limit is None:
            raise ValueError('SimulatedPhotometry requires magnitude_limit to be set in config.')
        elif self.config.magnitude_limit_col is not None:
            warnings.warn('SimulatedPhotometry uses magnitude_limit, but magnitude_limit_col has also been set.')

        #Config must have ref_mag_hi set, NOT ref_mag_hi_sigma for simulations
        if self.config.ref_mag_hi is None:
            raise ValueError('SimulatedPhotometry requires ref_mag_hi to be set in config.')
        elif self.config.ref_mag_hi_sigma is not None:
            warnings.warn('SimulatedPhotometry uses ref_mag_hi, but ref_mag_hi_sigma has also been set.')

        self.num_sims = num_sims
        self.num_components = num_components
        self.num_measurements = self.responses.filters.num_filters

        if max_redshift is None:
            self.max_redshift = self.config.z_hi
        else:
            self.max_redshift = max_redshift

        if magnitude_bounds is None:
            self.magnitude_bounds = [self.config.ref_mag_lo, self.config.ref_mag_hi]
        else:
            self.magnitude_bounds = magnitude_bounds

        self.num_walkers = num_walkers
        self.burn_len = burn_len

        self.zero_point_errors = self.config.zero_point_errors
        self.zero_point_frac = 10.**(0.4*self.zero_point_errors) - 1.

        #Signal to noise is an array with an element for each filters
        #If the arg is a float, assume every element the same
        if isinstance(signal_to_noise, float):
            self.signal_to_noise = np.ones(self.model.responses.filters.num_filters) * signal_to_noise
        else:
            self.signal_to_noise = signal_to_noise

        #Use an incrementing counter for the seed to make sure its always
        #different between various different function calls
        if seed is None:
            self.sim_seed = incrementCount(np.random.randint(1e9))
        else:
            self.sim_seed = incrementCount(seed)

        self.simulateRandomGalaxies(self.num_components, self.num_sims,
                                    max_redshift=self.max_redshift,
                                    measurement_component_specification=measurement_component_specification,
                                    magnitude_bounds=self.magnitude_bounds)

        #Remove simulated galaxies that aren't within the selection
        in_selection = np.where(np.array([self._fluxDataWithinSelection(g.flux_data)
                                          for g in self.all_galaxies]))[0]
        self.all_galaxies = [g for i, g in enumerate(self.all_galaxies)
                             if i in in_selection]
        #Go back through and set the galaxy indices to the correct value
        for i, gal in enumerate(self.all_galaxies):
            gal.index = i

    def drawParametersFromPrior(self, num_components, num_sims, burn_len=10000, num_walkers=100, num_thin=50):
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
        burn_rstate = np.random.RandomState(next(self.sim_seed))
        burn_pos, burn_prob, burn_state = sampler.run_mcmc(start_pos,
                                                           burn_sample_len,
                                                           rstate0=burn_rstate)
        #Sample our parameters until we have enough selected sources
        num_selected = 0
        #Sample in batches of num_sims samples, plus a bit more
        #because some are lost due to not making the selection criteria
        main_sample_len = int(ceil(num_sims / float(num_walkers)) * 1.2 * num_thin)
        params = np.zeros((num_sims, num_pars))
        #Fill up the return params array with samples from the chain
        #that obey the magnitude selection criteria
        with tqdm(total=num_sims) as pbar:
            while num_selected < num_sims:
                sampler.reset()
                main_rstate = np.random.RandomState(next(self.sim_seed))
                sampler.run_mcmc(burn_pos, main_sample_len,
                                 thin=num_thin, rstate0=main_rstate)
                chain = sampler.flatchain
                for i in range(np.shape(chain)[0]):
                    sample_i = chain[-i, :]
                    if num_selected<num_sims:# and self._fluxDataWithinSelection(sample_i):
                        params[num_selected, :] = sample_i
                        params[num_selected, num_components:2*num_components] = \
                            np.around(params[num_selected, num_components:2*num_components])
                        num_selected += 1
                        pbar.update()


        return params

    def _fluxDataWithinSelection(self, flux_data):
        '''Take in an array of flux data and return a bool of whether
        is is within the selection criteria
        '''
        ref_band_mag = np.log10(flux_data[self.config.ref_band]) / (-0.4)
        return ref_band_mag <= self.config.magnitude_limit

    def generateObservables(self, params):
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
                resp_c = self.responses.interp(zc)[tc, :]
                norm = (10.**(-0.4 * mc)) / resp_c[self.config.ref_band]
                true_flux[g, :] += resp_c * norm * self.model.measurement_component_mapping[c, :]

        #Errors
        flux_err = true_flux / self.signal_to_noise #quoted error
        err_rstate = np.random.RandomState(next(self.sim_seed))
        rand_err = err_rstate.normal(loc=0., scale=flux_err, size=out_shape) #actual error
        obs_flux = true_flux + rand_err #added actual error to flux

        #Conversions
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
                               measurement_component_specification=None,
                               magnitude_bounds=None, burn_len=None, num_walkers=None):
        if max_redshift is None:
            max_redshift = self.max_redshift
        if magnitude_bounds is None:
            magnitude_bounds = self.magnitude_bounds
        if burn_len is None:
            burn_len = self.burn_len
        if num_walkers is None:
            num_walkers = self.num_walkers

        self.model._setMeasurementComponentMapping(measurement_component_specification, num_components)

        prior_parameters = self.drawParametersFromPrior(num_components, num_sims, burn_len=burn_len, num_walkers = num_walkers)
        all_mag_data, all_mag_sigma = self.generateObservables(prior_parameters)
        all_truths = self.createTruthDicts(prior_parameters)

        for g in range(num_sims):
            new_galaxy = Galaxy(all_mag_data[g, :], all_mag_sigma[g, :], self.config, self.zero_point_frac, g)
            new_galaxy.truth = all_truths[g]
            new_galaxy.magnitude_limit = self.config.magnitude_limit
            new_galaxy.ref_mag_hi = self.config.ref_mag_hi
            self.all_galaxies.append(new_galaxy)
