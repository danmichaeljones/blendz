from builtins import *
import sys
import warnings
import itertools as itr
import numpy as np
from scipy.special import erf
import nestle
from tqdm import tqdm
import dill
from blendz import Configuration
from blendz.fluxes import Responses
from blendz.photometry import Photometry, SimulatedPhotometry
from blendz.model import BPZ
from blendz.utilities import incrementCount


class Photoz(object):
    def __init__(self, model=None, photometry=None, config=None,\
                 load_state_path=None, **kwargs):
        if load_state_path is not None:
            self.loadState(load_state_path)
        else:
            #Warn user is config and either/or responses/photometry given that config ignored
            if ((model is not None and config is not None) or
                    (photometry is not None and config is not None)):
                warnings.warn("""A configuration object was provided to Photoz object
                                as well as a Model/Photometry object, though these
                                should be mutually exclusive. The configuration
                                provided will be ignored.""")
            #Responses and photometry given, merge their configs
            if (model is not None) and (photometry is not None):
                self.config = Configuration(**kwargs)
                self.config.mergeFromOther(model.config)
                self.config.mergeFromOther(photometry.config)
                self.model = model
                self.responses = self.model.responses
                self.photometry = photometry
            #Only responses given, use its config to load photometry
            elif (model is not None) and (photometry is None):
                self.config = Configuration(**kwargs)
                self.config.mergeFromOther(model.config)
                self.model = model
                self.responses = self.model.responses
                self.photometry = Photometry(config=self.config)
            #Only photometry given, use its config to load responses
            elif (model is None) and (photometry is not None):
                self.config = Configuration(**kwargs)
                self.config.mergeFromOther(photometry.config)
                self.photometry = photometry
                self.model = BPZ(config=self.config)
                self.responses = self.model.responses
            #Neither given, load both from provided (or default, if None) config
            else:
                self.config = Configuration(**kwargs)
                if config is not None:
                    self.config.mergeFromOther(config)

                self.model = BPZ(config=self.config)
                self.responses = self.model.responses
                self.photometry = Photometry(config=self.config)

            self.num_templates = self.responses.templates.num_templates
            self.num_measurements = self.responses.filters.num_filters
            self.num_galaxies = self.photometry.num_galaxies

            #Default to assuming single component, present in all measurements
            self.model._setMeasurementComponentMapping(None, 1)

            #Set up empty dictionaries to put results into
            self.sample_results = {}
            self.reweighted_samples = {}
            for g in range(self.num_galaxies):
                #Each value is a dictionary which will be filled by sample function
                #The keys of this inner dictionary will be the number of blends for run
                self.sample_results[g] = {}
                self.reweighted_samples[g] = {}

    def saveState(self, filepath):
        """Save this entire Photoz instance to file.

        This saves the exact state of the current object, including all data and any
        reults from sampling.

        Args:
            filepath (str): Path to file to save to.
        """
        if isinstance(self.photometry, SimulatedPhotometry):
            try:
                current_seed = self.photometry.sim_seed.next()
                self.photometry.sim_seed = current_seed
            except:
                warnings.warn('SimulatedPhotometry seed not saved.')
        with open(filepath, 'wb') as f:
            state = {key: val for key, val in self.__dict__.items() if key!='pbar'}
            dill.dump(state, f)
        #Put the random seed back how it was after the saving is done
        if isinstance(self.photometry, SimulatedPhotometry):
            try:
                self.photometry.sim_seed = incrementCount(current_seed)
            except:
                pass

    def loadState(self, filepath):
        with open(filepath, 'rb') as f:
            self.__dict__.update(dill.load(f))
        #If the photometry is simulated, replace the seed currently saved as
        #a number with the generator it was before saving
        if isinstance(self.photometry, SimulatedPhotometry):
            try:
                current_seed = self.photometry.sim_seed
                self.photometry.sim_seed = incrementCount(current_seed)
            except:
                warnings.warn('SimulatedPhotometry seed not loaded.')

    def _lnLikelihood_flux(self, model_flux):
        chi_sq = -1. * np.sum((self.photometry.current_galaxy.flux_data_noRef - model_flux)**2 / self.photometry.current_galaxy.flux_sigma_noRef**2)
        return chi_sq

    def _lnLikelihood_mag(self, total_ref_flux):
        #chi_sq = -1. * np.sum((self.photometry.current_galaxy.ref_mag_data - total_ref_mag)**2 / self.photometry.current_galaxy.ref_mag_sigma**2)
        chi_sq = -1. * np.sum((self.photometry.current_galaxy.ref_flux_data - total_ref_flux)**2 / self.photometry.current_galaxy.ref_flux_sigma**2)
        return chi_sq

    def _lnSelection(self, flux):
        flim = 10.**(-0.4*self.config.magnitude_limit)
        sigma = self.photometry.current_galaxy.ref_flux_sigma
        selection = 0.5 - (0.5 * erf((flim - flux) / (sigma * np.sqrt(2))))
        return np.log(selection)

    def _lnPosterior(self, params):
        num_components = int(len(params) // 2)
        redshifts = params[:num_components]
        magnitudes = params[num_components:]

        if not self.model._obeyPriorConditions(redshifts, magnitudes):
            return -np.inf
        else:
            #Precalculate all quantities we'll need in the template loop
            template_priors = np.zeros((num_components, self.num_templates))
            redshift_priors = np.zeros((num_components, self.num_templates))
            #Single interp call -> Shape = (N_template, N_band, N_component)
            model_fluxes = self.responses.interp(redshifts)

            for T in range(self.num_templates):
                tmpType = self.responses.templates.templateType(T)
                for nb in range(num_components):
                    template_priors[nb, T] = self.model.lnTemplatePrior(tmpType, magnitudes[nb])
                    redshift_priors[nb, T] = self.model.lnRedshiftPrior(redshifts[nb], tmpType, magnitudes[nb])
            redshift_correlation = np.log(1. + self.model.correlationFunction(redshifts))

            #We assume independent magnitudes, so sum over log priors for joint prior
            joint_magnitude_prior = np.sum([self.model.lnMagnitudePrior(m) for m in magnitudes])
            #Get total flux in reference band  = transform to flux & sum
            total_ref_flux = np.sum(10.**(-0.4 * magnitudes))
            selection_effect = self._lnSelection(total_ref_flux)

            #Loop over all templates - discrete marginalisation
            #All log probabilities so (multiply -> add) and (add -> logaddexp)
            lnProb = -np.inf

            #At each iteration template_combo is a tuple of (T_1, T_2... T_num_components)
            for template_combo in itr.product(*itr.repeat(range(self.num_templates), num_components)):
                #One redshift prior, template prior and model flux for each blend component
                tmp = 0.
                blend_flux = np.zeros(self.num_measurements)
                component_scaling_norm = 0.
                for nb in range(num_components):
                    T = template_combo[nb]
                    component_scaling = 10.**(-0.4*magnitudes[nb]) / model_fluxes[T, self.config.ref_band, nb]
                    blend_flux += model_fluxes[T, :, nb] * component_scaling * self.model.measurement_component_mapping[nb, :]
                    tmp += template_priors[nb, T]
                    tmp += redshift_priors[nb, T]
                    #################################################print('SCALING')
                    ################################################print component_scaling
                    ################################################print('BLEND FLUX')
                    #################################################print blend_flux
                #Remove ref_band from blend_fluxes, as that goes into the magnitude
                #likelihood, not the flux likelihood
                blend_flux = blend_flux[self.config.non_ref_bands]


                #Other terms only appear once per summation-step
                tmp += redshift_correlation
                tmp += self._lnLikelihood_flux(blend_flux)
                tmp += self._lnLikelihood_mag(total_ref_flux)
                tmp += joint_magnitude_prior
                tmp += selection_effect

                #logaddexp contribution from this template to marginalise
                lnProb = np.logaddexp(lnProb, tmp)

            return lnProb

    def _priorTransform(self, params):
        '''
        Transform params from [0, 1] uniform random to [min, max] uniform random,
        where the min redshift is zero, max redshift is set in the configuration,
        and the min/max magnitudes (numerically, not brightness) are set by configuration.
        '''
        num_components = int(len(params) // 2)

        trans = np.ones(len(params))
        trans[:num_components] = self.config.z_hi - self.config.z_lo
        trans[num_components:] = self.config.ref_mag_hi - self.config.ref_mag_lo

        shift = np.zeros(len(params))
        shift[:num_components] = self.config.z_lo
        shift[num_components:] = self.config.ref_mag_lo
        return (params * trans) + shift

    def _sampleProgressUpdate(self, info):
        if info['it']%self.num_between_print==0:
            self.pbar.set_description('[Gal: {}/{}, Comp: {}/{}, Itr: {}] '.format(self.gal_count,
                                                                                   self.num_galaxies_sampling,
                                                                                   self.blend_count,
                                                                                   self.num_components_sampling,
                                                                                   info['it']))
            self.pbar.refresh()

    def sample(self, num_components, galaxy=None, resample=10000, seed=False, measurement_component_mapping=None, npoints=150, num_between_print=10):
        """Sample the posterior for a particular number of components.

        What happens to the results?

        Args:
            num_components (int):
                Sample the posterior defined for this number of components in the source.

            galaxy (int or None):
                Index of the galaxy to sample. If None, sample every galaxy in the
                photometry. Defaults to None.

            resample (int or None):
                Number of non-weighted samples to sample from the weighted samples
                distribution from Nested Sampling. Defaults to 10000.

            seed (bool or int):
                Random seed for sampling to ensure deterministic results when
                ampling again. If False, do not seed. If True, seed with value
                derived from galaxy index. If int, seed with specific value.

            measurement_component_mapping (None or list of tuples):
                If None, sample from the fully blended posterior. For a partially
                blended posterior, this should be a list of tuples (length = number of
                measurements), where each tuples contains the (zero-based) indices of
                the components that measurement contains. Defaults to None.

            npoints (int):
                Number of live points for the Nested Sampling algorithm. Defaults to 150.

        """

        if isinstance(num_components, int):
            num_components = [num_components]
        else:
            if measurement_component_mapping is not None:
                #TODO: This is a time-saving hack to avoid dealing with multiple specifications
                #The solution would probably be to rethink the overall design
                raise ValueError('measurement_component_mapping cannot be set when sampling multiple numbers of components in one call. Do the separate cases separately.')

        self.num_components_sampling = len(num_components)
        self.num_between_print = float(round(num_between_print))

        if galaxy is None:
            start = None
            stop = None
            self.num_galaxies_sampling = self.num_galaxies
        elif isinstance(galaxy, int):
            start = galaxy
            stop = galaxy + 1
            self.num_galaxies_sampling = 1
        else:
            raise TypeError('galaxy may be either None or an integer, but got {} instead'.format(type(galaxy)))

        with tqdm(total=self.num_galaxies_sampling * self.num_components_sampling) as self.pbar:
            self.gal_count = 1
            for gal in self.photometry.iterate(start, stop):
                self.blend_count = 1
                for nb in num_components:

                    if seed is False:
                        rstate = np.random.RandomState()
                    elif seed is True:
                        rstate = np.random.RandomState(gal.index)
                    else:
                        rstate = np.random.RandomState(seed + gal.index)

                    num_param = 2 * nb
                    self.model._setMeasurementComponentMapping(measurement_component_mapping, nb)
                    results = nestle.sample(self._lnPosterior, self._priorTransform,
                                            num_param, method='multi', npoints=npoints,
                                            rstate=rstate, callback=self._sampleProgressUpdate)
                    self.sample_results[gal.index][nb] = results
                    if resample is not None:
                        #self.reweighted_samples[gal.index][nb] = nestle.resample_equal(results.samples, results.weights)
                        self.reweighted_samples[gal.index][nb] = results.samples[rstate.choice(len(results.weights), size=resample, p=results.weights)]

                    self.gal_count += 1
                    self.blend_count += 1
                    self.pbar.update()
