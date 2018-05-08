try:
    from mpi4py import MPI
    MPI_RANK = MPI.COMM_WORLD.Get_rank()
except:
    MPI_RANK = 0

from builtins import *
import sys
import os
import warnings
from math import ceil
from multiprocessing import cpu_count
import itertools as itr
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erf
import nestle
import emcee
from tqdm import tqdm
import dill
import blendz
from blendz import Configuration
from blendz.fluxes import Responses
from blendz.photometry import Photometry, SimulatedPhotometry
from blendz.model import BPZ
from blendz.utilities import incrementCount, Silence

try:
    import pymultinest
    PYMULTINEST_AVAILABLE = True
except ImportError:
    PYMULTINEST_AVAILABLE = False
    warnings.warn('PyMultinest not installed, so falling back to (slower) python implementation.'
        + ' See http://johannesbuchner.github.com/PyMultiNest/install.html for installation help.')
except (SystemExit, OSError):
    PYMULTINEST_AVAILABLE = False
    warnings.warn('PyMultinest failed to load, so falling back to (slower) python implementation.'
        + ' See http://johannesbuchner.github.com/PyMultiNest/install.html for installation help.')


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

            self.tmp_ind_to_type_ind = self.responses.templates.tmp_ind_to_type_ind
            self.possible_types = self.responses.templates.possible_types
            self.num_types = len(self.possible_types)

            #Default to assuming single component, present in all measurements
            self.model._setMeasurementComponentMapping(None, 1)

            #Set up empty dictionaries to put results into
            self._samples = {}
            self._logevd = {}
            self._logevd_error = {}
            for g in range(self.num_galaxies):
                #Each value is a dictionary which will be filled by sample function
                #The keys of this inner dictionary will be the number of blends for run
                self._samples[g] = {}
                self._logevd[g] = {}
                self._logevd_error[g] = {}

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
        flim = 10.**(-0.4*self.photometry.current_galaxy.magnitude_limit)
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
            #Single interp call -> Shape = (N_template, N_band, N_component)
            model_fluxes = self.responses.interp(redshifts)

            priors = np.zeros((num_components, self.num_types))
            for nb in range(num_components):
                priors[nb, :] = self.model.lnPrior(redshifts[nb], magnitudes[nb])

            redshift_correlation = np.log(1. + self.model.correlationFunction(redshifts))

            #Get total flux in reference band  = transform to flux & sum
            total_ref_flux = np.sum(10.**(-0.4 * magnitudes))
            selection_effect = self._lnSelection(total_ref_flux)

            #Loop over all templates - discrete marginalisation
            #All log probabilities so (multiply -> add) and (add -> logaddexp)
            lnProb = -np.inf

            #At each iteration template_combo is a tuple of (T_1, T_2... T_num_components)
            for template_combo in itr.product(*itr.repeat(range(self.num_templates), num_components)):
                #One redshift/template/magnitude prior and model flux for each blend component
                tmp = 0.
                blend_flux = np.zeros(self.num_measurements)
                component_scaling_norm = 0.
                for nb in range(num_components):
                    T = template_combo[nb]
                    component_scaling = 10.**(-0.4*magnitudes[nb]) / model_fluxes[T, self.config.ref_band, nb]
                    blend_flux += model_fluxes[T, :, nb] * component_scaling * self.model.measurement_component_mapping[nb, :]
                    type_ind = self.tmp_ind_to_type_ind[T]
                    tmp += priors[nb, type_ind]
                #Remove ref_band from blend_fluxes, as that goes into the magnitude
                #likelihood, not the flux likelihood
                blend_flux = blend_flux[self.config.non_ref_bands]

                #Other terms only appear once per summation-step
                tmp += redshift_correlation
                tmp += self._lnLikelihood_flux(blend_flux)
                tmp += self._lnLikelihood_mag(total_ref_flux)
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


    def _priorTransform_multinest(self, cube, ndim, nparams):
        '''
        Transform params from [0, 1] uniform random to [0, max] uniform random,
        where the max redshift is set in the configuration, and the max fraction
        is 1.
        '''
        num_components = ndim // 2

        trans = np.zeros(ndim)
        trans[:num_components] = self.config.z_hi
        trans[num_components:] = self.config.ref_mag_hi - self.config.ref_mag_lo

        shift = np.zeros(ndim)
        shift[:num_components] = self.config.z_lo
        shift[num_components:] = self.config.ref_mag_lo

        for i in range(ndim):
            cube[i] = (cube[i] * trans[i]) + shift[i]

    def _lnPosterior_multinest(self, cube, ndim, nparams):
        self.num_posterior_evals += 1

        params = np.array([cube[i] for i in range(ndim)])

        with self.breakSilence():
            if (self.num_posterior_evals%self.num_between_print==0) and MPI_RANK==0:
                self.pbar.set_description('[Gal: {}/{}, Comp: {}/{}, Itr: {}] '.format(self.gal_count,
                                                                                       self.num_galaxies_sampling,
                                                                                       self.blend_count,
                                                                                       self.num_components_sampling,
                                                                                       self.num_posterior_evals))
                self.pbar.refresh()

        return self._lnPosterior(params)

    def _sampleProgressUpdate(self, info):
        if (info['it']%self.num_between_print==0) and MPI_RANK==0:
            self.pbar.set_description('[Gal: {}/{}, Comp: {}/{}, Itr: {}] '.format(self.gal_count,
                                                                                   self.num_galaxies_sampling,
                                                                                   self.blend_count,
                                                                                   self.num_components_sampling,
                                                                                   info['it']))
            self.pbar.refresh()

    def sample(self, num_components, galaxy=None, nresample=1000, seed=False,
               measurement_component_mapping=None, npoints=150, print_interval=10,
               use_pymultinest=None, save_path=None, save_interval=None):
        """Sample the posterior for a particular number of components.

        Args:
            num_components (int):
                Sample the posterior defined for this number of components in the source.

            galaxy (int or None):
                Index of the galaxy to sample. If None, sample every galaxy in the
                photometry. Defaults to None.

            nresample (int):
                Number of non-weighted samples to draw from the weighted samples
                distribution from Nested Sampling. Defaults to 1000.

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

            print_interval (int):
                Update the progress bar with number of posterior evaluations every
                print_interval calls. Defaults to 10.

            save_path (None or str):
                Filepath for saving the Photoz object for reloading with `Photoz.loadState`.
                If None, do not automatically save. If given, the Photoz object will
                be saved to this path after all galaxies are sampled. If save_interval
                is also not None, the Photoz object will be saved to this path every
                save_interval galaxies. Defaults to None.

            save_interval (None or int)
                If given and save_path is not None, the Photoz object will be
                saved to save_path every save_interval galaxies. Defaults to None.

            use_pymultinest (bool or None)
                If True, sample using the pyMultinest sampler. This requires PyMultiNest
                to be installed separately. If False, sample using the Nestle sampler,
                which is always installed when blendz is. If None, check whether pyMultinest
                is installed and use it if it is, otherwise use Nestle. Defaults to None.
        """

        if use_pymultinest is None:
            use_pymultinest = PYMULTINEST_AVAILABLE

        if isinstance(num_components, int):
            num_components = [num_components]
        else:
            if measurement_component_mapping is not None:
                #TODO: This is a time-saving hack to avoid dealing with multiple specifications
                #The solution would probably be to rethink the overall design
                raise ValueError('measurement_component_mapping cannot be set when sampling multiple numbers of components in one call. Do the separate cases separately.')

        self.num_components_sampling = len(num_components)
        self.num_between_print = float(round(print_interval))

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

        with tqdm(total=self.num_galaxies_sampling * self.num_components_sampling, unit='galaxy') as self.pbar:
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

                    if use_pymultinest:
                        if not os.path.exists('chains'):
                            os.makedirs('chains')
                        with Silence() as self.breakSilence:
                            self.num_posterior_evals = 0
                            pymultinest.run(self._lnPosterior_multinest, self._priorTransform_multinest,
                                            num_param, resume=False, verbose=False, sampling_efficiency='model',
                                            n_live_points=npoints)#,
                                            #outputfiles_basename=os.path.join(blendz.CHAIN_PATH, 'chain_'))
                            results = pymultinest.analyse.Analyzer(num_param)#, outputfiles_basename=os.path.join(blendz.CHAIN_PATH, 'chain_'))

                        self._samples[gal.index][nb] = results.get_equal_weighted_posterior()[:, :-1]
                        self._logevd[gal.index][nb] = results.get_mode_stats()['global evidence']
                        self._logevd_error[gal.index][nb] = results.get_mode_stats()['global evidence error']

                    else:
                        results = nestle.sample(self._lnPosterior, self._priorTransform,
                                                num_param, method='multi', npoints=npoints,
                                                rstate=rstate, callback=self._sampleProgressUpdate)
                        self._samples[gal.index][nb] = results.samples[rstate.choice(len(results.weights), size=nresample, p=results.weights)]
                        self._logevd[gal.index][nb] = results.logz
                        self._logevd_error[gal.index][nb] = results.logzerr

                    self.gal_count += 1
                    self.blend_count += 1
                    if MPI_RANK==0:
                        self.pbar.update()
                if (save_path is not None) and (save_interval is not None):
                    if gal.index % save_interval == 0:
                        self.saveState(save_path)
        if save_path is not None:
            self.saveState(save_path)


    def _lnPriorCalibrationPosterior(self, params):
        calibration_model = self.CalibrationModel(responses=self.responses, prior_params=params, **self.calibration_model_kwargs)
        calibration_prior = calibration_model.lnPriorCalibrationPrior()
        if not np.isfinite(calibration_prior):
            return -np.inf
        else:
            lnProb_all = 0.
            for g in self.photometry:
                total_ref_mag = g.ref_mag_data
                magnitude_prior = calibration_model.lnMagnitudePrior(total_ref_mag)
                if not np.isfinite(magnitude_prior):
                    pass
                else:
                    total_ref_flux = 10.**(-0.4 * total_ref_mag)
                    selection_effect = self._lnSelection(total_ref_flux)
                    template_priors = np.zeros(self.num_templates)
                    redshift_priors = np.zeros(self.num_templates)
                    lnProb_g = -np.inf

                    cache_lnTemplatePrior = {}
                    cache_lnRedshiftPrior = {}
                    for tmpType in self.responses.templates.possible_types:
                        cache_lnTemplatePrior[tmpType] = calibration_model.lnTemplatePrior(tmpType, total_ref_mag)
                        cache_lnRedshiftPrior[tmpType] = calibration_model.lnRedshiftPrior(g.truth[0]['redshift'], tmpType, total_ref_mag)

                    #Sum over template
                    for T in range(self.num_templates):
                        tmp = 0.

                        tmpType = self.responses.templates.templateType(T)
                        tmp += cache_lnTemplatePrior[tmpType]
                        tmp += cache_lnRedshiftPrior[tmpType]
                        tmp += self.fixed_lnLikelihood_flux[g.index, T]
                        tmp += magnitude_prior
                        tmp += selection_effect

                        lnProb_g = np.logaddexp(lnProb_g, tmp)
                    lnProb_all += lnProb_g
            if not np.isfinite(lnProb_all):
                return -np.inf
            else:
                return lnProb_all + calibration_prior

    def calibrate(self, num_samples, chain_path='chains/calibration-chain',
                  calibration_model=BPZ, num_walkers=200, num_threads=1,
                  #prior_params_scale=[1., 1., 1., 1., 5., 5., 5., 1., 1., 1., 0.25, 0.25, 0.25],
                  prior_params_start=np.array([0., 0., 0.33, 0.33, 1.25, 1.25, 1.25, 0.1, 0.1, 0.1, 0., 0., 0.]),
                  prior_params_spread=np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
                  **calibration_model_kwargs):
        self.CalibrationModel = calibration_model
        self.calibration_model_kwargs = calibration_model_kwargs
        self.model._setMeasurementComponentMapping(None, 1)

        chain_path = chain_path + '-{}.txt'.format(MPI_RANK)

        num_dims = len(prior_params_start)

        chain_dir_path = os.path.dirname(chain_path)
        if not os.path.exists(chain_dir_path):
            os.makedirs(chain_dir_path)
        chain_file = open(chain_path, 'w')
        chain_file.close()

        #Single interp call -> Shape = (N_template, N_band, N_component)
        self.fixed_model_fluxes = {}
        self.fixed_lnLikelihood_flux = np.zeros((self.photometry.num_galaxies, self.num_templates))
        for g in self.photometry:
            self.fixed_model_fluxes[g.index] = self.responses.interp(np.array([g.truth[0]['redshift']]))
            for T in range(self.num_templates):
                scaling = 10.**(-0.4*g.ref_mag_data) / self.fixed_model_fluxes[g.index][T, self.config.ref_band, 0]
                self.fixed_model_fluxes[g.index][T, :, 0] *= scaling
                #Cache the flux likelihoods
                non_ref_flux = self.fixed_model_fluxes[g.index][T, self.config.non_ref_bands, 0]
                self.fixed_lnLikelihood_flux[g.index, T] = self._lnLikelihood_flux(non_ref_flux)


        pos0 = np.zeros((num_walkers, num_dims))
        with tqdm(total=num_walkers) as pbar:
            for w in range(num_walkers):
                done = False
                while not done:
                    #rand_pars = np.random.random(num_dims) * prior_params_scale
                    rand_pars = (np.random.random(num_dims) * prior_params_spread * 2.) - prior_params_spread
                    rand_pars += prior_params_start
                    if np.isfinite(self._lnPriorCalibrationPosterior(rand_pars)):
                        pos0[w, :] = rand_pars
                        done = True
                pbar.update()

        # Plan is to use num_threads=cpu_count(), but can't pickle at the moment.
        # Possibly because of interp1d having __slots__, see https://stackoverflow.com/a/37726646
        # Instead, just catch it here for now.
        if num_threads!=1:
            raise ValueError('Multithreading is currently broken, so num_threads must be 1.')
        sampler = emcee.EnsembleSampler(num_walkers, num_dims,
                                        self._lnPriorCalibrationPosterior,
                                        threads=num_threads)

        num_iter = int(ceil(num_samples / float(num_walkers)))
        with tqdm(total=num_iter) as pbar:
            for i, result in enumerate(sampler.sample(pos0, iterations=num_iter, storechain=False)):
                #Append new step to file
                position = result[0]
                with open(chain_path, 'a') as chain_file:
                    for k in range(position.shape[0]):
                        chain_file.write(u"{1:s}\n".format(k, " ".join(map(str, position[k]))))
                pbar.update()

    def samples(self, num_components, galaxy=None):
        """Return the (unweighted) posterior samples.

        Args:
            num_components (int):
                Number of components.

            galaxy (int or None):
                Index of the galaxy to calculate log-evidence for. If None, return array
                of log-evidence for every galaxy. Defaults to None.
        """
        if galaxy is None:
            return [self._samples[g][num_components] for g in range(self.num_galaxies)]
        else:
            return self._samples[galaxy][num_components]

    def logevd(self, num_components, galaxy=None, return_error=False):
        """Return the base-10 log of the evidence.

        Args:
            num_components (int):
                Number of components.

            galaxy (int or None):
                Index of the galaxy to calculate log-evidence for. If None, return array
                of log-evidence for every galaxy. Defaults to None.

            return_error (bool):
                If True, also return the error on the log-evidence. If galaxy is None, this
                is also an array. Defaults to False.
        """
        if galaxy is None:
            if return_error:
                return np.array([self._logevd[g][num_components] for g in range(self.num_galaxies)]),\
                            np.array([self._logevd_error[g][num_components] for g in range(self.num_galaxies)])
            else:
                return np.array([self._logevd[g][num_components] for g in range(self.num_galaxies)])
        else:
            if return_error:
                return self._logevd[galaxy][num_components], self._logevd_error[galaxy][num_components]
            else:
                return self._logevd[galaxy][num_components]

    def logbayes(self, m, n, galaxy=None):
        """Return the base-10 log of the Bayes factor between m and n components, log[B_mn].

        A positive value suggests that that evidence prefers the m-component model over
        the n-component model.

        Args:
            m (int):
                First number of components.

            n (int):
                Second number of components.

            galaxy (int or None):
                Index of the galaxy to calculate B_mn for. If None, return array
                of B_mn for every galaxy. Defaults to None.
        """
        return (self.logevd(m, galaxy=galaxy) - self.logevd(n, galaxy=galaxy)) / np.log(10.)

    def applyToMarginals(self, func, num_components, galaxy=None, **kwargs):
        """Apply a function to the 1D marginal distribution samples of each parameter.

        Args:

            func (function):
                The function to apply to the marginal distribution samples.
                It should accept an array of the samples as its first argument,
                with optional keyword arguments.

            num_components (int):
                Number of components.

            galaxy (int or None):
                Index of the galaxy to apply the function to. If None, return
                array with a row for each galaxy. Defaults to None.

            **kwargs:
                Any optional keyword arguments to pass to the function.
        """
        if galaxy is None:
            out = np.zeros((self.num_galaxies, num_components * 2))
            for g in range(self.num_galaxies):
                for n in range(num_components * 2):
                    out[g, n] = func(self._samples[g][num_components][:, n], **kwargs)
        else:
            out = np.zeros(num_components * 2)
            for n in range(num_components * 2):
                out[n] = func(self._samples[galaxy][num_components][:, n], **kwargs)
        return out

    def _MAP1d(self, samps, bins=50):
        vals, edges = np.histogram(samps, bins=bins)
        return edges[np.argmax(vals)] + ((edges[1] - edges[0]) / 2.)

    def max(self, num_components, galaxy=None, bins=20):
        """Return the maximum-a-posteriori point for the 1D marginal distribution of each parameter.

        This is calculated by forming a 1D histogram of each marginal distribution
        and assigning the MAP of that parameter as the centre of the tallest bin.

        Args:
            num_components (int):
                Number of components.

            galaxy (int or None):
                Index of the galaxy to calculate the MAP for. If None, return array
                with rows of MAPs for each galaxy. Defaults to None.

            bins (int):
                Number of bins to use for each 1D histogram.
        """
        return self.applyToMarginals(self._MAP1d, num_components, galaxy=galaxy, bins=bins)

    def mean(self, num_components, galaxy=None):
        """Return the mean point for the 1D marginal distribution of each parameter.

        Args:
            num_components (int):
                Number of components.

            galaxy (int or None):
                Index of the galaxy to calculate the MAP for. If None, return array
                with rows of means for each galaxy. Defaults to None.
        """
        return self.applyToMarginals(np.mean, num_components, galaxy=galaxy)


    def std(self, num_components, galaxy=None):
        """Return the standard deviation for the 1D marginal distribution of each parameter.

        Args:
            num_components (int):
                Number of components.

            galaxy (int or None):
                Index of the galaxy to calculate the MAP for. If None, return array
                with rows of means for each galaxy. Defaults to None.
        """
        return self.applyToMarginals(np.std, num_components, galaxy=galaxy)

    def quantiles(self, num_components, galaxy=None, q=(0.16, 0.84)):
        try:
            pcnt = [qq*100. for qq in q]
        except TypeError:
            pcnt = [100. * q]
        #call numpy.percentile with q *= 100
        return [self.applyToMarginals(np.percentile, num_components, galaxy=galaxy, q=pp) for pp in pcnt]

    def plotRedshiftVsTrue(self, plot_equal_line=True, line_color='k',
                           point_estimate='max1d', errorbar='quantiles',
                           color=['b','g','r','c','m','y'], marker=['o'], plot_components=True,
                           rel_error_line=0.15, abs_err_line=1., legend=0, figure=None, axes=None,
                           xlabel='Spectroscopic redshift', ylabel='Photometric redshift',
                           heatmap=False, hist_bins=25, hist_cmap='viridis', colorbar=True, hist_normed=False,
                           xlim=None, ylim=None, **fig_kwargs):
        if (figure is None) and (axes is None):
            fig, ax = plt.subplots(nrows=1, ncols=1, **fig_kwargs)
            return_figure = True
        elif axes is None:
            fig = figure
            ax = fig.axes
            return_figure = False
        else:
            ax = axes
            return_figure = False

        if xlim is None:
            xlim=(self.config.z_lo, self.config.z_hi)
        if ylim is None:
            ylim=(self.config.z_lo, self.config.z_hi)
        #Line extends far beyond z_hi for error line plotting
        line = np.linspace(self.config.z_lo, self.config.z_hi * 10., 100)

        #Plotting code later expects iterables but user can specify single string instead
        if type(color) == str:
            color = [color]
        if type(marker) == str:
            marker = [marker]

        errors = {}
        specz = {}
        photoz = {}
        ##max_num_components = 0
        num_components_seen = []

        #Fill the arrays for plotting
        for g in range(self.photometry.num_galaxies):
            try:
                gal_num_components = self.photometry[g].truth['num_components']
                gal_true_redshifts = [self.photometry[g].truth[cmp]['redshift']
                                        for cmp in range(gal_num_components)]

                #Need to create new arrays for up-to this number of components
                #NaN so that galaxies with less components don't plot
                #The -1 is for indexing in for loop later
                for cmp in range(gal_num_components):
                    if cmp not in num_components_seen:
                        errors[cmp] = np.zeros((2, self.photometry.num_galaxies)) * np.nan
                        specz[cmp] = np.zeros(self.photometry.num_galaxies) * np.nan
                        photoz[cmp] = np.zeros(self.photometry.num_galaxies) * np.nan
                        num_components_seen.append(cmp)

                for cmp in range(gal_num_components):
                    if errorbar=='quantiles':
                        q = self.quantiles(gal_num_components, galaxy=g, q=(0.16, 0.5, 0.84))
                        err_up = q[2][cmp] - q[1][cmp]
                        err_down = q[1][cmp] - q[0][cmp]
                    elif errorbar=='std':
                        std = self.std(gal_num_components, galaxy=g)
                        err_up = std
                        err_down = std
                    elif errorbar is None:
                        err_up = np.nan
                        err_down = np.nan
                    errors[cmp][0, g] = err_up
                    errors[cmp][1, g] = err_down

                    if point_estimate=='max1d':
                        photoz[cmp][g] = self.max(gal_num_components, galaxy=g)[cmp]
                    elif point_estimate=='mean1d':
                        photoz[cmp][g] = self.mean(gal_num_components, galaxy=g)[cmp]
                    else:
                        raise NotImplementedError('No option other than "max1d" or "mean1d" '
                                                + 'for point_estimate is implemented yet.')

                    specz[cmp][g] = gal_true_redshifts[cmp]
            except KeyError:
                for cmp in photoz:
                    photoz[cmp][g] = np.nan
                    specz[cmp][g] = np.nan
                    errors[cmp][:, g] = np.nan
                warnings.warn('Galaxy {} does not have the required '.format(g)
                               + 'spectroscopic information.')

        #Plot the results
        if not heatmap:
            for cmp in num_components_seen:
                if plot_components:
                    col = color[cmp%len(color)]
                    mark = marker[cmp%len(marker)]
                    lab = r'$z_{}$'.format(cmp+1)
                else:
                    col = color[0]
                    mark = marker[0]
                    lab = None
                ax.errorbar(specz[cmp], photoz[cmp], yerr=errors[cmp], linestyle='none',
                            marker=mark, capsize=0, color=col, label=lab)
        else:
            specz_all = np.concatenate([specz[cmp] for cmp in num_components_seen])
            photoz_all = np.concatenate([photoz[cmp] for cmp in num_components_seen])
            ax.hist2d(specz_all, photoz_all, bins=hist_bins, cmap=hist_cmap, normed=hist_normed)
            if colorbar:
                plt.colorbar()

        #Line plotting
        if plot_equal_line:
            ax.plot(line, line, color=line_color, label='Photo-z = True')
        if rel_error_line is not None:
            ax.plot(line, (rel_error_line*(1.+line))+line, color=line_color, linestyle='--')
            ax.plot(line, (-1*rel_error_line*(1.+line))+line, color=line_color, linestyle='--', label='Error / (1+z) > {}'.format(rel_error_line))
        if abs_err_line is not None:
            ax.plot(line, line+abs_err_line, color=line_color, linestyle=':')
            ax.plot(line, line-abs_err_line, color=line_color, linestyle=':', label='Error > {}'.format(abs_err_line))

        #Set plot details
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if legend is not None:
            ax.legend(loc=legend)

        if return_figure:
            return fig

    def plotPrior(self, num_galaxies=1000, bins=50, magnitude_limit=None, figure=None, axes=None,
                  xlabel=r'$z$', ylabel=r'$P(z)$', legend=0, linestyle='-',
                  color=['k', 'b', 'g', 'r', 'c', 'y', 'm'], **fig_kwargs):

        if magnitude_limit is not None:
            sim_data = SimulatedPhotometry(num_galaxies, num_components=1,
                                           model=self.model, magnitude_limit=magnitude_limit)
        else:
            sim_data = SimulatedPhotometry(num_galaxies, num_components=1, model=self.model)
        sim_redshifts = np.array([gal.truth[0]['redshift'] for gal in sim_data])
        sim_types = np.array([sim_data.responses.templates.templateType(
                                    int(gal.truth[0]['template']))
                              for gal in sim_data])
        sim_redshifts_tmp = {}
        for tmp in self.responses.templates.possible_types:
            sim_redshifts_tmp[tmp] = np.array([sim_redshifts[i] for i in
                                               range(len(sim_redshifts))
                                               if sim_types[i]==tmp])

        if (figure is None) and (axes is None):
            fig, ax = plt.subplots(nrows=1, ncols=1, **fig_kwargs)
            return_figure = True
        elif axes is None:
            fig = figure
            ax = fig.axes
            return_figure = False
        else:
            ax = axes
            return_figure = False

        # Plotting code expects iterables but user can specify single string instead
        if type(color) == str:
            color = [color]

        ax.hist(sim_redshifts, bins=bins, histtype='step',
                linestyle=linestyle, color=color[0], label='Total')

        for i, tmp in enumerate(sim_redshifts_tmp):
            ax.hist(sim_redshifts_tmp[tmp], bins=bins, histtype='step',
                linestyle=linestyle, color=color[(i+1)%len(color)], label=tmp)

        '''
        for i, tmp in enumerate(sim_redshifts_tmp):
            hist, bins = np.histogram(sim_redshifts_tmp[tmp], bins=bins, density=True)
            widths = np.diff(bins)
            # Normalise the type histograms to fraction of galaxies of that type
            hist *= float(len(sim_redshifts_tmp[tmp])) / float(num_galaxies)

            #Repeat each height twice, padding with single zero at each end
            heights = np.zeros((len(hist)*2)+2)
            heights[2::2] = hist
            heights[1:-1:2] = hist

            #Repeat each edge twice
            edges = np.zeros(2*len(bins))
            edges[:-1:2] = bins
            edges[1::2] = bins

            ax.plot(edges, heights, linestyle=linestyle,
                     color=color[i%len(color)], label=tmp)
        '''
        if legend is not None:
            ax.legend(loc=legend)
        if xlabel is not None:#into args
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if return_figure:
            return fig
