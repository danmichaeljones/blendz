from builtins import *
import sys
import warnings
import itertools as itr
import numpy as np
import nestle
from tqdm import tqdm
import dill
from blendz.config import _config
from blendz.fluxes import Responses
from blendz.photometry import Photometry, SimulatedPhotometry
from blendz.model import BPZ
from blendz.utilities import incrementCount


class PhotozMag(object):
    def __init__(self, model=None, photometry=None, config=None,\
                 load_state_path=None, sort_redshifts=True):
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
            #Responses and photometry given, just check if configs are equal
            if (model is not None) and (photometry is not None):
                if model.config == photometry.config:
                    self.model = model
                    self.config = self.model.config
                    self.responses = self.model.responses
                    self.photometry = photometry
                else:
                    raise ValueError('Configuration of responses and photometry must be the same.')
            #Only responses given, use its config to load photometry
            elif (model is not None) and (photometry is None):
                self.model = model
                self.config = self.model.config
                self.responses = self.model.responses
                self.photometry = Photometry(config=self.config)
            #Only photometry given, use its config to load responses
            elif (model is None) and (photometry is not None):
                self.config = photometry.config
                self.photometry = photometry
                self.model = BPZ(config=self.config)
                self.responses = self.model.responses
            #Neither given, load both from provided (or default, if None) config
            else:
                if config is None:
                    warnings.warn('USING DEFAULT CONFIG IN PHOTOMETRY, USE THIS FOR TESTING PURPOSES ONLY!')
                    self.config = _config
                else:
                    self.config = config
                self.model = BPZ(config=self.config)
                self.responses = self.model.responses
                self.photometry = Photometry(config=self.config)

            self.num_templates = self.responses.templates.num_templates
            self.num_measurements = self.responses.filters.num_filters
            self.num_galaxies = self.photometry.num_galaxies

            #Move these to config...?
            self.sort_redshifts = sort_redshifts

            #Default to assuming single component, present in all measurements
            self.setMeasurementComponentMapping(None, 1)

            #Set up empty dictionaries to put results into
            self.sample_results = {}
            self.reweighted_samples = {}
            for g in range(self.num_galaxies):
                #Each value is a dictionary which will be filled by sample function
                #The keys of this inner dictionary will be the number of blends for run
                self.sample_results[g] = {}
                self.reweighted_samples[g] = {}


    def precalculateTemplatePriors(self):
        self.template_priors = np.zeros((self.num_galaxies, self.num_templates))
        for gal in self.photometry:
            for T in range(self.num_templates):
                tmpType = self.responses.templates.templateType(T)
                self.template_priors[gal.index, T] = self.lnTemplatePrior(tmpType)

    def saveState(self, filepath):
        #If the photometry is simulated, save the seed as a number rather
        #than as a generator as that will not pickle
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
        with open(filepath, 'r') as f:
            self.__dict__.update(dill.load(f))
        #If the photometry is simulated, replace the seed currently saved as
        #a number with the generator it was before saving
        if isinstance(self.photometry, SimulatedPhotometry):
            try:
                current_seed = self.photometry.sim_seed
                self.photometry.sim_seed = incrementCount(current_seed)
            except:
                warnings.warn('SimulatedPhotometry seed not loaded.')

    def setMeasurementComponentMapping(self, specification, num_components):
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
            for m in range(self.num_measurements):
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

    def lnLikelihood_flux(self, model_flux):
        chi_sq = -1. * np.sum((self.photometry.current_galaxy.flux_data_noRef - model_flux)**2 / self.photometry.current_galaxy.flux_sigma_noRef**2)
        return chi_sq

    def lnLikelihood_mag(self, total_ref_flux):
        #chi_sq = -1. * np.sum((self.photometry.current_galaxy.ref_mag_data - total_ref_mag)**2 / self.photometry.current_galaxy.ref_mag_sigma**2)
        chi_sq = -1. * np.sum((self.photometry.current_galaxy.ref_flux_data - total_ref_flux)**2 / self.photometry.current_galaxy.ref_flux_sigma**2)
        return chi_sq

    def lnPosterior(self, params):
        nblends = int(len(params) // 2)
        redshifts = params[:nblends]
        magnitudes = params[nblends:]
        #Impose prior conditions
        redshift_positive = np.all(redshifts >= 0.)
        if nblends>1:
            #Only need sorting condition if redshifts are exchangable
            # (depends on measurement_component_mapping)
            if self.redshifts_exchangeable:
                #Either sort on redshifts...
                if self.sort_redshifts:
                    sort_condition = np.all(redshifts[1:] >= redshifts[:-1])
                # ... or on magnitudes -
                # magnitudes sort in increasing *numerical* order like redshifts
                # not in brightness!
                else:
                    sort_condition = np.all(magnitudes[1:] >= magnitudes[:-1])
                prior_checks_okay = sort_condition and redshift_positive
            else:
                prior_checks_okay = redshift_positive
        else:
            #Single redshift case, only need to impose redshift being positive
            prior_checks_okay = redshift_positive

        if not prior_checks_okay:
            return -np.inf
        else:
            #Precalculate all quantities we'll need in the template loop
            template_priors = np.zeros((nblends, self.num_templates))
            redshift_priors = np.zeros((nblends, self.num_templates))
            #Single interp call -> Shape = (N_template, N_band, N_component)
            model_fluxes = self.responses.interp(redshifts)

            for T in range(self.num_templates):
                tmpType = self.responses.templates.templateType(T)
                for nb in range(nblends):
                    template_priors[nb, T] = self.model.lnTemplatePrior(tmpType, magnitudes[nb])
                    redshift_priors[nb, T] = self.model.lnRedshiftPrior(redshifts[nb], tmpType, magnitudes[nb])
            redshift_correlation = np.log(1. + self.model.correlationFunction(redshifts))
            #We assume independent magnitudes, so sum over log priors for joint prior
            joint_magnitude_prior = np.sum([self.model.lnMagnitudePrior(m) for m in magnitudes])

            #Loop over all templates - discrete marginalisation
            #All log probabilities so (multiply -> add) and (add -> logaddexp)
            lnProb = -np.inf

            #At each iteration template_combo is a tuple of (T_1, T_2... T_nblends)
            for template_combo in itr.product(*itr.repeat(range(self.num_templates), nblends)):
                #One redshift prior, template prior and model flux for each blend component
                tmp = 0.
                blend_flux = np.zeros(self.num_measurements)
                component_scaling_norm = 0.
                for nb in range(nblends):
                    T = template_combo[nb]
                    component_scaling = 10.**(-0.4*magnitudes[nb]) / model_fluxes[T, self.config.ref_band, nb]
                    blend_flux += model_fluxes[T, :, nb] * component_scaling * self.measurement_component_mapping[nb, :]
                    tmp += template_priors[nb, T]
                    tmp += redshift_priors[nb, T]
                    #################################################print('SCALING')
                    ################################################print component_scaling
                    ################################################print('BLEND FLUX')
                    #################################################print blend_flux
                #Remove ref_band from blend_fluxes, as that goes into the magnitude
                #likelihood, not the flux likelihood
                blend_flux = blend_flux[self.config.non_ref_bands]

                #Get total magnitude in reference band
                # = transform to flux, sum, transform to magnitudes
                ##total_ref_mag = np.log10(np.sum(10.**(-0.4 * magnitudes))) / (-0.4)
                total_ref_flux = np.sum(10.**(-0.4 * magnitudes))
                ################################################print('REF MAG')
                ################################################print total_ref_mag

                #Other terms only appear once per summation-step
                tmp += redshift_correlation
                tmp += self.lnLikelihood_flux(blend_flux)
                ##tmp += self.lnLikelihood_mag(total_ref_mag)
                tmp += self.lnLikelihood_mag(total_ref_flux)
                tmp += joint_magnitude_prior

                #logaddexp contribution from this template to marginalise
                lnProb = np.logaddexp(lnProb, tmp)

            return lnProb

    def priorTransform(self, params): #CHANGE THIS FROM FRACS TO MAGNITUDES!!!!!!!!!!!!!!!!!!!!!!!!!!1
        '''
        Transform params from [0, 1] uniform random to [min, max] uniform random,
        where the min redshift is zero, max redshift is set in the configuration,
        and the min/max magnitudes (numerically, not brightness) are set by configuration.
        '''
        nblends = int(len(params) // 2)

        trans = np.ones(len(params))
        trans[:nblends] = self.config.z_hi
        trans[nblends:] = self.config.ref_mag_hi - self.config.ref_mag_lo

        shift = np.zeros(len(params))
        shift[nblends:] = self.config.ref_mag_lo
        return (params * trans) + shift

    def sampleProgressUpdate(self, info):
        if info['it']%100.==0:
            self.pbar.set_description('[Gal: {}/{}, Comp: {}/{}, Itr: {}] '.format(self.gal_count,
                                                                                   self.num_galaxies_sampling,
                                                                                   self.blend_count,
                                                                                   self.num_components_sampling,
                                                                                   info['it']))
            self.pbar.refresh()

    def sample(self, nblends, galaxy=None, npoints=150, resample=None, seed=None, colour_likelihood=True, measurement_component_mapping=None):
        '''
        nblends should be int, or could be a list so that multiple
        different nb's can be done and compared for evidence etc.

        galaxy should be an int choosing which galaxy of all in photometry
        to estimate the redshift of. If none, do all of them.
        '''
        self.colour_likelihood = colour_likelihood

        if isinstance(nblends, int):
            nblends = [nblends]
        else:
            if measurement_component_mapping is not None:
                #TODO: This is a time-saving hack to avoid dealing with multiple specifications
                #The solution would probably be to rethink the overall design
                raise ValueError('measurement_component_mapping cannot be set when sampling multiple numbers of components in one call. Do the separate cases separately.')

        self.num_components_sampling = len(nblends)

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
                for nb in nblends:

                    if seed is None:
                        rstate = np.random.RandomState()
                    elif seed is True:
                        rstate = np.random.RandomState(gal.index)
                    else:
                        rstate = np.random.RandomState(seed + gal.index)

                    num_param = 2 * nb
                    self.setMeasurementComponentMapping(measurement_component_mapping, nb)
                    results = nestle.sample(self.lnPosterior, self.priorTransform,
                                            num_param, method='multi', npoints=npoints,
                                            rstate=rstate, callback=self.sampleProgressUpdate)
                    self.sample_results[gal.index][nb] = results
                    if resample is not None:
                        #self.reweighted_samples[gal.index][nb] = nestle.resample_equal(results.samples, results.weights)
                        self.reweighted_samples[gal.index][nb] = results.samples[rstate.choice(len(results.weights), size=resample, p=results.weights)]

                    self.gal_count += 1
                    self.blend_count += 1
                    self.pbar.update()
