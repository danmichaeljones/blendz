import sys
import abc
#Python 2 & 3 compatibility for abstract base classes, from
#https://stackoverflow.com/questions/35673474/
if sys.version_info >= (3, 4):
    ABC_meta = abc.ABC
else:
    ABC_meta = abc.ABCMeta('ABC', (), {})
import itertools as itr
import numpy as np
import nestle
from tqdm import tqdm
import dill
from blendz.config import _config
from blendz.fluxes import Responses
from blendz.photometry import Photometry

#TODO: Write sample function, which should deal with setting galaxy to the current Galaxy object

class Base(ABC_meta):
    def __init__(self, responses=None, photometry=None, load_state_path=None):
        if load_state_path is not None:
            self.loadState(load_state_path)
        else:
            if responses is None:
                self.responses = Responses()
            else:
                self.responses = responses
            if photometry is None:
                self.photometry = Photometry()
            else:
                self.photometry = photometry
            self.num_templates = self.responses.templates.num_templates
            self.num_filters = self.responses.filters.num_filters
            self.num_galaxies = self.photometry.num_galaxies
            self.colour_likelihood = True

            #Set up empty dictionaries to put results into
            self.sample_results = {}
            self.reweighted_samples = {}
            for g in xrange(self.num_galaxies):
                #Each value is a dictionary which will be filled by sample function
                #The keys of this inner dictionary will be the number of blends for run
                self.sample_results[g] = {}
                self.reweighted_samples[g] = {}


    def precalculateTemplatePriors(self):
        self.template_priors = np.zeros((self.num_galaxies, self.num_templates))
        for gal in self.photometry:
            for T in xrange(self.num_templates):
                tmpType = self.responses.templates.template_type(T)
                self.template_priors[gal.index, T] = self.lnTemplatePrior(tmpType)

    def saveState(self, filepath):
        with open(filepath, 'wb') as f:
            #Need to exclude the progress bar object as cannot pickle
            state = {key: val for key, val in self.__dict__.items() if key!='pbar'}
            dill.dump(state, f)

    def loadState(self, filepath):
        with open(filepath, 'r') as f:
            #self.__dict__.update(dill.load(f).__dict__)
            self.__dict__.update(dill.load(f))

    def lnLikelihood_col(self, model_colour):
        out = -1. * np.sum((self.photometry.current_galaxy.colour_data - model_colour)**2 / self.photometry.current_galaxy.colour_sigma**2)
        return out

    def lnLikelihood_bpz(self, model_flux):
        F_tt = np.sum(model_flux**2. / self.photometry.current_galaxy.flux_sigma**2.)
        F_ot = np.sum((self.photometry.current_galaxy.flux_data * model_flux) / self.photometry.current_galaxy.flux_sigma**2.)
        scaling = F_ot / F_tt
        scaled_model_flux = model_flux * scaling

        chi_sq = -1. * np.sum((self.photometry.current_galaxy.flux_data - scaled_model_flux)**2 / self.photometry.current_galaxy.flux_sigma**2)
        return chi_sq

    def lnFracPrior(self, frac):
        #Uniform prior on fractions
        return 0.

    def lnPosterior(self, params):
        nblends = (len(params)+1)/2
        redshifts = params[:nblends]
        fracs = params[nblends:]
        #Impose prior on redshifts being sorted/positive and fracs will (after f_nb) sum to 1
        if nblends>1:
            redshifts_sorted = np.all(redshifts[1:] >= redshifts[:-1])
            redshift_positive = np.all(redshifts >= 0.)
            frac_maximum = (np.sum(fracs) <= 1.)
            prior_checks_okay = redshifts_sorted and redshift_positive and frac_maximum
        else:
            #Single redshift case, only need to impose redshift being positive
            prior_checks_okay = np.all(redshifts >= 0.)

        if not prior_checks_okay:
            return -np.inf
        else:
            #Get final frac by imposing sum-to-one condition
            fracs = np.append(fracs, 1.-np.sum(fracs))
            #Get m_{0, a}, m_{0, b} ... using reference flux and fractions, convert back to mags for priors
            component_ref_mags = np.log10(self.photometry.current_galaxy.ref_flux_data * fracs) / (-0.4)

            #Precalculate all quantities we'll need in the template loop
            template_priors = np.zeros((nblends, self.num_templates))
            redshift_priors = np.zeros((nblends, self.num_templates))
            #Single interp call -> Shape = (N_template, N_band, N_component)
            #model_fluxes = np.zeros((nblends, self.num_templates, self.responses.filters.num_filters))
            #Different order indices to before!
            model_fluxes = self.responses.interp(redshifts)

            for T in xrange(self.num_templates):
                tmpType = self.responses.templates.template_type(T)
                for nb in xrange(nblends):
                    template_priors[nb, T] = self.lnTemplatePrior(tmpType, component_ref_mags[nb])
                    redshift_priors[nb, T] = self.lnRedshiftPrior(redshifts[nb], tmpType, component_ref_mags[nb])
            redshift_correlation = np.log(1. + self.correlationFunction(redshifts))
            frac_prior = self.lnFracPrior(fracs)

            #Loop over all templates - discrete marginalisation
            #All log probabilities so (multiply -> add) and (add -> logaddexp)
            lnProb = -np.inf

            #At each iteration template_combo is a tuple of (T_1, T_2... T_nblends)
            for template_combo in itr.product(*itr.repeat(xrange(self.num_templates), nblends)):
                #One redshift prior, template prior and model flux for each blend component
                tmp = 0.
                blend_flux = np.zeros(self.num_filters)
                component_scaling_norm = 0.
                for nb in xrange(nblends):
                    T = template_combo[nb]
                    component_scaling = fracs[nb] / model_fluxes[T, _config.ref_band, nb]
                    component_scaling_norm += component_scaling
                    blend_flux += model_fluxes[T, :, nb] * component_scaling
                    tmp += template_priors[nb, T]
                    tmp += redshift_priors[nb, T]
                #Enforce the scaling summing to one
                blend_flux /= component_scaling_norm

                #Define colour wrt reference band
                blend_colour = blend_flux / blend_flux[_config.ref_band]

                #Other terms only appear once per summation-step
                tmp += redshift_correlation
                tmp += frac_prior
                if self.colour_likelihood:
                    tmp += self.lnLikelihood_col(blend_colour)
                else:
                    tmp += self.lnLikelihood_bpz(blend_flux)

                #logaddexp contribution from this template to marginalise
                lnProb = np.logaddexp(lnProb, tmp)

            return lnProb

    def priorTransform(self, params):
        '''
        Transform params from [0, 1] uniform random to [0, max] uniform random,
        where the max redshift is set in the configuration, and the max fraction
        is 1.
        '''
        nblends = (len(params)+1)/2
        trans = np.ones(len(params))
        trans[:nblends] = _config.z_hi
        return params * trans

    def sampleProgressUpdate(self, info):
        if info['it']%100.==0:
            self.pbar.set_description('[Gal: {}/{}, Comp: {}/{}, Itr: {}] '.format(self.gal_count,
                                                                                   self.num_galaxies_sampling,
                                                                                   self.blend_count,
                                                                                   self.num_components_sampling,
                                                                                   info['it']))
            self.pbar.refresh()

    def sample(self, nblends, galaxy=None, npoints=150, resample=None, seed=None, colour_likelihood=True):
        '''
        nblends should be int, or could be a list so that multiple
        different nb's can be done and compared for evidence etc.

        galaxy should be an int choosing which galaxy of all in photometry
        to estimate the redshift of. If none, do all of them.
        '''
        self.colour_likelihood = colour_likelihood

        if isinstance(nblends, int):
            nblends = [nblends]
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

                    num_param = (2 * nb) - 1
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

    @abc.abstractmethod
    def correlationFunction(self, redshifts):
        pass

    @abc.abstractmethod
    def lnTemplatePrior(self, template_type, component_ref_mag):
        pass

    @abc.abstractmethod
    def lnRedshiftPrior(self, redshift, template_type, component_ref_mag):
        pass
