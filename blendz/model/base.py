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
from blendz.config import _config
from blendz.fluxes import Responses
from blendz.photometry import Photometry

#TODO: Write sample function, which should deal with setting galaxy to the current Galaxy object

class Base(ABC_meta):
    def __init__(self, responses=None, photometry=None):
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

    def lnLikelihood(self, model_colour):
        out = -1. * np.sum((self.photometry.current_galaxy.colour_data - model_colour)**2 / self.photometry.current_galaxy.colour_sigma**2)
        return out

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

            #Precalculate all quantities we'll need in the template loop
            redshift_priors = np.zeros((nblends, self.num_templates))
            model_fluxes = np.zeros((nblends, self.num_templates, self.responses.filters.num_filters))
            for iz, Z in enumerate(redshifts):
                for T in xrange(self.num_templates):
                    tmpType = self.responses.templates.template_type(T)
                    redshift_priors[iz, T] = self.lnRedshiftPrior(Z, tmpType)
                    model_fluxes[iz, T, :] = self.responses(T, None, Z)
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
                for b in xrange(nblends):
                    blend_flux += model_fluxes[b, template_combo[b], :] * fracs[b]
                    tmp += redshift_priors[b, template_combo[b]]
                    #Precalculate all the template priors the first time posterior is called
                    try:
                        tmp += self.template_priors[self.photometry.current_galaxy.index, template_combo[b]]
                    except AttributeError:
                        incoming_galaxy_index = self.photometry.current_galaxy.index
                        self.precalculateTemplatePriors()
                        self.photometry.current_galaxy = self.photometry[incoming_galaxy_index]
                        tmp += self.template_priors[self.photometry.current_galaxy.index, template_combo[b]]

                #Define colour wrt reference band
                blend_colour = blend_flux / blend_flux[_config.ref_band]

                #Other terms only appear once per summation-step
                tmp += redshift_correlation
                tmp += frac_prior
                tmp += self.lnLikelihood(blend_colour)

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

    def sample(self, nblends, galaxy=None, npoints=150, resample=None):
        '''
        nblends should be int, or could be a list so that multiple
        different nb's can be done and compared for evidence etc.

        galaxy should be an int choosing which galaxy of all in photometry
        to estimate the redshift of. If none, do all of them.
        '''

        if isinstance(nblends, int):
            nblends = [nblends]
        num_components_sampling = len(nblends)

        if galaxy is None:
            start = None
            stop = None
            num_galaxies_sampling = self.num_galaxies
        elif isinstance(galaxy, int):
            start = galaxy
            stop = galaxy + 1
            num_galaxies_sampling = 1
        else:
            raise TypeError('galaxy may be either None or an integer, but got {} instead'.format(type(galaxy)))

        with tqdm(total=num_galaxies_sampling*num_components_sampling) as pbar:
            gal_count = 1
            for gal in self.photometry.iterate(start, stop):
                blend_count = 1
                for nb in nblends:
                    pbar.set_description('[Galaxy {}/{}, Component {}/{}] '.format(gal_count,
                                                                               num_galaxies_sampling,
                                                                               blend_count,
                                                                               num_components_sampling))
                    pbar.refresh()
                    gal_count += 1
                    blend_count += 1

                    num_param = (2 * nb) - 1
                    results = nestle.sample(self.lnPosterior, self.priorTransform,
                                            num_param, method='multi', npoints=npoints)
                    self.sample_results[gal.index][nb] = results
                    if resample is not None:
                        #self.reweighted_samples[gal.index][nb] = nestle.resample_equal(results.samples, results.weights)
                        self.reweighted_samples[gal.index][nb] = results.samples[np.random.choice(len(results.weights), size=resample, p=results.weights)]
                    pbar.update()

    @abc.abstractmethod
    def correlationFunction(self, redshifts):
        pass

    @abc.abstractmethod
    def lnTemplatePrior(self, template_type):
        #Would make sense to pass type, not index, to hide the index->type check
        pass

    @abc.abstractmethod
    def lnRedshiftPrior(self, redshift, template_type):
        pass
