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
        self.num_galaxies = self.photometry.num_galaxies
        self.galaxy = None
        self.precalculateTemplatePriors()

    def precalculateTemplatePriors(self):
        self.template_priors = np.zeros(self.num_galaxies, self.num_templates)
        for gal in self.photometry:
            mag0 = gal.ref_mag_data
            for T in xrange(self.num_templates):
                tmpType = self.responses.templates.template_type(T)
                template_priors[gal.index, T] = self.lnPriorTemplate(tmpType, mag0)

    def lnLikelihood(self, model_colour):
        out = -1. * np.sum((self.galaxy.colour_data - model_colour)**2 / self.galaxy.colour_sigma**2)
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
                    redshift_priors[iz, T] = self.lnPriorRedshift(Z, tmpType)
                    model_fluxes[iz, T, :] = self.responses(T, None, Z)
            redshift_correlation = np.log(1. + self.correlationFunction(redshifts))
            frac_prior = self.lnPriorFrac(fracs)

            #Loop over all templates - discrete marginalisation
            #All log probabilities so (multiply -> add) and (add -> logaddexp)
            lnProb = -np.inf

            #At each iteration template_combo is a tuple of (T_1, T_2... T_nblends)
            for template_combo in itr.product(*itr.repeat(xrange(self.num_templates), nblends)):
                #One redshift prior, template prior and model flux for each blend component
                tmp = 0.
                blend_flux = np.zeros(nblends)
                for b in xrange(nblends):
                    tmp += self.template_priors[self.galaxy.index, template_combo[b]]
                    tmp += redshift_priors[redshifts[b], template_combo[b]]
                    blend_flux += self.model_fluxes[redshifts[b], template_combo[b], :] * fracs[b]

                #Other terms only appear once per summation-step
                tmp += redshift_correlation
                tmp += frac_prior
                tmp += self.lnLikelihood(blend_colour)

                #Define colour wrt reference band
                blend_colour = blend_flux / blend_flux[_config.ref_band]

                #logaddexp contribution from this template to marginalise
                lnProb = np.logaddexp(lnProb, tmp)

            return lnProb

    def sample(self, nblends, gal=None):
        #TODO: Implement this function.
        '''
        nblends should be int, or could be a list so that multiple
        different nb's can be done and compared for evidence etc.

        galaxy should be an int choosing which galaxy of all in photometry
        to estimate the redshift of. If none, do all of them.
        '''
        #self.galaxy = self.photometry[gal]
        pass

    @abc.abstractmethod
    def correlationFunction(self, redshifts):
        pass

    @abc.abstractmethod
    def lnTemplatePrior(self, templateType):
        #Would make sense to pass type, not index, to hide the index->type check
        pass

    @abc.abstractmethod
    def lnRedshiftPrior(self, redshift, templateType):
        pass

    @abc.abstractmethod
    def lnFracPrior(self, fracs):
        pass
