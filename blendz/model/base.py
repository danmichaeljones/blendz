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

#TODO: Edit this to use the Responses and Photometry objects
#TODO: Think about how to generalise this to N-blend case

'''
The model class should have attriutes for current_galaxy and number of blends.
These only need to be set when the posterior is used in the sampling. So, the cleanest
thing to do is to set them to None at init, and set them to whatever is necessary inside
the sampling function (function in this class that interfaces between the model and the sampler).
If the posterior is called outside of the sampling function for whatever reason,
there should be an Exception explaining how this works.

All this might work better just passing parameters to lnPosterior()
'''

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

    def negChiSq(self, model_colour):
        out = -1. * np.sum((self.galaxy.colour_data - model_colour)**2 / self.galaxy.colour_sigma**2)
        return out

    def lnPosterior(self, params):
        nblends = (len(params)+1)/2
        redshifts = params[:nblends]
        fracs = params[nblends:]
        if nblends>1:
            #Impose a prior on redshifts being sorted
            redshifts_okay = np.all(redshifts[1:] >= redshifts[:-1])
            #Impose a prior on sum(frac) = 1
            #fracs currently has length nblends-1
            #so impose sum <= 1 and append final element after
            fracs_okay = np.sum(fracs) <= 1.
            params_okay = redshifts_okay * fracs_okay
        else:
            #Single object case, so no need to check sorting/sum conditions
            params_okay = True

        if not params_okay:
            return -np.inf
        else:
            #Get final frac by imposing sum-to-one condition
            fracs = np.append(fracs, 1.-np.sum(fracs))

            #Precalculate all quantities we'll need in the template loop
            redshift_priors = np.zeros((nblends, self.num_templates))
            model_fluxes = np.zeros((nblends, self.num_templates))
            for iz, Z in enumerate(redshifts):
                for T in xrange(self.num_templates):
                    tmpType = self.responses.templates.template_type(T)
                    redshift_priors[iz, T] = self.lnPriorRedshift(Z, tmpType)
                    model_fluxes[iz, T] = self.modelFlux(Z, T)
            correlation_function = self.correlationFunction(redshifts)
            frac_prior = self.lnPriorFrac(fracs)

            #Loop over all templates - discrete marginalisation
            #All log probabilities so (multiply -> add)
            # and (add -> logaddexp)
            lnProb = -np.inf #Init at -ve inf (i.e., zero probability)

            #At each iteration template_combo is a tuple of (T_1, T_2... T_nblends)
            for template_combo in itr.product(*itr.repeat(xrange(self.num_templates), nblends)):
                #Add a loop over nblends components here probably, should be
                #able to combine the loops below into this one

                #Template prior
                #Assume independance between template of components so
                #P(T1, T2 | m0) = P(T1 | m0) * P(T2 | m0) * ...
                for T in template_combo:
                    tmp += self.template_priors[self.galaxy.index, T]

                #Add redshift prior
                #Product of P(z1) and P(z2), with extra correlation from xi(r[z1, z2])
                for b in xrange(nblends): #this could go into the T loop...
                    tmp += redshift_priors[redshifts[b], template_combo[b]]

                #Extra correlation between redshifts over independent
                tmp += (1. + correlation_function)

                #Calculate the model flux at these parameters
                #Two components - total model flux is the sum
                #This is *actual* model flux not the log of it, so
                #just sum them together
                f_model_unscaled1 = self.modelFlux(redshift1, t1) * frac1
                f_model_unscaled2 = self.modelFlux(redshift2, t2) * (1. - frac1)
                f_model_unscaled = f_model_unscaled1 + f_model_unscaled2
                f_model = f_model_unscaled * self.getModelScaling(f_model_unscaled)

                #Add flux fraction prior
                tmp = tmp + self.lnPriorFrac(frac1)

                #Add likelihood
                #lnProbTemplate[ti] += self.negChiSq(f_model)
                tmp = tmp + self.negChiSq(f_model)

                #logaddexp contribution from this template to marginalise
                lnProb = np.logaddexp(lnProb, tmp)

        #Catch nans from two negative redshifts
        if np.isfinite(lnProb):
            return lnProb
        else:
            return -np.inf

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
    def lnRedshift(self, redshift, templateType):
        pass

    @abc.abstractmethod
    def lnFracPrior(self, fracs):
        pass
