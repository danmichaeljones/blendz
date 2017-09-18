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
        self.precalculateTemplatePriors()

    def precalculateTemplatePriors(self):
        self.template_priors = np.zeros(self.num_galaxies, self.num_templates)
        for g in xrange(self.num_galaxies):
            mag0 = self.ref_mag_data[g]
            for t in xrange(self.num_templates):
                tmpType = self.responses.templates.template_type(t)
                template_priors[g, t] = self.lnPriorTemplate(tmpType, mag0)

    def negChiSq(self, c_model, gal):
        out = -1. * np.sum((self.photometry.colour_data[gal] - c_model)**2 / self.photometry.colour_sigma_data[gal]**2)
        return out

    def lnPosterior(self, redshifts, frac1, nblends, galaxy):
        #Loop over all templates - discrete marginalisation
        #All log probabilities so (multiply -> add)
        # and (add -> logaddexp)
        lnProb = -np.inf #Init at -ve inf (i.e., zero probability)

        #Should move things out of the loops a bit to speed up execution.
        #e.g., for 8 templates, each self.modelFlux line is called 64 times each
        #instead of the 8 required. Should calculate what's needed then cache rather
        #than wasting calculations.

        #Precalculate the redshift priors
        redshift_priors = np.zeros(len(redshifts))

        #At each iteration template_combo is a tuple of (T_1, T_2... T_nblends)
        for template_combo in itr.product(*itr.repeat(xrange(self.num_templates), nblends)):
            #Inside here, instead of fixing each line to have tmp += f(t1) + f(t2)
            #which is for 2-blends, can can increment tmp with f(ti) for ti in x from above
            #The redshift sorting should be done as checking ascending(zArray)
            #Maybe would make sense to add a z-positive prior in here rather
            #than relying on the prior for easier usage.
            #This whole idea might require frac be an array nb long. This could be done
            #by moving to a Dirichlet prior on frac so that we can enforce sum-to-one.
            #The posterior will have to be changed to accept parameters as a 1d array, theta.
            #Since we only need reshifts and frac, each are nb long, and theta is 2*nb long.

            #Template prior
            #Assume independance between template of components so
            #P(T1, T2 | m0) = P(T1 | m0) * P(T2 | m0) * ...
            for T in template_combo:
                tmp += self.template_priors(galaxy, T)

            #Add redshift prior - check z1 <= z2 <= ... for sorting
            if redshift1 <= redshift2:
                #Product of P(z1) and P(z2), with extra correlation from xi(r[z1, z2])
                tmp = tmp + self.lnPriorRedshift(redshift1, t1)
                tmp = tmp + self.lnPriorRedshift(redshift2, t2)
                tmp = tmp + (1. + self.correlationFunction(redshift1, redshift2))
            else:
                tmp = tmp -np.inf

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

    def sample(self, nblends=None, gal=None):
        #TODO: Implement this function.
        '''
        nblends should be a list so that multiple
        different nb's can be done and compared for evidence etc.
        If none, some default maybe? Maybe demand it's set..

        galaxy should be an int choosing which galaxy of all in photometry
        to estimate the redshift of. If none, do all of them.
        '''
        pass

    @abc.abstractmethod
    def correlationFunction(self, redshift1, redshift2):
        pass

    @abc.abstractmethod
    def lnTemplatePrior(self, templateIndex):
        #Would make sense to pass type, not index, to hide the index->type check
        pass

    @abc.abstractmethod
    def lnRedshift(self, redshift, templateIndex):
        pass

    @abc.abstractmethod
    def lnFracPrior(self, frac):
        pass
