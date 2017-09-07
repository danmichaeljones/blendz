from abc import ABCMeta, abstractmethod

import numpy as np

#TODO: Edit this to use the Responses and Photometry objects
#TODO: Think about how to generalise this to N-blend case

class Base(object):
    __metaclass__ = ABCMeta

    def __init__(self, fluxes, sigmas, mag0):
        self.fluxes = fluxes
        self.sigmas = sigmas

    def modelFlux(self, redshift, templateIndex):
        #Returns array of model fluxes, unscaled
        numBands = len(templateFilterArr[0])
        f_model = np.array([templateFilterArr[templateIndex][F].interp(redshift) for F in xrange(numBands)])
        return f_model

    def negChiSq(self, f_model):
        out = -1. * np.sum((self.fluxes - f_model)**2 / self.sigmas**2)
        return out

    def getModelScaling(self, f_model):
        F_tt = np.sum(f_model**2. / self.sigmas**2.)
        F_ot = np.sum( (self.fluxes*f_model) / self.sigmas**2. )
        scaling = F_ot / F_tt #am from paper, the chiSq minimising scaling
        return scaling

    def lnPosterior(self, redshift1, redshift2, frac1):
        #Loop over all templates - discrete marginalisation
        #All log probabilities so (multiply -> add)
        # and (add -> logaddexp)
        #No in-place incrementing (i.e., +=) as this breaks autograd
        #lnProbTemplate = np.zeros(len(templateList))
        lnProb = -np.inf #Init at -ve inf (i.e., zero probability)
        #Could vary number of for loops (for different number of components in general) with recursion
        # https://stackoverflow.com/questions/7186518/function-with-varying-number-of-for-loops-python

        #Should move things out of the loops a bit to speed up execution.
        #e.g., for 8 templates, each self.modelFlux line is called 64 times each
        #instead of the 8 required. Should calculate what's needed then cache rather
        #than wasting calculations.
        for t1 in xrange(len(templateList)):
            for t2 in xrange(len(templateList)):
                #Template prior
                #Assume independance between template of components so
                #P(T1, T2 | m0) = P(T1 | m0) * P(T2 | m0)
                tmp = self.lnPriorTemplate(t1) + self.lnPriorTemplate(t2)

                #Add redshift prior - check z1 <= z2 for sorting
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
        #logsumexp marginalise
        #lnProb = logsumexp(lnProbTemplate)

        #Catch nans from two negative redshifts
        if np.isfinite(lnProb):
            return lnProb
        else:
            return -np.inf

    @abstractmethod
    def correlationFunction(self, redshift1, redshift2):
        pass

    @abstractmethod
    def lnPriorTemplate(self, templateIndex):
        pass

    @abstractmethod
    def lnPriorRedshift(self, redshift, templateIndex):
        pass

    @abstractmethod
    def lnPriorFrac(self, frac):
        pass
