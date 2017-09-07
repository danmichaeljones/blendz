import numpy as np

from base import base

#TODO: Edit this to use the Responses and Photometry objects
#TODO: Move some of this functionality to a base class, and subclass the HDFN prior parts to keep general
#TODO: Think about how to generalise this to N-blend case (presumably in the base class)
#TODO: Would these priors be better taking template types? The idea is that these should take
#the relevant scalars only, and all the complexity of interacting with the rest of the
#model an handling the N-blend parts is handled completely by the Base class. In that spirit, the
#priors should only care about a particular template type, not it's integer
#(although the template prior will need the number of each type for normalising, maybe it's best we
#just assume that the prior functions will use the self.attributs, like Templates, that we have in Base)

class TwoBlendModel(Base):
    def __init__(self, fluxes, sigmas, mag0):
        self.fluxes = fluxes
        self.sigmas = sigmas
        if mag0 > 32.:
            self.mag0 = 32.
        elif mag0 < 20.:
            self.mag0 = 20.
        else:
            self.mag0 = mag0

    def lnPriorTemplate(self, templateIndex):
        #This is the parameters the BPZ code uses, NOT what is in the paper...
        #The BPZ code prior *seems* to have early/late mixed up...
        k_t_early = 0.45#0.147
        f_t_early = 0.35
        k_t_late = 0.147#0.45
        f_t_late = 0.5

        templateType = templateTypeList[templateIndex]
        #All include a scaling of 1/Number of templates of that type
        if templateType == 'early':
            # f_t * exp(-k_t[m0 - 20]) / NumberOfEarlyTemplates
            Nte = numTemplateType['early']
            out = np.log(f_t_early/Nte) - (k_t_early * (self.mag0 - 20.))
        elif templateType == 'late':
            # f_t * exp(-k_t[m0 - 20]) / NumberOfLateTemplates
            Ntl = numTemplateType['late']
            out = np.log(f_t_late/Ntl) - (k_t_late * (self.mag0 - 20.))
        elif templateType == 'irr':
            Nte = numTemplateType['early']
            Ntl = numTemplateType['late']
            Nti = numTemplateType['irr']
            early = (f_t_early * np.exp(-k_t_early * (self.mag0 - 20.)))
            late = (f_t_late * np.exp(-k_t_late * (self.mag0 - 20.)))
            out = np.log(1. - early - late) - np.log(Nti)
        return out

    def lnPriorRedshift(self, redshift, templateIndex):
        templateType = templateTypeList[templateIndex]

        #Values from Benitez 2000
        if templateType == 'early':
            alpha_T = 2.46
            redshift_0T = 0.431
            k_mT = 0.091
        elif templateType == 'late':
            alpha_T = 1.81
            redshift_0T = 0.39
            k_mT = 0.0636
        elif templateType == 'irr':
            alpha_T = 0.91
            redshift_0T = 0.063
            k_mT = 0.123

        #Prior on redshift being non-negative
        out = (alpha_T * np.log(redshift)) - (redshift / (redshift_0T + k_mT*(self.mag0 - 20.)))**alpha_T
        try:
            out[redshift<0.] = -np.inf
        except:
            if redshift<0.:
                out = -np.inf
        return out

    def lnPriorFrac(self, frac):
        #Prior on the fraction
        #Uniform(0, 1) -- no preference for 1/2 component models
        if frac < 0.:
            return -np.inf
        elif frac > 1:
            return -np.inf
        else:
            return 0.

    def correlationFunction(self, redshift1, redshift2):
        #Extra correlation between objects at z1 and z2
        #For now, assume no extra correlation, i.e., xi = 0
        return 0.
