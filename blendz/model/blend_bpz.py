import numpy as np
from blendz.model import Base

class BlendBPZ(Base):
    def __init__(self, responses=None, photometry=None, prior_params=None):
        super(BlendBPZ, self).__init__(responses=responses, photometry=photometry)
        #BPZ priors assume the reference magnitude is bounded [20, 32]
        for gal in self.photometry:
            if gal.ref_mag_data > 32.:
                gal.bounded_ref_mag = 32.
            elif gal.ref_mag_data < 20.:
                gal.bounded_ref_mag = 20.
            else:
                gal.bounded_ref_mag = gal.ref_mag_data

        #Default to the prior parameters given in Benitez 2000
        if prior_params is not None:
            self.prior_params = prior_params
        else:
            self.prior_params = {'k_t': {'early': 0.45, 'late': 0.147},\
                                 'f_t': {'early': 0.35, 'late': 0.5},\
                                 'alpha_t': {'early': 2.46, 'late': 1.81, 'irr': 0.91},\
                                 'z_0T': {'early': 0.431, 'late': 0.39, 'irr': 0.063},\
                                 'k_mT': {'early': 0.091, 'late': 0.0636, 'irr': 0.123}}

    def lnTemplatePrior(self, template_type):
        mag0 = self.current_galaxy.bounded_ref_mag
        #All include a scaling of 1/Number of templates of that type
        if template_type == 'early':
            Nte = self.responses.templates.num_type('early')
            coeff = np.log(self.prior_params['f_t']['early'] / Nte)
            expon = self.prior_params['k_t']['early'] * (mag0 - 20.)
            out = coeff - expon
        elif template_type == 'late':
            Ntl = self.responses.templates.num_type('late')
            coeff = np.log(self.prior_params['f_t']['late'] / Ntl)
            expon = self.prior_params['k_t']['late'] * (mag0 - 20.)
            out = coeff - expon
        elif template_type == 'irr':
            Nte = self.responses.templates.num_type('early')
            Ntl = self.responses.templates.num_type('late')
            Nti = self.responses.templates.num_type('irr')
            expone = self.prior_params['k_t']['early'] * (mag0 - 20.)
            exponl = self.prior_params['k_t']['late'] * (mag0 - 20.)
            early = self.prior_params['f_t']['early'] * np.exp(-expone)
            late = self.prior_params['f_t']['late'] * np.exp(-exponl)
            out = np.log(1. - early - late) - np.log(Nti)
        else:
            raise ValueError('The BPZ priors are only defined for templates of \
                              types "early", "late" and "irr", but the template \
                              prior was called with type ' + template_type)
        return out

    def lnRedshiftPrior(self, redshift, template_type):
        if template_type not in ['early', 'late', 'irr']:
            raise ValueError('The BPZ priors are only defined for templates of \
                              types "early", "late" and "irr", but the redshift \
                              prior was called with type ' + template_type)

        mag0 = self.current_galaxy.bounded_ref_mag
        first = (alpha_T * np.log(redshift))
        second = self.prior_params['z_0t'][template_type] + (k_mT * (mag0 - 20.))

        return first - (redshift / second)**self.prior_params['alpha_t'][template_type]


    def correlationFunction(self, redshifts):
        #Extra correlation between objects at z1 and z2
        #For now, assume no extra correlation, i.e., xi = 0
        return 0.
