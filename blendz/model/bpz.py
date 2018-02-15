from builtins import *
import numpy as np
from scipy.interpolate import interp1d
from blendz.model import ModelBase

class BPZ(ModelBase):
    def __init__(self, mag_grid_len=100, **kwargs):
        super(BPZ, self).__init__(**kwargs)
        #Default to the prior parameters given in Benitez 2000
        self.prior_params_dict = {'k_t': {'early': self.prior_params[0], \
                                          'late': self.prior_params[1]}, \
                                 'f_t': {'early': self.prior_params[2], \
                                         'late': self.prior_params[3]}, \
                                 'alpha_t': {'early': self.prior_params[4], \
                                             'late': self.prior_params[5], \
                                             'irr': self.prior_params[6]}, \
                                  'z_0t': {'early': self.prior_params[7], \
                                           'late': self.prior_params[8], \
                                           'irr': self.prior_params[9]}, \
                                  'k_mt': {'early': self.prior_params[10], \
                                           'late': self.prior_params[11], \
                                           'irr': self.prior_params[12]}}
        self.mag_grid_len = mag_grid_len
        self._calculateRedshiftPriorNorm()

    def _calculateRedshiftPriorNorm(self):
        self.redshift_prior_norm = {}
        mag_range = np.linspace(self.config.ref_mag_lo, self.config.ref_mag_hi, self.mag_grid_len)
        for T in self.responses.templates.possible_types:
            norms = np.zeros(self.mag_grid_len)
            for i, mag in enumerate(mag_range):
                zi = np.exp(np.array([self.lnRedshiftPrior(zz, T, mag, norm=False) for zz in self.responses.zGrid]))
                norms[i] = np.log(1./np.trapz(zi[np.isfinite(zi)], x=self.responses.zGrid[np.isfinite(zi)]))
            self.redshift_prior_norm[T] = interp1d(mag_range, norms)

    def lnTemplatePrior(self, template_type, component_ref_mag):
        #All include a scaling of 1/Number of templates of that type
        if template_type in ['early', 'late']:
            Nt = self.responses.templates.numType(template_type)
            coeff = np.log(self.prior_params_dict['f_t'][template_type] / Nt)
            expon = self.prior_params_dict['k_t'][template_type] * (component_ref_mag - 20.)
            out = coeff - expon
        elif template_type == 'irr':
            Nte = self.responses.templates.numType('early')
            Ntl = self.responses.templates.numType('late')
            Nti = self.responses.templates.numType('irr')
            expone = self.prior_params_dict['k_t']['early'] * (component_ref_mag - 20.)
            exponl = self.prior_params_dict['k_t']['late'] * (component_ref_mag - 20.)
            early = self.prior_params_dict['f_t']['early'] * np.exp(-expone)
            late = self.prior_params_dict['f_t']['late'] * np.exp(-exponl)
            out = np.log(1. - early - late) - np.log(Nti)
        else:
            raise ValueError('The BPZ priors are only defined for templates of \
                              types "early", "late" and "irr", but the template \
                              prior was called with type ' + template_type)
        return out

    def lnRedshiftPrior(self, redshift, template_type, component_ref_mag, norm=True):
        try:
            if redshift==0:
                first = -np.inf
            else:
                first = (self.prior_params_dict['alpha_t'][template_type] * np.log(redshift))
            second = self.prior_params_dict['z_0t'][template_type] + (self.prior_params_dict['k_mt'][template_type] * (component_ref_mag - 20.))
            out = first - (redshift / second)**self.prior_params_dict['alpha_t'][template_type]
        except KeyError:
            raise ValueError('The BPZ priors are only defined for templates of \
                              types "early", "late" and "irr", but the redshift \
                              prior was called with type ' + template_type)
        if norm:
            return out + self.redshift_prior_norm[template_type](component_ref_mag)
        else:
            return out

    def correlationFunction(self, redshifts):
        if len(redshifts)==1:
            return 0.
        elif len(redshifts)==2:
            if not self.config.sort_redshifts:
                redshifts = np.sort(redshifts)
            separation = self.comovingSeparation(redshifts[0], redshifts[1])
            #Small-scale cutoff
            if separation < self.config.xi_r_cutoff:
                separation = self.config.xi_r_cutoff
            return (self.config.r0 / separation)**self.config.gamma
        else:
            raise NotImplementedError('No N>2 yet...')

    def lnMagnitudePrior(self, magnitude):
        return 0.6*(magnitude - self.config.ref_mag_hi) * np.log(10.)

    def lnPriorCalibrationPrior(self):
        '''Returns the prior on the prior parameters for the calibration procedure.'''
        #Assume a flat prior, except that f_t_early + f_t_late <= 1. ...
        if self.prior_params_dict['f_t']['early'] + par_dict['f_t']['late'] > 1.:
            return -np.inf
        #... and all parameters are positive
        elif np.any(self.prior_params < 0.):
            return -np.inf
        else:
            return 0.
