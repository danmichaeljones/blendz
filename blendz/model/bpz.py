from builtins import *
import numpy as np
from scipy.interpolate import interp1d
from blendz.model import ModelBase

class BPZ(ModelBase):
    def __init__(self, mag_grid_len=100, **kwargs):
        super(BPZ, self).__init__(**kwargs)

        self.mag_grid_len = mag_grid_len
        self.possible_types = self.responses.templates.possible_types
        if self.prior_params is not np.nan:
            self._loadParameterDict()
            self._calculateRedshiftPriorNorm()

    def _loadParameterDict(self):
        nt = len(self.possible_types)

        if len(self.prior_params) != 5 * len(self.possible_types) - 2:
            raise ValueError('Wrong number of parameters')

        kt = {t: self.prior_params[i] for i, t in enumerate(self.possible_types[:-1])}
        ft = {t: self.prior_params[i + nt - 1] for i, t in enumerate(self.possible_types[:-1])}
        alpt = {t: self.prior_params[i + 2*nt - 2] for i, t in enumerate(self.possible_types)}
        z0t = {t: self.prior_params[i + 3*nt - 2] for i, t in enumerate(self.possible_types)}
        kmt = {t: self.prior_params[i + 4*nt - 2] for i, t in enumerate(self.possible_types)}

        self.prior_params_dict = {
            'k_t': kt, 'f_t': ft, 'alpha_t': alpt, 'z_0t': z0t, 'k_mt': kmt
        }

    def _calculateRedshiftPriorNorm(self):
        self.redshift_prior_norm = {}
        mag_range = np.linspace(self.config.ref_mag_lo, self.config.ref_mag_hi, self.mag_grid_len)
        for T in self.possible_types:
            norms = np.zeros(self.mag_grid_len)
            for i, mag in enumerate(mag_range):
                zi = np.exp(np.array([self.lnRedshiftPrior(zz, T, mag, norm=False) for zz in self.responses.zGrid]))
                norms[i] = np.log(1./np.trapz(zi[np.isfinite(zi)], x=self.responses.zGrid[np.isfinite(zi)]))
            self.redshift_prior_norm[T] = interp1d(mag_range, norms)

    def lnTemplatePrior(self, template_type, component_ref_mag):
        mag_diff = component_ref_mag - self.config.ref_mag_lo
        #All include a scaling of 1/Number of templates of that type
        if template_type in self.possible_types[:-1]:
            Nt = self.responses.templates.numType(template_type)
            coeff = np.log(self.prior_params_dict['f_t'][template_type] / Nt)
            expon = self.prior_params_dict['k_t'][template_type] * mag_diff
            out = coeff - expon
        #Prior for final type = 1 - prior of other types
        elif template_type == self.possible_types[-1]:
            Nt = self.responses.templates.numType(template_type)
            other_types = 0.
            for T in self.possible_types[:-1]:
                expon = self.prior_params_dict['k_t'][T] * mag_diff
                other_types += self.prior_params_dict['f_t'][T] * np.exp(-expon)
            out = np.log(1. - other_types) - np.log(Nt)
        else:
            raise ValueError('The possible galaxy types based on your template '
                             'set are "' + '", "'.join(self.possible_types) + '", but the '
                             'template prior was called with type ' + template_type)
        return out

    def lnRedshiftPrior(self, redshift, template_type, component_ref_mag, norm=True):
        try:
            if redshift==0:
                first = -np.inf
            else:
                first = (self.prior_params_dict['alpha_t'][template_type] * np.log(redshift))
            second = self.prior_params_dict['z_0t'][template_type] + (self.prior_params_dict['k_mt'][template_type] * (component_ref_mag - self.config.ref_mag_lo))
            out = first - (redshift / second)**self.prior_params_dict['alpha_t'][template_type]
        except KeyError:
            raise ValueError('The possible galaxy types based on your template '
                             'set are "' + '", "'.join(self.possible_types) + '", but the '
                             'redshift prior was called with type ' + template_type)
        if norm:
            try:
                out = out + self.redshift_prior_norm[template_type](component_ref_mag)
            except ValueError:
                raise ValueError('Magnitude = {} is outside of prior-precalculation '
                                 'range. Check your configuration ref-mag limits'
                                 'cover your input magnitudes.'.format(component_ref_mag))
            return out
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

    def lnPrior(self, redshift, magnitude):
        #Just the single component prior given TYPE index (not template)
        prior = np.zeros(len(self.possible_types))
        for i, template_type in enumerate(self.possible_types):
            p_z = self.lnRedshiftPrior(redshift, template_type, magnitude)
            p_t = self.lnTemplatePrior(template_type, magnitude)
            p_m = self.lnMagnitudePrior(magnitude)
            prior[i] = p_z * p_t * p_m
        return prior

    def lnPriorCalibrationPrior(self):
        '''Returns the prior on the prior parameters for the calibration procedure.'''
        #Assume a flat prior, except that sum(type fractions) <= 1. ...
        if sum(self.prior_params_dict['f_t'].values()) > 1.:
            return -np.inf
        #... and all parameters are positive
        # TEMPORARY TEST - ALLOW NEGATIVE Ks, IMPOSE REST POSITIVE IN CALIBRATION
        #elif np.any(self.prior_params < 0.):
        #    return -np.inf
        else:
            return 0.
