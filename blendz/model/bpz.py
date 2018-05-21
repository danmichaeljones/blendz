from builtins import *
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from itertools import repeat
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
                                 'range. Check your configuration ref-mag limits '
                                 'cover your input magnitudes.'.format(component_ref_mag))
            return out
        else:
            return out

    def correlationFunction(self, redshifts):
        if len(redshifts)==1:
            return 0.
        elif len(redshifts)==2:
            theta = self.config.angular_resolution
            redshifts = np.sort(redshifts)
            r_2 = self.comovingSeparation(0., redshifts[1])
            delta_r = self.comovingSeparation(redshifts[0], redshifts[1])
            power = 1. - (self.config.gamma/2.)
            one = (self.config.r0**2.) / (power * r_2 * r_2 * theta * theta)
            two = (delta_r**2 + (r_2 * r_2 * theta * theta)) / (self.config.r0**2.)
            three = (delta_r**2) / (self.config.r0**2.)
            return one * ( (two**power) - (three**power) )
        else:
            #Could define this recursively by calling with a
            #len 2 slice of redshifts - assuming bispectrum and above = 0
            raise NotImplementedError('No N>2 yet...')

    def lnMagnitudePrior(self, magnitude):
        return 0.6*(magnitude - self.config.ref_mag_hi) * np.log(10.)

    def lnPrior(self, redshift, magnitude):
        #Just the single component prior given TYPE index (not template)
        lnPriorOut = np.zeros(len(self.possible_types))
        for i, template_type in enumerate(self.possible_types):
            p_z = self.lnRedshiftPrior(redshift, template_type, magnitude)
            p_t = self.lnTemplatePrior(template_type, magnitude)
            p_m = self.lnMagnitudePrior(magnitude)
            lnPriorOut[i] = p_z + p_t + p_m
        return lnPriorOut

    def lnPriorCalibrationPrior(self):
        '''Returns the prior on the prior parameters for the calibration procedure.'''
        #Assume a flat prior, except that sum(type fractions) <= 1. ...
        if sum(self.prior_params_dict['f_t'].values()) > 1.:
            return -np.inf
        elif np.any(self.prior_params < 0.):
            return -np.inf
        else:
            return 0.


    def _lnPriorCalibrationPosterior(self, params, photometry):
        self.prior_params = params
        self._loadParameterDict()
        self._calculateRedshiftPriorNorm()

        calibration_prior = self.lnPriorCalibrationPrior()

        if not np.isfinite(calibration_prior):
            return -np.inf
        else:
            lnProb_all = 0.
            for g in photometry:
                total_ref_mag = g.ref_mag_data
                magnitude_prior = self.lnMagnitudePrior(total_ref_mag)
                if not np.isfinite(magnitude_prior):
                    pass
                else:
                    total_ref_flux = 10.**(-0.4 * total_ref_mag)
                    selection_effect = self.lnSelection(total_ref_flux, g)
                    template_priors = np.zeros(self.num_templates)
                    redshift_priors = np.zeros(self.num_templates)
                    lnProb_g = -np.inf

                    cache_lnTemplatePrior = {}
                    cache_lnRedshiftPrior = {}
                    for tmpType in self.responses.templates.possible_types:
                        cache_lnTemplatePrior[tmpType] = self.lnTemplatePrior(tmpType, total_ref_mag)
                        cache_lnRedshiftPrior[tmpType] = self.lnRedshiftPrior(g.truth[0]['redshift'], tmpType, total_ref_mag)

                    #Sum over template
                    for T in range(self.num_templates):
                        tmp = 0.

                        tmpType = self.responses.templates.templateType(T)
                        tmp += cache_lnTemplatePrior[tmpType]
                        tmp += cache_lnRedshiftPrior[tmpType]
                        tmp += self._fixed_lnLikelihood_flux[g.index, T]
                        tmp += magnitude_prior
                        tmp += selection_effect

                        lnProb_g = np.logaddexp(lnProb_g, tmp)
                    lnProb_all += lnProb_g
            if not np.isfinite(lnProb_all):
                return -np.inf
            else:
                return lnProb_all + calibration_prior

    def _negativeLnPriorCalibrationPosterior(self, params, photometry):
        return -1. * self._lnPriorCalibrationPosterior(params, photometry)

    def calibrate(self, photometry, cached_likelihood, mag_grid_len=10,
                  frac_tol=10.,
                  config_save_path='calibrated_prior_config.txt'):

        tolerance = frac_tol * np.finfo(float).eps
        self.mag_grid_len = mag_grid_len
        self._fixed_lnLikelihood_flux = cached_likelihood

        # Initial guesses:
        # Assume 1/3 fraction for each type
        # Assume no magnitude dependence --> every k zero
        # With no mag dependence, z0 is the peak of the redshift prior ~ mean(redshifts)
        # 1/alpha controls width of prior ~ 1 / (2 pi std(redshifts))
        redshifts = [g.truth[0]['redshift'] for g in photometry]
        init_z0 = np.mean(redshifts)
        init_a = 1. / (2. * np.pi * np.std(redshifts))
        ntype = len(self.responses.templates.possible_types)

        init_guess = np.array(list(repeat(0., ntype-1)) +
                              list(repeat(1./ntype, ntype-1)) +
                              list(repeat(init_a, ntype)) +
                              list(repeat(init_z0, ntype))+
                              list(repeat(0., ntype)))

        # Bound parameters to be positive, and fractions < 1
        param_bounds = list(repeat((0, None), ntype-1)) + \
                       list(repeat((0, 1), ntype-1)) + \
                       list(repeat((0, None), 3*ntype))

        results = minimize(self._negativeLnPriorCalibrationPosterior,
                           init_guess, method='L-BFGS-B', jac=False,
                           bounds=param_bounds, tol=tolerance, args=(photometry,),
                           options={'disp':True, 'ftol':tolerance})
        opt_params = results.x

        # Set up self with optimal parameters
        self.prior_params = opt_params
        self._loadParameterDict()
        self._calculateRedshiftPriorNorm()

        #Save out to a config file ready to load into Photoz()
        if config_save_path is not None:
            array_str = np.array2string(opt_params, separator=',', max_line_width=np.inf)[1:-1]
            cfg_str = u'[Run]\n\nprior_params = ' + array_str
            with open(config_save_path, 'w') as config_file:
                config_file.write(cfg_str)
