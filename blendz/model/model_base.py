from builtins import *
#Python 2 & 3 compatibility for abstract base classes, from
#https://stackoverflow.com/questions/35673474/
import sys
import warnings
import abc
from future.utils import with_metaclass
import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from blendz import Configuration
from blendz.fluxes import Responses

class ModelBase(with_metaclass(abc.ABCMeta)):
    def __init__(self, responses=None, config=None, **kwargs):
        #Warn user is config and responses given that config ignored
        if ((responses is not None) and (config is not None)):
            warnings.warn('A configuration was provided to Model object '
                          + 'in addition to the Responses, though these '
                          + 'should be mutually exclusive. The configuration '
                          + 'provided will be ignored.')
        #If responses given, use that
        if responses is not None:
            self.config = Configuration(**kwargs)
            self.config.mergeFromOther(responses.config)
            self.responses = responses
        #Otherwise use config to create a responses object
        else:
            self.config = Configuration(**kwargs)
            if config is not None:
                self.config.mergeFromOther(config)

            self.responses = Responses(config=self.config)

        self.prior_params = self.config.prior_params
        self.num_templates = self.responses.templates.num_templates
        self.possible_types = self.responses.templates.possible_types

        #If ref_mag_hi_sigma is not set, use ref_mag_hi instead, but prefer sigma
        if self.config.ref_mag_hi_sigma is None:
            if self.config.ref_mag_hi is not None:
                # Use fixed ref_mag_hi
                self.max_ref_mag_hi = self.config.ref_mag_hi
            else:
                # None available, so raise exception
                raise ValueError('One of ref_mag_hi or ref_mag_hi_sigma must be '
                                 + 'set in config to use the model.')
        else:
            # Using ref_mag_hi_sigma needs photometry, so set to None and
            # let Photoz() deal with setting it once model+photometry available
            self.max_ref_mag_hi = None

    def _setMeasurementComponentMapping(self, specification, num_components):
        '''
        Construct the measurement-component mapping matrix from the specification.

        If specification is None, it is assumed that all measurements contain
        num_components components. Otherwise, specification should be a list of
        num_measurements tuples, where each tuples contains the (zero-based)
        indices of the components that measurement contains.

        If specification is given, the reference band must contain all components.
        '''
        num_measurements = self.responses.filters.num_filters
        if specification is None:
            self.measurement_component_mapping = np.ones((num_components, num_measurements))
            self.redshifts_exchangeable = True
        else:
            measurement_component_mapping = np.zeros((num_components, num_measurements))
            for m in range(num_measurements):
                measurement_component_mapping[specification[m], m] = 1.

            if not np.all(measurement_component_mapping[:, self.config.ref_band] == 1.):
                #TODO: Currently enforcing the ref band to have all components. This is needed
                # to be able to specifiy the fractions (IS IT??). Also ref band is currently used in the priors,
                # though the magnitudes going to the priors either have to be in the reference band
                # *OR* on their own, in which case no separation in necessary (like original BPZ case)
                raise ValueError('The reference band must contain all components.')

            #Set the mapping
            self.measurement_component_mapping = measurement_component_mapping
            #Set whether the redshifts are exchangable and so need sorting condition
            #Only need to check if there's more than one component
            if num_components > 1:
                self.redshifts_exchangeable = np.all(measurement_component_mapping[1:, :] ==
                                                     measurement_component_mapping[:-1, :])
            else:
                self.redshifts_exchangeable = None

    def _obeyPriorConditions(self, redshifts, magnitudes, ref_mag_hi):
        '''Check that the (arrays of) redshift and magnitude have sensible
        values (within bounds set by config) and that they obey the sorting conditions.
        '''
        num_components = len(redshifts)
        redshifts_in_bounds = np.all(redshifts >= 0.) \
                              and np.all(redshifts >= self.config.z_lo) \
                              and np.all(redshifts <= self.config.z_hi)
        magnitudes_in_bounds = np.all(magnitudes >= self.config.ref_mag_lo) \
                               and np.all(magnitudes <= ref_mag_hi)
        bounds_check = redshifts_in_bounds and magnitudes_in_bounds
        #Only need sorting condition if redshifts are exchangable
        # (depends on measurement_component_mapping) and if there's
        # multiple components
        if num_components>1 and self.redshifts_exchangeable:
            if self.config.sort_redshifts:
                sort_condition = np.all(redshifts[1:] >= redshifts[:-1])
            else:
                sort_condition = np.all(magnitudes[1:] >= magnitudes[:-1])
            prior_checks_okay = sort_condition and bounds_check
        else:
            prior_checks_okay = bounds_check
        return prior_checks_okay

    def _lnTotalPrior(self, params):
        ''' Total prior given array of parameters.

        This returns

        p({z}, {t}, {m0}) = [1 + xi({z})] Prod_a  Lambda_a P(z_a | t_z, m_0a) P(t_a | m_0a) P(m_0a)

        i.e., it EXCLUDES the selection effect, and is NOT marginalised over template.

        Parameter array is given in order
        [z1, z2... t1_continuous, t2_continuous... m0_1, m0_2...] where tn_continuous
        is a continuous parameter 0:num_templates that gets rounded to an int
        for discrete template.
        '''
        num_components = int(len(params) // 3)

        redshifts = params[:num_components]
        templates_cont = params[num_components:2*num_components]
        magnitudes = params[2*num_components:]
        templates_disc = np.around(templates_cont).astype(int)

        #Prior conditions
        template_positive = np.all(templates_disc >= 0.)
        template_within_bounds = np.all(templates_disc <= self.responses.templates.num_templates - 1)
        template_okay = template_positive and template_within_bounds
        # ref_mag_hi must be set in config, which is okay since this function is
        # only used in SimulatedPhotometry which assumes that must be set
        redshift_magnitude_okay = self._obeyPriorConditions(redshifts, magnitudes, self.config.ref_mag_hi)

        if not (template_okay and redshift_magnitude_okay):
            return -np.inf
        else:
            lnPrior = np.log(1. + self.correlationFunction(redshifts))
            for a in range(num_components):
                tmp_ind_a = self.responses.templates.tmp_ind_to_type_ind[int(templates_disc[a])]
                lnPrior += self.lnPrior(redshifts[a], magnitudes[a])[tmp_ind_a]

            return lnPrior

    def comovingSeparation(self, z_lo, z_hi):
        '''Returns the comoving distance between two objects along the
        line of sight, given their redshifts.
        '''

        integral = quad(lambda zp: 1./np.sqrt((self.config.omega_mat * (1+zp)**3.) +
                                              (self.config.omega_k * (1+zp)**2.) +
                                              self.config.omega_lam), z_lo, z_hi)[0]
        return (3.e5 / self.config.hubble) * integral

    def lnSelection(self, flux, galaxy):
        flim = 10.**(-0.4 * galaxy.magnitude_limit)
        sigma = galaxy.ref_flux_sigma
        selection = 0.5 - (0.5 * erf((flim - flux) / (sigma * np.sqrt(2))))
        return np.log(selection)

    def lnPriorCalibrationPrior(self):
        '''Returns the prior on the prior parameters for the calibration procedure.'''
        #Default to P(theta) = 1 --> ln P(theta) = 0
        return 0.

    def correlationFunction(self, redshifts):
        #Default to no correlation
        return 0.

    @abc.abstractmethod
    def lnPrior(self, redshift, magnitude):
        pass
