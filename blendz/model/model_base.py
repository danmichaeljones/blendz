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

    def _setMeasurementComponentMapping(self, num_components):
        '''
        Construct the measurement-component mapping matrix from the specification.
        The specification is the measurement_component_mapping config option,
        and is a list of length num_measurements*num_components, organised
        [C1B1, C1B2... C1Bn, C2B1...] where CnBm is a bool indicating whether
        component Cn is present in band (measurement) Bm.

        If specification is None, it is assumed that all measurements contain
        num_components components. Otherwise, specification should be a list of
        num_measurements tuples, where each tuples contains the (zero-based)
        indices of the components that measurement contains.

        If specification is given, the reference band must contain all components.
        '''

        specification = self.config.measurement_component_mapping
        num_measurements = self.responses.filters.num_filters

        #While we're here setting the number of components, check that ref_band
        #has the right length, which is either:
        # 1 - fully blended systems and partial blended systems with blended reference band
        # num_components - partial blended systems with resolved reference band
        if len(self.config.ref_band)!=1 and len(self.config.ref_band)!=num_components:
            msg = 'ref_band should either be length 1 (fully blended systems ' +\
                  'and partial blended systems with blended reference band) ' +\
                  'or length num_components={} '.format(num_components) +\
                  '(partial blended systems with every reference band measurement resolved), ' +\
                  'but ref_band currently has length {}.'.format(len(self.config.ref_band))
            raise ValueError(msg)

        #Create the mapping matrix
        if specification is not None:
            # If the spec is given in the config file, check that the number of
            # components is compatible, i.e., that the list is the right size
            if len(specification) != (num_measurements * num_components):
                msg = 'The number of components is incompatible with the ' +\
                      'measurement_component_mapping list you have set. ' +\
                      'This list needs to be num_filters*num_components ' +\
                      '= {} '.format(num_measurements * num_components) +\
                      'long, but it is currently {}'.format(len(specification)) + ' long.'

                msg += ' - DEBUG - {}'.format(specification)
                raise ValueError(msg)

            #Construct the mapping matrix from this list
            mc_map_matrix = np.zeros((num_components, num_measurements))
            i = 0
            for c in range(num_components):
                for m in range(num_measurements):
                    mc_map_matrix[c, m] = specification[i]
                    i += 1
            self.mc_map_matrix = mc_map_matrix

            #Set whether the redshifts are exchangable and so need sorting condition
            #Only need to check if there's more than one component
            #Do this by, for each row, iterating over every row after it and
            #checking whether it is equal to any other row.
            #If any True, they are exchangable
            if num_components > 1:
                redshifts_exchangeable = False
                for c in range(num_components):
                    for r in range(c+1, num_components):
                        tmp = np.all(mc_map_matrix[c, :] == mc_map_matrix[r, :])
                        redshifts_exchangeable = bool(redshifts_exchangeable + tmp)
                self.redshifts_exchangeable = redshifts_exchangeable
            else:
                self.redshifts_exchangeable = False
        else:
            self.mc_map_matrix = np.ones((num_components, num_measurements))
            self.redshifts_exchangeable = True



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
        # (depends on mc_map_matrix) and if there's
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
        #Depending on the measurement-component mapping, the galaxy ref_sigma
        #can be an array of length either 1 or num_components.
        #flux (which is ref-band flux) should be of the same length

        #TODO: No calls should be made where this isn't the case, so remove the assert
        tmp_a = len(galaxy.ref_flux_sigma)
        try:
            tmp_b = len(flux)
        except TypeError:
            flux = np.array([flux])
            tmp_b = len(flux)
        assert(tmp_a == tmp_b)

        flim = 10.**(-0.4 * galaxy.magnitude_limit)
        sigma = galaxy.ref_flux_sigma
        selection = 0.5 - (0.5 * erf((flim - flux) / (sigma * np.sqrt(2))))
        #If flux has multiple elements, the selection is a product over S,
        #so return the sum of the log
        #If it's only one element, the sum does nothing anyway
        return np.sum(np.log(selection))

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
