from builtins import *
#Python 2 & 3 compatibility for abstract base classes, from
#https://stackoverflow.com/questions/35673474/
import sys
import abc
if sys.version_info >= (3, 4):
    ABC_meta = abc.ABC
else:
    ABC_meta = abc.ABCMeta('ABC', (), {})
import numpy as np
from blendz import Configuration
from blendz.fluxes import Responses

class ModelBase(ABC_meta):
    def __init__(self, responses=None, config=None, **kwargs):
        #Warn user is config and responses given that config ignored
        if ((responses is not None) and (config is not None)):
            warnings.warn("""A configuration was provided to Model object
                            in addition to the Responses, though these
                            should be mutually exclusive. The configuration
                            provided will be ignored.""")
        #If responses given, use that
        if responses is not None:
            self.responses = responses
            self.config = self.responses.config
        #Otherwise use config to create a responses object
        else:
            self.config = Configuration(**kwargs)
            if config is not None:
                self.config.mergeFromOther(config)

            self.responses = Responses(config=self.config)

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

    def _obeyPriorConditions(self, redshifts, magnitudes):
        '''Check that the (arrays of) redshift and magnitude have sensible
        values and that they obey the sorting conditions.
        '''
        num_components = len(redshifts)
        redshift_positive = np.all(redshifts >= 0.)
        #Only need sorting condition if redshifts are exchangable
        # (depends on measurement_component_mapping) and if there's
        # multiple components
        if num_components>1 and self.redshifts_exchangeable:
            if self.config.sort_redshifts:
                sort_condition = np.all(redshifts[1:] >= redshifts[:-1])
            else:
                sort_condition = np.all(magnitudes[1:] >= magnitudes[:-1])
            prior_checks_okay = sort_condition and redshift_positive
        else:
            prior_checks_okay = redshift_positive
        return prior_checks_okay

    @abc.abstractmethod
    def correlationFunction(self, redshifts):
        pass

    @abc.abstractmethod
    def lnTemplatePrior(self, template_type, component_ref_mag):
        pass

    @abc.abstractmethod
    def lnRedshiftPrior(self, redshift, template_type, component_ref_mag):
        pass

    @abc.abstractmethod
    def lnMagnitudePrior(self, magnitude):
        pass
