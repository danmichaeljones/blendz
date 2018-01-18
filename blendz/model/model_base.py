from builtins import *
#Python 2 & 3 compatibility for abstract base classes, from
#https://stackoverflow.com/questions/35673474/
import sys
import abc
if sys.version_info >= (3, 4):
    ABC_meta = abc.ABC
else:
    ABC_meta = abc.ABCMeta('ABC', (), {})
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
        #Otherwise use config to create a responses object,
        #loading from default if no config given
        else:
            if config is not None:
                self.config = config
            else:
                self.config = Configuration()
            self.config.update(kwargs)
            self.responses = Responses(config=self.config)

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
