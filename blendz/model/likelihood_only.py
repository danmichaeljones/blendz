import numpy as np
from blendz.model import ModelBase

class LikelihoodOnly(ModelBase):

    def lnTemplatePrior(self, template_type, component_ref_mag):
        return 0.

    def lnRedshiftPrior(self, redshift, template_type, component_ref_mag):
        return 0.

    def correlationFunction(self, redshifts):
        return 0.

    def lnMagnitudePrior(self, magnitude):
        return 0.
