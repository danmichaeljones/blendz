import numpy as np
from blendz.model import Base

class LikelihoodOnly(Base):

    def lnTemplatePrior(self, template_type):
        return 0.

    def lnRedshiftPrior(self, redshift, template_type):
        return 0.

    def correlationFunction(self, redshifts):
        return 0.
