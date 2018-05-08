from builtins import *
import numpy as np
from blendz.model import ModelBase

class LikelihoodOnly(ModelBase):
    def lnPrior(self, redshift, magnitude):
        return 0.
