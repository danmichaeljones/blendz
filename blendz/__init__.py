import os

PACKAGE_PATH = os.path.dirname(__file__)
RESOURCE_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'resources'))

from .photoz import Photoz
from .photoz_mag import PhotozMag
from blendz.model import BPZ, LikelihoodOnly
