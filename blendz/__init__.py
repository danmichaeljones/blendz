import os

PACKAGE_PATH = os.path.dirname(__file__)
RESOURCE_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'resources'))
DEFAULT_CONFIG_PATH = os.path.join(RESOURCE_PATH, 'config/defaultConfig.txt')
TEST_CONFIG_PATH = os.path.join(RESOURCE_PATH, 'config/testComboConfig.txt')

from .config import Configuration
from .photoz import Photoz
from blendz.model import BPZ, LikelihoodOnly
