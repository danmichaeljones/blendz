import os

PACKAGE_PATH = os.path.dirname(__file__)
RESOURCE_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'resources'))
CHAIN_PATH = os.path.abspath(os.path.join(RESOURCE_PATH, 'chains'))
DEFAULT_CONFIG_PATH = os.path.join(RESOURCE_PATH, 'config/defaultConfig.txt')
TEST_CONFIG_PATH = os.path.join(RESOURCE_PATH, 'config/testComboConfig.txt')

from blendz.config import Configuration
from blendz.photoz import Photoz
from blendz.model import BPZ, LikelihoodOnly
