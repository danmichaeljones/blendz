from builtins import *
from os.path import join
import blendz


class TestPhotoz(object):
    def loadConfig(self):
        data_path = join(blendz.RESOURCE_PATH, 'config/testDataConfig.txt')
        run_path = join(blendz.RESOURCE_PATH, 'config/testRunConfig.txt')
        test_config = blendz.config.Configuration(config_path=[data_path, run_path])
        return test_config

    def test_sample(self):
        test_config = self.loadConfig()
        pz = blendz.PhotozMag(config=test_config)
        pz.sample(1, galaxy=0)
