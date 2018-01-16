from builtins import *
from os.path import join
import blendz


class TestPhotozMag(object):
    def loadConfig(self):
        default_config = blendz.config.Configuration()
        data_path = join(default_config.resource_path, 'config/testDataConfig.txt')
        run_path = join(default_config.resource_path, 'config/testRunConfig.txt')
        test_config = blendz.config.Configuration(data_config_path=data_path,
                                                 run_config_path=run_path)
        return test_config

    def test_sample(self):
        test_config = self.loadConfig()
        pz = blendz.PhotozMag(config=test_config)
        pz.sample(1, galaxy=0)
