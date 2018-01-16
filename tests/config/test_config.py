from builtins import *
from os.path import join
import blendz


class TestConfiguration(object):
    def loadConfig(self):
        default_config = blendz.config.Configuration()
        data_path = join(default_config.resource_path, 'config/testDataConfig.txt')
        run_path = join(default_config.resource_path, 'config/testRunConfig.txt')
        test_config = blendz.config.Configuration(data_config_path=data_path,
                                                 run_config_path=run_path)
        return test_config

    def test_init(self):
        test_config = self.loadConfig()

    def test_convertValuesFromString_types(self):
        #TODO : FINISH THIS TEST, AND WORK OUT WHY STRING CHECK BROKEN!
        cfg = self.loadConfig()
        assert isinstance(cfg.z_lo, float)
        assert isinstance(cfg.z_hi, float)
        assert isinstance(cfg.z_len, int)
        assert isinstance(cfg.ref_band, int)
        assert isinstance(cfg.ref_mag_lo, float)
        assert isinstance(cfg.ref_mag_hi, float)
        #assert isinstance(cfg.data_path, str)
        #assert isinstance(cfg.filter_path, str)
        assert isinstance(cfg.mag_cols, list)
        assert isinstance(cfg.sigma_cols, list)
        assert isinstance(cfg.ref_band, int)
        assert (isinstance(cfg.spec_z_col, int)) or (isinstance(cfg.spec_z_col, None))

    def test_eq_equal(self):
        cfg1 = self.loadConfig()
        cfg2 = self.loadConfig()
        assert cfg1 == cfg2

    def test_eq_notEqual(self):
        cfg1 = self.loadConfig()
        cfg2 = self.loadConfig()
        cfg2.z_len = cfg1.z_len + 1
        assert cfg1 != cfg2
