from builtins import *
from os.path import join
import numpy as np
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

    def test_eq_notEqualSameClass(self):
        cfg1 = self.loadConfig()
        cfg2 = self.loadConfig()
        cfg2.z_len = cfg1.z_len + 1
        assert cfg1 != cfg2

    def test_eq_notEqualDifferentClass(self):
        cfg1 = self.loadConfig()
        cfg2 = [1,2,3]
        assert cfg1 != cfg2

    def test_redshift_grid_start(self):
        cfg = self.loadConfig()
        grid = cfg.redshift_grid
        assert np.all(grid == np.linspace(cfg.z_lo, cfg.z_hi, cfg.z_len))

    def test_redshift_grid_change(self):
        cfg = self.loadConfig()
        cfg.z_lo = cfg.z_lo + 1
        cfg.z_hi = cfg.z_hi + 1
        cfg.z_len = cfg.z_len + 1

        grid = cfg.redshift_grid
        assert np.all(grid == np.linspace(cfg.z_lo, cfg.z_hi, cfg.z_len))
