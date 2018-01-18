from builtins import *
from os.path import join
import numpy as np
import blendz


class TestConfiguration(object):
    def loadConfig(self):
        data_path = join(blendz.RESOURCE_PATH, 'config/testDataConfig.txt')
        run_path = join(blendz.RESOURCE_PATH, 'config/testRunConfig.txt')
        test_config = blendz.config.Configuration(config_path=[data_path, run_path])
        return test_config

    def loadAndMakeConfig(self):
        loaded_config = self.loadConfig()
        loaded_combo_config = blendz.config.Configuration(config_path=
                                join(blendz.RESOURCE_PATH, 'config/testComboConfig.txt'))

        made_config = blendz.config.Configuration(
                        data_path=join(blendz.RESOURCE_PATH, 'data/bpz/UDFtest.cat'),
                        mag_cols=[1, 3, 5, 7, 9, 11], sigma_cols=[2, 4, 6, 8, 10, 12],
                        spec_z_col=None, ref_band=2, filter_file_extension='.res',
                        filters=['HST_ACS_WFC_F435W', 'HST_ACS_WFC_F606W', 'HST_ACS_WFC_F775W', \
                                 'HST_ACS_WFC_F850LP', 'nic3_f110w', 'nic3_f160w'],
                        zero_point_errors = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                        ref_mag_lo = 20, ref_mag_hi = 32)


        return loaded_config, loaded_combo_config, made_config

    def test_init(self):
        for cfg in self.loadAndMakeConfig():
            test = cfg

    def test_init_empty(self):
        test_config = blendz.config.Configuration(fallback_to_default=False)

    def test_convertValuesFromString_types(self):
        for cfg in self.loadAndMakeConfig():
            assert isinstance(cfg.z_lo, float)
            assert isinstance(cfg.z_hi, float)
            assert isinstance(cfg.z_len, int)
            assert isinstance(cfg.ref_band, int)
            assert isinstance(cfg.template_set, str)
            assert isinstance(cfg.template_set_path, str)
            assert isinstance(cfg.ref_mag_lo, float)
            assert isinstance(cfg.ref_mag_hi, float)
            assert isinstance(cfg.data_path, str)
            assert isinstance(cfg.mag_cols, list)
            assert isinstance(cfg.sigma_cols, list)
            assert isinstance(cfg.ref_band, int)
            assert (isinstance(cfg.spec_z_col, int)) or (cfg.spec_z_col is None)
            assert isinstance(cfg.filter_path, str)
            assert isinstance(cfg.filter_file_extension, str)
            assert isinstance(cfg.filters, list)
            assert isinstance(cfg.zero_point_errors, np.ndarray)

    def test_eq_equal(self):
        cfg1, cfg2, _ = self.loadAndMakeConfig()
        assert cfg1 == cfg2
        assert cfg2 == cfg1

    def test_eq_notEqualSameClass(self):
        cfg1, _, cfg2 = self.loadAndMakeConfig()
        assert cfg1 != cfg2
        assert cfg2 != cfg1

    def test_eq_notEqualDifferentClass(self):
        cfg1 = self.loadConfig()
        cfg2 = [1,2,3]
        assert cfg1 != cfg2
        assert cfg2 != cfg1

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
