from builtins import *
from os.path import join
import pytest
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
                        ref_mag_lo = 20, ref_mag_hi = 32, sort_redshifts=True,
                        magnitude_limit=32, z_len=500)


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
            assert isinstance(cfg.sort_redshifts, bool)
            assert isinstance(cfg.magnitude_limit, float)
            assert isinstance(cfg.omega_k, float)
            assert isinstance(cfg.omega_lam, float)
            assert isinstance(cfg.omega_mat, float)
            assert isinstance(cfg.hubble, float)

    def test_sort_redshifts(self):
        cfg1, _, cfg2 = self.loadAndMakeConfig()
        assert cfg1.sort_redshifts is False
        assert cfg2.sort_redshifts is True

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

    def test_merge(self):
        #Empty config, loaded config and made config with some difference to loaded
        empty_config = blendz.Configuration(fallback_to_default=False)
        loaded_test_config = blendz.Configuration(config_path=join(blendz.RESOURCE_PATH,
                                                        'config/testComboConfig.txt'))
        made_config1 = blendz.config.Configuration(
                        data_path=join(blendz.RESOURCE_PATH, 'data/bpz/UDFtest.cat'),
                        mag_cols=[1, 3, 5, 7, 9, 11], sigma_cols=[2, 4, 6, 8, 10, 12],
                        ref_band=2, filter_file_extension='.res',
                        filters=['HST_ACS_WFC_F435W', 'HST_ACS_WFC_F606W', 'HST_ACS_WFC_F775W', \
                                 'HST_ACS_WFC_F850LP', 'nic3_f110w', 'nic3_f160w'],
                        zero_point_errors = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                        ref_mag_lo = 20, ref_mag_hi = 32, sort_redshifts=True,
                        magnitude_limit=32, template_set='BPZ6')

        made_config2 = blendz.config.Configuration(
                        data_path=join(blendz.RESOURCE_PATH, 'data/bpz/UDFtest.cat'),
                        mag_cols=[1, 3, 5, 7, 9, 11], sigma_cols=[2, 4, 6, 8, 10, 12],
                        ref_band=2, filter_file_extension='.res',
                        filters=['HST_ACS_WFC_F435W', 'HST_ACS_WFC_F606W', 'HST_ACS_WFC_F775W', \
                                 'HST_ACS_WFC_F850LP', 'nic3_f110w', 'nic3_f160w'],
                        zero_point_errors = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                        spec_z_col=10, ref_mag_lo = 20, ref_mag_hi = 32, sort_redshifts=True,
                        magnitude_limit=32, template_set='BPZ6')

        made_config3 = blendz.config.Configuration(
                        data_path=join(blendz.RESOURCE_PATH, 'data/bpz/UDFtest.cat'),
                        mag_cols=[1, 3, 5, 7, 9, 11], sigma_cols=[2, 4, 6, 8, 10, 12],
                        ref_band=2, filter_file_extension='.res',
                        filters=['HST_ACS_WFC_F435W', 'HST_ACS_WFC_F606W', 'HST_ACS_WFC_F775W', \
                                 'HST_ACS_WFC_F850LP', 'nic3_f110w', 'nic3_f160w'],
                        zero_point_errors = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                        spec_z_col=10, ref_mag_lo = 20, ref_mag_hi = 32, sort_redshifts=True,
                        magnitude_limit=32, template_set='BPZ6')


        #Setting not preset here but is in other = merge in
        with pytest.raises(AttributeError):
            test1 = empty_config.ref_mag_lo + 1
        test2 = loaded_test_config.ref_mag_lo + 1
        empty_config.mergeFromOther(loaded_test_config)
        test3 = empty_config.ref_mag_lo + 1
        test4 = loaded_test_config.ref_mag_lo + 1
        # ... and mark it as a default if it was a default in other_config
        #ref_band wasn't a default
        assert not loaded_test_config.keyIsDefault('ref_band')
        assert not empty_config.keyIsDefault('ref_band')
        assert loaded_test_config.ref_band == empty_config.ref_band
        #but filter_path was
        assert loaded_test_config.keyIsDefault('filter_path')
        assert empty_config.keyIsDefault('filter_path')


        #Setting here == setting in other = no error, just make sure equal after
        assert loaded_test_config != made_config1
        assert np.all(loaded_test_config.mag_cols == made_config1.mag_cols)
        #Setting in other = default, setting in here not default = no merge, no error
        assert loaded_test_config.template_set != made_config1.template_set
        assert loaded_test_config.template_set == 'BPZ8'
        assert loaded_test_config.default.template_set == 'BPZ8'
        assert loaded_test_config.keyIsDefault('template_set')
        assert not made_config1.keyIsDefault('template_set')
        assert made_config1.template_set == 'BPZ6'
        #Default setting here, non-default in other and allowed to overwrite it = merge in
        #Test both spec_z_col...
        assert made_config1.spec_z_col != loaded_test_config.spec_z_col
        assert not loaded_test_config.keyIsDefault('spec_z_col')
        assert made_config1.keyIsDefault('spec_z_col')
        assert made_config1.spec_z_col is None
        assert loaded_test_config.spec_z_col == 13
        #...and z_len
        assert made_config1.z_len != loaded_test_config.z_len
        assert not loaded_test_config.keyIsDefault('z_len')
        assert made_config1.keyIsDefault('z_len')
        assert made_config1.z_len == 1000
        assert loaded_test_config.z_len == 500

        #Now do the merge and check again
        made_config1.mergeFromOther(loaded_test_config)
        assert np.all(loaded_test_config.mag_cols == made_config1.mag_cols)
        assert made_config1.template_set == 'BPZ6'
        assert made_config1.z_len == 500
        assert made_config1.spec_z_col == 13

        #Non-default setting here, non-default in other and not allowed to overwrite it = error
        assert made_config2.spec_z_col == 10
        assert not made_config2.keyIsDefault('spec_z_col')
        assert loaded_test_config.spec_z_col == 13
        assert not loaded_test_config.keyIsDefault('spec_z_col')
        with pytest.raises(ValueError):
            made_config2.mergeFromOther(loaded_test_config)

        #Non-default setting here, non-default in other and allowed to overwrite it = merge in
        assert made_config3.spec_z_col == 10
        assert not made_config3.keyIsDefault('spec_z_col')
        assert loaded_test_config.spec_z_col == 13
        assert not loaded_test_config.keyIsDefault('spec_z_col')
        made_config3.mergeFromOther(loaded_test_config, overwrite_any_setting=True)

        #Different defaults = error on merge
        existing_default_path = blendz.DEFAULT_CONFIG_PATH
        cfg_before_change_default = blendz.Configuration()
        blendz.DEFAULT_CONFIG_PATH = '/Path/To/Nowhere.txt'
        cfg_after_change_default = blendz.Configuration()
        with pytest.raises(ValueError):
            cfg_after_change_default.mergeFromOther(cfg_before_change_default)
        #Put the default path back
        blendz.DEFAULT_CONFIG_PATH = existing_default_path
