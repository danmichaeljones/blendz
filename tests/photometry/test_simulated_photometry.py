from builtins import *
from os.path import join
import pytest
import numpy as np
import blendz


class TestSimulatedPhotometry(object):
    def loadPhotometry(self):
        data_path = join(blendz.RESOURCE_PATH, 'config/testDataConfig.txt')
        run_path = join(blendz.RESOURCE_PATH, 'config/testRunConfig.txt')
        config = blendz.config.Configuration(config_path=[data_path, run_path])
        photo = blendz.photometry.SimulatedPhotometry(1000, config=config)
        return config, photo

    def test_all(self):
        #test_init
        config, photo = self.loadPhotometry()

        #test_iter_iterateIndex
        for i, gal in enumerate(photo):
            assert gal.index == i

        #test_iter_currentIndex(self):
        for i, gal in enumerate(photo):
            assert photo.current_galaxy.index == i

        #test_iter_currentStartsEndsNone(self):
        assert photo.current_galaxy is None
        for i, gal in enumerate(photo):
            pass
        assert photo.current_galaxy is None

        #test_iterate_currentStartsEndsNone(self):
        assert photo.current_galaxy is None
        for i, gal in enumerate(photo.iterate(start=1, stop=-1, step=2)):
            pass
        assert photo.current_galaxy is None

        #test_slice_index(self):
        for i in range(photo.num_galaxies):
            assert photo[i].index == i
            assert photo.current_galaxy is None

        #test_slice_listAndWarning(self):
        with pytest.warns(UserWarning):
            test = photo[0:2]

        #test_contextManager(self):
        for i in range(photo.num_galaxies):
            assert photo.current_galaxy is None
            with photo.galaxy(i) as gal:
                assert gal.index == i
                assert photo.current_galaxy.index == i
            assert photo.current_galaxy is None

        #test_data_not_nan
        for gal in photo:
            assert np.all(np.isfinite(gal.flux_data))
            assert np.all(np.isfinite(gal.flux_sigma))
            assert np.all(np.isfinite(gal.mag_data))
            assert np.all(np.isfinite(gal.mag_sigma))
            assert np.isfinite(gal.ref_mag_data)
            assert np.isfinite(gal.ref_mag_sigma)

        #test_data_within_priors
        for gal in photo:
            for c in range(gal.truth['num_components']):
                assert gal.truth[c]['redshift'] >= config.z_lo
                assert gal.truth[c]['redshift'] <= config.z_hi
                assert gal.truth[c]['magnitude'] >= config.ref_mag_lo
                assert gal.truth[c]['magnitude'] <= config.ref_mag_hi
                assert gal.truth[c]['template'] >= 0
                assert gal.truth[c]['template'] <= photo.responses.templates.num_templates
