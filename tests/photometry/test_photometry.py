from builtins import *
from os.path import join
import blendz


class TestPhotometry(object):
    def loadPhotometry(self):
        data_path = join(blendz.RESOURCE_PATH, 'config/testDataConfig.txt')
        run_path = join(blendz.RESOURCE_PATH, 'config/testRunConfig.txt')
        test_config = blendz.config.Configuration(config_path=[data_path, run_path])
        photo = blendz.photometry.Photometry(config=test_config)
        return photo

    def test_init(self):
        photo = self.loadPhotometry()

    def test_iter_iterateIndex(self):
        photo = self.loadPhotometry()
        for i, gal in enumerate(photo):
            assert gal.index == i

    def test_iter_currentIndex(self):
        photo = self.loadPhotometry()
        for i, gal in enumerate(photo):
            assert photo.current_galaxy.index == i

    def test_iter_currentStartsEndsNone(self):
        photo = self.loadPhotometry()
        assert photo.current_galaxy is None
        for i, gal in enumerate(photo):
            pass
        assert photo.current_galaxy is None

    def test_iterate_currentStartsEndsNone(self):
        photo = self.loadPhotometry()
        assert photo.current_galaxy is None
        for i, gal in enumerate(photo.iterate(start=1, stop=-1, step=2)):
            pass
        assert photo.current_galaxy is None
