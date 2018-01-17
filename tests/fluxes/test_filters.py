from builtins import *
from os.path import join
import pytest
import blendz


class TestFilters(object):
    def loadFilters(self):
        data_path = join(blendz.RESOURCE_PATH, 'config/testDataConfig.txt')
        run_path = join(blendz.RESOURCE_PATH, 'config/testRunConfig.txt')
        test_config = blendz.config.Configuration(config_path=[data_path, run_path])
        test_filters = blendz.fluxes.Filters(config=test_config)
        return test_filters

    def test_init(self):
        filters = self.loadFilters()

    def test_wavelength_correctInts(self):
        filters = self.loadFilters()
        for i in range(filters.num_filters):
            filters.wavelength(i)

    def test_wavelength_wrongInts(self):
        filters = self.loadFilters()
        with pytest.raises(ValueError):
            filters.wavelength(filters.num_filters)
        with pytest.raises(ValueError):
            filters.wavelength(-1)

    def test_wavelength_wrongTypes(self):
        filters = self.loadFilters()
        with pytest.raises(ValueError):
            filters.wavelength(1.5)
        with pytest.raises(ValueError):
            filters.wavelength([1,2,3])
        with pytest.raises(ValueError):
            filters.wavelength('test')

    def test_response_correctInts(self):
        filters = self.loadFilters()
        for i in range(filters.num_filters):
            filters.response(i)

    def test_response_wrongInts(self):
        filters = self.loadFilters()
        with pytest.raises(ValueError):
            filters.response(filters.num_filters)
        with pytest.raises(ValueError):
            filters.response(-1)

    def test_response_wrongTypes(self):
        filters = self.loadFilters()
        with pytest.raises(ValueError):
            filters.response(1.5)
        with pytest.raises(ValueError):
            filters.response([1,2,3])
        with pytest.raises(ValueError):
            filters.response('test')

    def test_norm_correctInts(self):
        filters = self.loadFilters()
        for i in range(filters.num_filters):
            filters.norm(i)

    def test_norm_wrongInts(self):
        filters = self.loadFilters()
        with pytest.raises(ValueError):
            filters.norm(filters.num_filters)
        with pytest.raises(ValueError):
            filters.norm(-1)

    def test_norm_wrongTypes(self):
        filters = self.loadFilters()
        with pytest.raises(ValueError):
            filters.norm(1.5)
        with pytest.raises(ValueError):
            filters.norm([1,2,3])
        with pytest.raises(ValueError):
            filters.norm('test')

    def test_wavelengthResponse_length(self):
        filters = self.loadFilters()
        for i in range(filters.num_filters):
            assert len(filters.wavelength(i)) == len(filters.response(i))
