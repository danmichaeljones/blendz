from builtins import *
from os.path import join
import numpy as np
import blendz


class TestPhotoz(object):
    def loadConfig(self):
        data_path = join(blendz.RESOURCE_PATH, 'config/testDataConfig.txt')
        run_path = join(blendz.RESOURCE_PATH, 'config/testRunConfig.txt')
        test_config = blendz.config.Configuration(config_path=[data_path, run_path])
        return test_config

    def checkPhotometry(self, pz1, pz2):
        #Check photometry is equal
        for g in pz1.photometry:
            assert np.all(g.flux_data==pz2.photometry[g.index].flux_data)
            assert np.all(g.flux_sigma==pz2.photometry[g.index].flux_sigma)
            assert np.all(g.ref_flux_data==pz2.photometry[g.index].ref_flux_data)
            assert np.all(g.ref_flux_sigma==pz2.photometry[g.index].ref_flux_sigma)
            assert np.all(g.flux_data_noRef==pz2.photometry[g.index].flux_data_noRef)
            assert np.all(g.flux_sigma_noRef==pz2.photometry[g.index].flux_sigma_noRef)
            assert np.all(g.zero_point_frac==pz2.photometry[g.index].zero_point_frac)
            assert np.all(g.ref_band==pz2.photometry[g.index].ref_band)
            assert np.all(g.ref_mag_data==pz2.photometry[g.index].ref_mag_data)
            assert np.all(g.ref_mag_sigma==pz2.photometry[g.index].ref_mag_sigma)

    def checkFilters(self, pz1, pz2):
        #Check filters are equal
        assert pz1.responses.filters.num_filters==pz2.responses.filters.num_filters
        assert np.all(pz1.responses.filters.filter_names==pz2.responses.filters.filter_names)
        for f in range(pz1.responses.filters.num_filters):
            assert np.all(pz1.responses.filters.response(f)==pz2.responses.filters.response(f))
            assert np.all(pz1.responses.filters.wavelength(f)==pz2.responses.filters.wavelength(f))
            assert np.all(pz1.responses.filters.norm(f)==pz2.responses.filters.norm(f))

    def checkTemplates(self, pz1, pz2):
        #Check filters are equal
        assert pz1.responses.templates.num_templates==pz2.responses.templates.num_templates
        assert np.all(pz1.responses.templates.possible_types==pz2.responses.templates.possible_types)
        for f in range(pz1.responses.templates.num_templates):
            assert np.all(pz1.responses.templates.templateType(f)==pz2.responses.templates.templateType(f))
            assert np.all(pz1.responses.templates.wavelength(f)==pz2.responses.templates.wavelength(f))
            assert np.all(pz1.responses.templates.flux(f)==pz2.responses.templates.flux(f))
            assert np.all(pz1.responses.templates.name(f)==pz2.responses.templates.name(f))

    def checkPhotoz(self, pz1, pz2, done_sample=False):
        assert pz1.config==pz2.config
        self.checkPhotometry(pz1, pz2)
        if done_sample:
            assert np.all(pz1.chain(1, galaxy=0)==pz2.chain(1, galaxy=0))
        self.checkFilters(pz1, pz2)
        self.checkTemplates(pz1, pz2)

    def test_all(self):
        test_config = self.loadConfig()
        pz = blendz.Photoz(config=test_config)

        #Test save/load before sampling
        pz.saveState('testPZ_no_samples.pkl')
        pz_load1 = blendz.Photoz(load_state_path='testPZ_no_samples.pkl')
        self.checkPhotoz(pz, pz_load1)

        #Test single-component sampling all galaxies with default sampler
        pz.sample(1)

        #Check that the max-a-post, mean and stan-devs of all parameters for all galaxies are okay
        assert np.all(np.isfinite(pz.max(1)))
        assert np.all(np.isfinite(pz.mean(1)))
        assert np.all(np.isfinite(pz.std(1)))

        #Test save/load after sampling single component
        pz.saveState('testPZ_one_component_samples.pkl')
        pz_load2 = blendz.Photoz(load_state_path='testPZ_one_component_samples.pkl')
        self.checkPhotoz(pz, pz_load2, done_sample=True)

        #Test single-component sampling of particular galaxy with nestle sampler
        pz.sample(1, galaxy=0, use_pymultinest=False)

        #Test save/load after sampling single component with nestle
        pz.saveState('testPZ_one_component_samples_nestle.pkl')
        pz_load3 = blendz.Photoz(load_state_path='testPZ_one_component_samples_nestle.pkl')
        self.checkPhotoz(pz, pz_load3, done_sample=True)
