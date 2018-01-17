from builtins import *
from os.path import join
import numpy as np
import pytest
import blendz


class TestTemplates(object):
    def loadTemplates(self):
        default_config = blendz.config.Configuration()
        data_path = join(default_config.resource_path, 'config/testDataConfig.txt')
        run_path = join(default_config.resource_path, 'config/testRunConfig.txt')
        test_config = blendz.config.Configuration(config_path=[data_path, run_path])
        test_templates = blendz.fluxes.Templates(config=test_config)
        return test_templates

    def test_init(self):
        templates = self.loadTemplates()

    def test_wavelength_correctInts(self):
        templates = self.loadTemplates()
        for i in range(templates.num_templates):
            templates.wavelength(i)

    def test_wavelength_wrongInts(self):
        templates = self.loadTemplates()
        with pytest.raises(ValueError):
            templates.wavelength(templates.num_templates)
        with pytest.raises(ValueError):
            templates.wavelength(-1)

    def test_wavelength_wrongTypes(self):
        templates = self.loadTemplates()
        with pytest.raises(ValueError):
            templates.wavelength(1.5)
        with pytest.raises(ValueError):
            templates.wavelength([1,2,3])
        with pytest.raises(ValueError):
            templates.wavelength('test')

    def test_flux_correctInts(self):
        templates = self.loadTemplates()
        for i in range(templates.num_templates):
            templates.flux(i)

    def test_flux_wrongInts(self):
        templates = self.loadTemplates()
        with pytest.raises(ValueError):
            templates.flux(templates.num_templates)
        with pytest.raises(ValueError):
            templates.flux(-1)

    def test_flux_wrongTypes(self):
        templates = self.loadTemplates()
        with pytest.raises(ValueError):
            templates.flux(1.5)
        with pytest.raises(ValueError):
            templates.flux([1,2,3])
        with pytest.raises(ValueError):
            templates.flux('test')

    def test_name_correctInts(self):
        templates = self.loadTemplates()
        for i in range(templates.num_templates):
            templates.name(i)

    def test_name_wrongInts(self):
        templates = self.loadTemplates()
        with pytest.raises(ValueError):
            templates.name(templates.num_templates)
        with pytest.raises(ValueError):
            templates.name(-1)

    def test_name_wrongTypes(self):
        templates = self.loadTemplates()
        with pytest.raises(ValueError):
            templates.name(1.5)
        with pytest.raises(ValueError):
            templates.name([1,2,3])
        with pytest.raises(ValueError):
            templates.name('test')

    def test_numType_correctTempTypes(self):
        templates = self.loadTemplates()
        for T in templates.possible_types:
            templates.numType(T)

    def test_numType_wrongTempTypes(self):
        templates = self.loadTemplates()
        with pytest.warns(UserWarning):
            templates.numType(1)
        with pytest.warns(UserWarning):
            templates.numType('test')
        with pytest.raises(TypeError):
            templates.numType([1, 2, 3])

    def test_templateType_correctInts(self):
        templates = self.loadTemplates()
        for i in range(templates.num_templates):
            templates.templateType(i)

    def test_templateType_wrongInts(self):
        templates = self.loadTemplates()
        with pytest.raises(ValueError):
            templates.templateType(templates.num_templates)
        with pytest.raises(ValueError):
            templates.templateType(-1)

    def test_templateType_wrongTypes(self):
        templates = self.loadTemplates()
        with pytest.raises(ValueError):
            templates.templateType(1.5)
        with pytest.raises(ValueError):
            templates.templateType([1,2,3])
        with pytest.raises(ValueError):
            templates.templateType('test')

    def test_interp_correctInts(self):
        templates = self.loadTemplates()
        for i in range(templates.num_templates):
            templates.interp(i, templates.wavelength(i)*1.1)

    def test_interp_arrayInterpolate(self):
        templates = self.loadTemplates()
        for i in range(templates.num_templates):
            templates.interp(i, np.linspace(templates.wavelength(i)[0]*1.1, templates.wavelength(i)[-1]*0.9, 10))

    def test_interp_arrayExtrapolate(self):
        templates = self.loadTemplates()
        for i in range(templates.num_templates):
            templates.interp(i, np.linspace(templates.wavelength(i)[0]*0.9, templates.wavelength(i)[-1]*1.1, 10))

    def test_interp_wrongInts(self):
        templates = self.loadTemplates()
        with pytest.raises(ValueError):
            templates.interp(templates.num_templates, templates.wavelength(0)*1.1)
        with pytest.raises(ValueError):
            templates.interp(-1, templates.wavelength(0)*1.1)

    def test_interp_wrongTypes(self):
        templates = self.loadTemplates()
        with pytest.raises(ValueError):
            templates.interp(1.5, templates.wavelength(0)*1.1)
        with pytest.raises(ValueError):
            templates.interp([1,2,3], templates.wavelength(0)*1.1)
        with pytest.raises(ValueError):
            templates.interp('test', templates.wavelength(0)*1.1)

    def test_wavelengthFlux_length(self):
        templates = self.loadTemplates()
        for i in range(templates.num_templates):
            assert len(templates.wavelength(i)) == len(templates.flux(i))
