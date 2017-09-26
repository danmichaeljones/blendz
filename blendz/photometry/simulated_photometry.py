import numpy as np
from blendz.config import _config
from blendz.photometry import PhotometryBase, Galaxy

class SimulatedPhotometry(PhotometryBase):
    def __init__(self, num_sims, responses=None, zero_point_errors=_config.zero_point_errors):
        super(SimulatedPhotometry, self).__init__()

        self.num_sims = num_sims
        self.zero_point_errors = zero_point_errors
        self.zero_point_frac = 10.**(0.4*self.zero_point_errors) - 1.
        if responses is None:
            self.responses = Responses()
        else:
            self.responses = responses

        self.simulateGalaxies(self.num_sims)

    def generateGalaxyData(self, redshift, scale, template_index, err_frac):
        true_flux = self.responses(template_index, None, redshift) * scale
        rand_err  = np.array([(np.random.rand() * (true_flux[i] * err_frac * 2)) - (true_flux[i] * err_frac) for i in xrange(len(true_flux))])
        obs_flux = true_flux + rand_err
        flux_err = true_flux * err_frac
        obs_mag = np.log10(obs_flux) / (-0.4)
        mag_err = np.log10((flux_err/obs_flux)+1.) / (-0.4)
        return obs_mag, mag_err

    def randomGalaxy(self, max_redshift, max_scale, max_err_frac):
        sim_redshift = np.random.rand() * max_redshift
        sim_scale = np.random.rand() * max_scale
        sim_template = np.random.randint(0, self.responses.templates.num_templates)
        sim_err_frac = np.random.rand() * max_err_frac

        obs_mag, mag_err = self.generateGalaxyData(sim_redshift, sim_scale, sim_template, sim_err_frac)
        truth = {'redshift': sim_redshift, 'scale': sim_scale, 'template': sim_template}

        return obs_mag, mag_err, truth

    def simulateGalaxies(self, num_sims, max_redshift=6., max_scale=50., max_err_frac=0.1):
        for g in xrange(num_sims):
            mag_data, mag_sigma, truth = self.randomGalaxy(max_redshift, max_scale, max_err_frac)
            new_galaxy = Galaxy(mag_data, mag_sigma, _config.ref_band, self.zero_point_frac, g)
            new_galaxy.truth = truth
            self.galaxies.append(new_galaxy)
