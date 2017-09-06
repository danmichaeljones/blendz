import numpy as np
import os
import warnings
try:
    #Python 2
    import ConfigParser
except ImportError:
    #Python 3
    import configparser as ConfigParser

class BlendzConfig(object):
    def __init__(self, data_config_path=None, run_config_path=None, combined_config_path=None):
        self.blendz_path = os.path.dirname(__file__)
        self.resource_path = os.path.abspath(os.path.join(self.blendz_path, 'resources'))

        self.getConfigPaths(data_config_path, run_config_path, combined_config_path)
        self.readConfig()
        self.convertValuesFromString()
        self.setDerivedValues()

    def getConfigPaths(self, data_config_path=None, run_config_path=None, combined_config_path=None):
        #Allow for either one combined config file, or split into run settings and data settings
        if combined_config_path is not None:
            #Warn user  if they try to set all three config paths - only combined is used in that case
            if (run_config_path is not None) or (data_config_path is not None):
                warnings.warn('Reading from the combined configuration file only. \
                               Ignoring the run and data configuration files, even \
                               though at least one of them has been set.')

            self.combined_config_path = combined_config_path
            self.configs_to_read = [self.combined_config_path]
        #Two-config-files case
        else:
            self.combined_config_path = combined_config_path
            if run_config_path is None:
                self.run_config_path = os.path.join(self.resource_path, 'config/defaultRunConfig.txt')
            else:
                self.run_config_path = run_config_path
            if data_config_path is None:
                self.data_config_path = os.path.join(self.resource_path, 'config/defaultDataConfig.txt')
            else:
                self.data_config_path = data_config_path
            self.configs_to_read = [self.run_config_path, self.data_config_path]

    def readConfig(self):
        self.config = ConfigParser.SafeConfigParser()
        #Add the resourse_path to the ConfigParser so that it can be referenced in the config files
        self.config.set('DEFAULT', 'resource_path', self.resource_path)
        self.config.read(self.configs_to_read)

    def convertValuesFromString(self):
        #Run config
        self.z_lo = self.config.getfloat('Run', 'z_lo')
        self.z_hi = self.config.getfloat('Run', 'z_hi')
        self.z_len = self.config.getint('Run', 'z_len')
        #Data config
        self.data_path = self.config.get('Data', 'data_path')
        self.mag_cols = [int(i) for i in self.config.get('Data', 'mag_cols').split(',')]
        self.sigma_cols = [int(i) for i in self.config.get('Data', 'sigma_cols').split(',')]
        self.ref_mag = self.config.getint('Data', 'ref_mag')

    def setDerivedValues(self):
        self.redshift_grid = np.linspace(self.z_lo, self.z_hi, self.z_len)
