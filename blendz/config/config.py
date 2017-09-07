from os.path import join, abspath, dirname
import warnings
try:
    #Python 2
    import ConfigParser
except ImportError:
    #Python 3
    import configparser as ConfigParser
import numpy as np

class BlendzConfig(object):
    def __init__(self, data_config_path=None, run_config_path=None, combined_config_path=None):
        self.blendz_path = join(dirname(__file__), '..')
        self.resource_path = abspath(join(self.blendz_path, 'resources'))

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
        #Two-config-files case - If nothing is set, we use the two default config files
        else:
            self.combined_config_path = combined_config_path
            if run_config_path is None:
                self.run_config_path = join(self.resource_path, 'config/defaultRunConfig.txt')
            else:
                self.run_config_path = run_config_path
            if data_config_path is None:
                self.data_config_path = join(self.resource_path, 'config/defaultDataConfig.txt')
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
        self.template_set = self.config.get('Run', 'template_set')
        self.template_set_path = self.config.get('Run', 'template_set_path')

        self.template_path = None
        #Data config
        self.data_path = self.config.get('Data', 'data_path')
        self.mag_cols = [int(i) for i in self.config.get('Data', 'mag_cols').split(',')]
        self.sigma_cols = [int(i) for i in self.config.get('Data', 'sigma_cols').split(',')]
        self.ref_mag = self.config.getint('Data', 'ref_mag')
        self.filter_path = self.config.get('Data', 'filter_path')
        self.filter_file_extension = self.config.get('Data', 'filter_file_extension')
        self.filters = [f.strip() for f in self.config.get('Data', 'filters').split(',')]
        self.zero_point_errors = np.array([float(i) for i in self.config.get('Data', 'mag_cols').split(',')])

    def setDerivedValues(self):
        self.redshift_grid = np.linspace(self.z_lo, self.z_hi, self.z_len)
        #Templates are a little different - main config points to an info file, which
        #is itself a configuration file, containing the path and type of each template
        self.template_config = ConfigParser.SafeConfigParser()
        self.template_config.read(join(self.template_set_path, self.template_set))
        self.template_dict = {}
        for i, template_name in enumerate(self.template_config.sections()):
            rel_path_t = self.template_config.get(template_name, 'path')
            abs_path_t = join(self.template_set_path, rel_path_t)
            type_t = self.template_config.get(template_name, 'type')
            self.template_dict[i] = {'name':template_name, 'path':abs_path_t, 'type':type_t}
