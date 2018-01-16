from builtins import *
from os.path import join, abspath, dirname
import warnings
try:
    #Python 2
    import ConfigParser
except ImportError:
    #Python 3
    import configparser as ConfigParser
import numpy as np

class Configuration(object):
    def __init__(self, data_config_path=None, run_config_path=None, combined_config_path=None):
        self.blendz_path = join(dirname(__file__), '..')
        self.resource_path = abspath(join(self.blendz_path, 'resources'))

        self.getConfigPaths(data_config_path, run_config_path, combined_config_path)
        self.readConfig()
        self.convertValuesFromString()
        #self.setDerivedValues()

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
        self._z_lo = self.config.getfloat('Run', 'z_lo')
        self._z_hi = self.config.getfloat('Run', 'z_hi')
        self._z_len = self.config.getint('Run', 'z_len')
        self._template_set = self.config.get('Run', 'template_set')
        self._template_set_path = self.config.get('Run', 'template_set_path')
        self.ref_mag_hi = self.config.getfloat('Run', 'ref_mag_hi')
        self.ref_mag_lo = self.config.getfloat('Run', 'ref_mag_lo')

        #Data config
        self.data_path = self.config.get('Data', 'data_path')
        self.mag_cols = [int(i) for i in self.config.get('Data', 'mag_cols').split(',')]
        self.sigma_cols = [int(i) for i in self.config.get('Data', 'sigma_cols').split(',')]
        self._ref_band = self.config.getint('Data', 'ref_band')
        #If spec_z_col is None, this gives ValueError, so set to None if it does
        try:
            self.spec_z_col = self.config.getint('Data', 'spec_z_col')
        except:
            self.spec_z_col = None
        self.filter_path = self.config.get('Data', 'filter_path')
        self.filter_file_extension = self.config.get('Data', 'filter_file_extension')
        self.filters = [f.strip() for f in self.config.get('Data', 'filters').split(',')]
        self.zero_point_errors = np.array([float(i) for i in self.config.get('Data', 'zero_point_errors').split(',')])


    #Derived attiribute -> property for ref_band and non_ref_bands indices
    @property #getter
    def ref_band(self):
        return self._ref_band
    @ref_band.setter
    def ref_band(self, value):
        self.recalculate_non_ref_bands = True
        self._ref_band = value

    @property #getter, no setter so read-only
    def non_ref_bands(self):
        try:
            recalc = self.recalculate_non_ref_bands
        except AttributeError:
            recalc = True
        if recalc:
            self._non_ref_bands = np.ones(len(self.filters), dtype=bool)
            self._non_ref_bands[self.ref_band] = False
            self.recalculate_non_ref_bands = False
        return self._non_ref_bands

    #Make the attributes controlling the redshift_grid properties so we can
    #detect if they have changed and mark redshift_grid as needing recalculation
    @property #getter
    def z_lo(self):
        return self._z_lo
    @z_lo.setter
    def z_lo(self, value):
        self.recalculate_redshift_grid = True
        self._z_lo = value

    @property #getter
    def z_hi(self):
        return self._z_hi
    @z_hi.setter
    def z_hi(self, value):
        self.recalculate_redshift_grid = True
        self._z_hi = value

    @property #getter
    def z_len(self):
        return self._z_len
    @z_len.setter
    def z_len(self, value):
        self.recalculate_redshift_grid = True
        self._z_len = value

    @property #getter, no setter so read-only
    def redshift_grid(self):
        try:
            recalc = self.recalculate_redshift_grid
        except AttributeError:
            recalc = True
        if recalc:
            self._redshift_grid = np.linspace(self.z_lo, self.z_hi, self.z_len)
            self.recalculate_redshift_grid = False
        return self._redshift_grid

    #Do the same thing for the template_dict
    #For the template_dict, main config points to an info file, which
    #is itself a configuration file, containing the path and type of each template
    @property #getter
    def template_set(self):
        return self._template_set
    @template_set.setter
    def template_set(self, value):
        self.recalculate_template_dict = True
        self._template_set = value

    @property #getter
    def template_set_path(self):
        return self._template_set_path
    @template_set_path.setter
    def template_set_path(self, value):
        self.recalculate_template_dict = True
        self._template_set_path = value

    @property #getter, no setter so read-only
    def template_dict(self):
        try:
            recalc = self.recalculate_template_dict
        except AttributeError:
            recalc = True
        if recalc:
            #Use the path and name of the template set to load in the templates
            #if we need to do it again (because one of those has changed)
            template_config = ConfigParser.SafeConfigParser()
            template_config.read(join(self.template_set_path, self.template_set))
            self._template_dict = {}
            for i, template_name in enumerate(template_config.sections()):
                rel_path_t = template_config.get(template_name, 'path')
                abs_path_t = join(self.template_set_path, rel_path_t)
                type_t = template_config.get(template_name, 'type')
                self._template_dict[i] = {'name':template_name, 'path':abs_path_t, 'type':type_t}
            #Mark as not needing recalculating (unless of the properties changes)
            self.recalculate_template_dict = False
        return self._template_dict

    def __eq__(self, other):
        '''
        Check for equality of configurations. This will return True if all the settings are the
        same in both configurations, even if they are different objects.
        '''
        if self.__class__ != other.__class__:
            return False
        else:
            all_true = True
            for key in self.__dict__:
                if key != 'config':
                    all_true *= np.all(self.__dict__[key] == other.__dict__[key])
            #return self.__dict__ == other.__dict__
            return all_true
