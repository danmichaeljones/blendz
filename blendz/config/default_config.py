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
import blendz

class DefaultConfiguration(object):
    def __init__(self):
        self.configs_to_read = [blendz.DEFAULT_CONFIG_PATH]
        self.kwargs = {}

    def populate_configuration(self):
        self.readConfigFiles()
        self.saveAndConvertValues()

    def readConfigFiles(self):
        '''
        Set up the configparser and read the configuration file from disk. The
        default file is read first then (optionally) overwritten by a user
        supplied configuration.
        '''
        self.config = ConfigParser.SafeConfigParser()
        #Add the resourse_path to the ConfigParser so that it can be referenced in the config files
        self.config.set('DEFAULT', 'resource_path', blendz.RESOURCE_PATH)
        self.config.read(self.configs_to_read)

    def maybeGet(self, section, key, typeFn):
        '''
        Return a setting `key` from section `section`, where `typeFn` converts
        a string into the desired type. Settings are checked for first in the
        keyword arguments to the Configuration class, and then read from
        configuration files.
        '''
        if key in self.kwargs:
            return typeFn(self.kwargs[key])
        #Put special behaviour for if typeFn is bool, as bool() on any
        #non-empty string returns True, even if that string is "False"
        elif typeFn==bool:
            return self.config.getboolean(section, key)
        else:
            return typeFn(self.config.get(section, key))

    def maybeGetList(self, section, key, typeFn):
        '''
        Return a setting `key` from section `section`, where the setting is a
        list values of a single type, and `typeFn` converts a string into that
        desired type. Settings are checked for first in the keyword arguments
        to the Configuration class, and then read from configuration files.
        '''
        if key in self.kwargs:
            if self.kwargs[key] is None:
                return None
            else:
                try:
                    return [typeFn(v) for v in self.kwargs[key]]
                except (TypeError, ValueError):
                    return [typeFn(self.kwargs[key])]
        else:
            try:
                return [typeFn(v.strip()) for v in self.config.get(section, key).split(',')]
            except (TypeError, ValueError):
                return None

    def saveAndConvertValues(self):
        #Run config
        # We reserve np.nan for unset parameters that are ALWAYS allowed to be overwritten when merging!
        try:
            self._z_lo = self.maybeGet('Run', 'z_lo', float)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self._z_hi = self.maybeGet('Run', 'z_hi', float)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self._z_len = self.maybeGet('Run', 'z_len', int)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self._template_set = self.maybeGet('Run', 'template_set', str)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self._template_set_path = self.maybeGet('Run', 'template_set_path', str)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self.sort_redshifts = self.maybeGet('Run', 'sort_redshifts', bool)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self.ref_mag_hi = self.maybeGet('Run', 'ref_mag_hi', float)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            self.ref_mag_hi = None

        try:
            self.ref_mag_hi_sigma = self.maybeGet('Run', 'ref_mag_hi_sigma', float)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            self.ref_mag_hi_sigma = None

        try:
            self.ref_mag_lo = self.maybeGet('Run', 'ref_mag_lo', float)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self.omega_mat = self.maybeGet('Run', 'omega_mat', float)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self.omega_lam = self.maybeGet('Run', 'omega_lam', float)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self.omega_k = self.maybeGet('Run', 'omega_k', float)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self.hubble = self.maybeGet('Run', 'hubble', float)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self.r0 = self.maybeGet('Run', 'r0', float)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self.gamma = self.maybeGet('Run', 'gamma', float)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self.xi_r_cutoff = self.maybeGet('Run', 'xi_r_cutoff', float)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self.model_type = self.maybeGet('Run', 'model_type', str)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        # _prior_params is either an array of parameters,
        # or np.nan for when we want to do calibration (allow overwrite)
        try:
            _prior_params = self.maybeGetList('Run', 'prior_params', float)
            if _prior_params is None:
                self.prior_params = np.nan
            else:
                self.prior_params = np.array(_prior_params)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        #Data config
        try:
            self.data_path = self.maybeGet('Data', 'data_path', str)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self.skip_data_rows = self.maybeGet('Data', 'skip_data_rows', int)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self.data_is_csv = self.maybeGet('Data', 'data_is_csv', bool)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            #self.mag_cols = [int(i) for i in self.maybeGet('Data', 'mag_cols').split(',')]
            self.mag_cols = self.maybeGetList('Data', 'mag_cols', int)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            #self.sigma_cols = [int(i) for i in self.maybeGet('Data', 'sigma_cols').split(',')]
            self.sigma_cols = self.maybeGetList('Data', 'sigma_cols', int)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self._ref_band = self.maybeGet('Data', 'ref_band', int)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        #spec_z_col is never allowed to not be set, force it to None if not set
        #reading None can cause exceptions too, so catch them and set to None
        try:
            self.spec_z_col = self.maybeGetList('Data', 'spec_z_col', int)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            self.spec_z_col = None

        try:
            self.magnitude_limit_col = self.maybeGet('Data', 'magnitude_limit_col', int)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            self.magnitude_limit_col = None

        try:
            self.magnitude_limit = self.maybeGet('Data', 'magnitude_limit', float)
            if self.magnitude_limit_col is not None:
                warnings.warn('Both magnitude_limit and magnitude_limit_col '
                              + 'have been set, so magnitude_limit will be ignored.')
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            self.magnitude_limit = None

        try:
            self.filter_path = self.maybeGet('Data', 'filter_path', str)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self.filters = self.maybeGetList('Data', 'filters', str)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self.zero_point_errors = np.array(self.maybeGetList('Data', 'zero_point_errors', float))
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self.no_detect_value = self.maybeGet('Data', 'no_detect_value', float)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        try:
            self.no_observe_value = self.maybeGet('Data', 'no_observe_value', float)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

        #Angle within which we assume the galaxies lie, given they're blended
        #For use in correlation functions, given in arcsec, converted to rads here
        try:
            res_arcsec = self.maybeGet('Data', 'angular_resolution', float)
            self.angular_resolution = (res_arcsec / (60. * 60.)) * (np.pi / 180.)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

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
        no_check = ['config', 'configs_to_read', 'kwargs']
        if self.__class__ != other.__class__:
            return False
        else:
            all_true = True
            for key in self.__dict__:
                if key not in no_check:
                    try:
                        all_true *= np.all(self.__dict__[key] == other.__dict__[key])
                    except KeyError:
                        #KeyError means a setting in self is not in other, so we
                        #know that the configs aren't equal.
                        all_true = False
            for key in other.__dict__:
                if key not in no_check:
                    try:
                        all_true *= np.all(self.__dict__[key] == other.__dict__[key])
                    except KeyError:
                        #KeyError means a setting in self is not in other, so we
                        #know that the configs aren't equal.
                        all_true = False
            return all_true

    def __ne__(self, other):
        #https://stackoverflow.com/a/30676267
        return not self == other
