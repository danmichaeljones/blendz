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
from blendz.config import DefaultConfiguration

class Configuration(DefaultConfiguration):
    def __init__(self, config_path=None, fallback_to_default=True, **kwargs):
        super(Configuration, self).__init__()
        default = DefaultConfiguration()
        default.populate_configuration()

        self.kwargs = kwargs

        if fallback_to_default:
            self.configs_to_read = default.configs_to_read
        else:
            self.configs_to_read = []
        #Add (maybe list of) user configs onto list to read
        if config_path is not None:
            if isinstance(config_path, list):
                self.configs_to_read.extend(config_path)
            else:
                self.configs_to_read.append(config_path)

        self.populate_configuration()

    def mergeFromOther(self, other_config, overwrite_default_settings=True,
                       overwrite_nondefault_settings=False):
        '''
        Merge another configuration object into this one. If a setting in the
        other object is not in this one, it is always merged. If it is, whether
        is is merged or not is controlled by `overwrite_default_settings` or
        `overwrite_nondefault_settings`, depending on whether the setting in
        this object is currently set to the default value.
        '''
        for key in other_config.__dict__:
            if key not in self.__dict__:
                self.__dict__[key] = other_config.__dict__[key]

            elif overwrite_default_settings and key in default.__dict__:
                self.__dict__[key] = other_config.__dict__[key]

            elif overwrite_nondefault_settings and key not in default.__dict__:
                self.__dict__[key] = other_config.__dict__[key]

            else:
                raise ValueError('Attempting to merge incompatible configurations.')
