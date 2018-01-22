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


'''
Config class description text.
'''

class Configuration(DefaultConfiguration):
    '''
    Container for every setting.
    '''

    def __init__(self, config_path=None, fallback_to_default=True, **kwargs):
        super(Configuration, self).__init__()
        self.default = DefaultConfiguration()
        self.default.populate_configuration()

        self.kwargs = kwargs

        if fallback_to_default:
            self.configs_to_read = self.default.configs_to_read
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
                       overwrite_any_setting=False):
        '''
        Safely merge another configuration object into this one.

        If a setting in the other object is not in this one, it is always merged.
        If it is, whether is is merged or not is controlled by
        `overwrite_default_settings` or `overwrite_nondefault_settings`,
        depending on whether the setting in this object is currently set to
        the default value. If a setting is in both objects but the other is a
        default, a merge is not attempted.
        '''
        no_compare = ['config', 'configs_to_read', 'kwargs', 'default']
        if self.default != other_config.default:
            raise ValueError('Attempting to merge incompatible configurations - Different defaults')

        for key in other_config.__dict__:
            if key not in no_compare:
                #Setting not here, merge in
                if key not in self.__dict__:
                    self.__dict__[key] = other_config.__dict__[key]
                    # ... and mark it as a default if it was a default in other_config
                    if key in other_config.default.__dict__:
                        self.default.__dict__[key] = other_config.default.__dict__[key]
                #If they're already equal it doesn't matter
                elif self.__dict__[key] = other_config.__dict__[key]:
                    continue
                #Prevent error if other_config is a default setting and is
                #different to one already here (don't overwrite with defaults)
                elif key in other_config.default.__dict__:
                    continue
                #Default setting here and allowed to overwrite it
                elif overwrite_default_settings and key in self.default.__dict__:
                    self.__dict__[key] = other_config.__dict__[key]
                #Allowed to overwrite any setting
                elif overwrite_any_setting:
                    self.__dict__[key] = other_config.__dict__[key]
                else:
                    raise ValueError('Attempting to merge incompatible configurations - ' + key)
