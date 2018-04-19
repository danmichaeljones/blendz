from builtins import *
import warnings
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

    def keyIsDefault(self, key):
        '''Returns True if the current value of the setting ``key`` is default value'''
        if (key in self.__dict__) and (key in self.default.__dict__) and \
                np.all(self.__dict__[key]==self.default.__dict__[key]):
            return True
        else:
            #Account for hidden properties
            key_mod = '_' + key
            if (key_mod in self.__dict__) and (key_mod in self.default.__dict__) and \
                    np.all(self.__dict__[key_mod]==self.default.__dict__[key_mod]):
                return True
            #Otherwise, not default
            else:
                return False

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
            warnings.warn('Merging configurations with different defaults.')

        for key in other_config.__dict__:
            if key not in no_compare:
                done_set = False
                #Setting not here, merge in
                if not done_set and key not in self.__dict__:
                    self.__dict__[key] = other_config.__dict__[key]
                    done_set = True
                    # ... and mark it as a default if it was a default in other_config
                    if other_config.keyIsDefault(key):
                        self.default.__dict__[key] = other_config.default.__dict__[key]

                #If they're already equal it doesn't matter
                if not done_set and np.all(self.__dict__[key] == other_config.__dict__[key]):
                    done_set = True
                    continue

                #Prevent error if other_config is a default setting and is
                #different to one already here (don't overwrite with defaults)
                if not done_set and other_config.keyIsDefault(key):
                    done_set = True
                    continue

                #Default setting here and allowed to overwrite it
                if not done_set and overwrite_default_settings and self.keyIsDefault(key):
                    self.__dict__[key] = other_config.__dict__[key]
                    done_set = True

                #Allowed to overwrite any setting
                if not done_set and overwrite_any_setting:
                    self.__dict__[key] = other_config.__dict__[key]
                    done_set = True

                #Setting here is nan, which designates it as always okay to overwrite
                if not done_set and self.__dict__[key] is np.nan:
                    self.__dict__[key] = other_config.__dict__[key]
                    done_set = True

                #Setting in other is nan, never overwrite with it
                if not done_set and other_config.__dict__[key] is np.nan:
                    done_set = True

                if not done_set:
                    raise ValueError('Attempting to merge incompatible configurations - ' + key)
