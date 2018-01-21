
Setting the configuration
=========================

.. code:: python

    import blendz

Classes in ``blendz`` use a ``Configuration`` object instance to manage
all of their settings. These *can* be created directly by instantiating
the class, and passed to classes that require them using the ``config``
keyword argument:

.. code:: python

    cfg = blendz.Configuration(configuration_option='setting_value')
    templates = blendz.fluxes.Templates(config=cfg)

However, constructing the configuration like this is usually not
necessary. The ``photoz`` class is designed as the only user-facing
class and handles the configuration for all of the classes it depends
on. Instead, there are two recommended ways of setting the
configuration:

Passing keyword arguments to classes
------------------------------------

The configuration can be set programmatically by passing settings as
keyword arguments:

.. code:: python

    from os.path import join

    data = blendz.Photoz(data_path=join(blendz.RESOURCE_PATH, 'data/bpz/UDFzspec.cat'),
                                        mag_cols = [22, 24, 26, 28, 30, 31],
                                        sigma_cols = [23, 25, 27, 29, 31, 33],
                                        #####################################################################
                                        #Settings in here shouldn't be necessary - remove them
                                        filter_file_extension='.res',
                                        spec_z_col=7,
                                        #####################################################################
                                        ref_band = 2,
                                        zero_point_errors = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                                        filters=['HST_ACS_WFC_F435W', 'HST_ACS_WFC_F606W', 'HST_ACS_WFC_F775W', \
                                                 'HST_ACS_WFC_F850LP', 'nic3_f110w', 'nic3_f160w'])


.. parsed-literal::

    /home/dan/anaconda2/lib/python2.7/site-packages/blendz/model/bpz.py:65: RuntimeWarning: divide by zero encountered in log
      first = (self.prior_params['alpha_t'][template_type] * np.log(redshift))


Read in a config file
---------------------

Configurations can also be read in from a file (or multiple files) by
using the ``config_path`` keyword argument.

``config_path`` should either be a string of the absolute file path to
the configuration file to be read, or a list of strings if you want to
read multiple files.

.. code:: python

    path1 = join(blendz.RESOURCE_PATH, 'config/testRunConfig.txt')
    path2 = join(blendz.RESOURCE_PATH, 'config/testDataConfig.txt')

    data = blendz.Photoz(config_path=[path1, path2])
