.. _set-config:

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

Pass keyword arguments to classes
------------------------------------

The configuration can be set programmatically by passing settings as
keyword arguments:

.. code:: python

.. code:: python

    pz = blendz.Photoz(data_path='path/to/data.txt',
                       mag_cols = [1, 2, 3, 4, 5],
                       sigma_cols = [6, 7, 8, 9, 10],
                       ref_band = 2,
                       filters=['sdss/u', 'sdss/g',
                                'sdss/r', 'sdss/i', 'sdss/z'])



Read in a configuration file
----------------------------

Configurations can also be read in from a file (or multiple files) by
using the ``config_path`` keyword argument.

``config_path`` should either be a string of the absolute file path to
the configuration file to be read, or a list of strings if you want to
read multiple files.

.. code:: python

    path1 = join(blendz.RESOURCE_PATH, 'config/testRunConfig.txt')
    path2 = join(blendz.RESOURCE_PATH, 'config/testDataConfig.txt')

    data = blendz.Photoz(config_path=[path1, path2])


Configuration file format
--------------------------

Configuration files are INI-style files read using the
`configparser <https://docs.python.org/3/library/configparser.html>`_
module of the standard python library. These consist of ``key = value`` pairs
separated by either a ``=`` or ``:`` separator. Whitespace around the separator is optional.

A few notes about their format:

- Configuration options *must* be separated into two (case-sensitive) sections, ``[Run]`` and ``[Data]``. An explanation of all possible configuration options, split by these sections can be found on the :ref:`config-options` page.

- Comments can be added to configuration files using ``#``

- If you want to use default settings, leave that option out of the coniguration file. Don't just leave an option blank after the ``=``/``:`` separator.

- Multiple configuration files can be loaded at once. While this provides a simple way to separate ``[Run]`` and ``[Data]`` settings (e.g., for running the same analysis on different datasets), options can be spread over different files however you want, provided that each setting is within its correct section.

An example of a configuration file (leaving some settings as default) is given below.

.. code:: ini

  [Data]

  data_path = path/to/datafile.txt
  mag_cols = 1, 3, 5, 7, 9
  sigma_cols = 2, 4, 6, 8, 10
  ref_band = 2
  filters = sdss/u, sdss/g, sdss/r, sdss/i, sdss/z
  zero_point_errors = 0.01, 0.01, 0.01, 0.01, 0.01


  [Run]

  z_hi = 2
  ref_mag_lo = 20
  ref_mag_hi = 32
  template_set = BPZ6
