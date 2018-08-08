.. _calibrate:

Prior calibration
=================

If the prior parameters are not set manually by :ref:`set-config`, they can be
estimated using a set of calibration data - unblended sources with photometric
fluxes and known spectroscopic redshifts.

The ``blendz.Photoz.calibrate`` function just calls the prior-specific calibration
function. As a result, if you define your own priors (see :ref:`new-prior`), you
will need to write your own calibration function.

The calibration function can be called by creating a ``Photoz`` object
with configuration set to the calibration data. In this configuration,
the ``prior_parameters`` option should be set to ``None``.


.. code:: python

    pz_calib = blendz.Photoz(config_path='calibration_config.txt')
    pz_calib.calibrate()

For the default priors, this will result in a file called ``calibrated_prior_config.txt``
being created. This is a configuration file with the prior parameters set to the
maximum *a posteriori* parameters found in the calibration. This can then be read
in alongside a photoz-configuration for sampling as normal.

.. code:: python

    pz = blendz.Photoz(config_path=['photoz_config.txt',
                                    'calibrated_prior_config.txt'])

    pz.sample([2, 1])

The photoz-configuration file should also have the ``prior_parameters`` option set to ``None``.
