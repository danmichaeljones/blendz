
Getting started
===============

For most normal uses of ``blendz``, the only class you should need is ``blendz.Photoz``.
This is designed to be the only user-facing class.

The order of things to do to use ``blendz`` is as follows:

- :ref:`set-config` - done either using configuration files or keyword arguments.

- :ref:`calibrate` - The prior parameters can be set manually in the configuration, or using the prior calibration procedure. The output of the default calibration procedure is another configuration file that can be read in, containing the prior parameters.

- :ref:`photoz` - After running the nested sampling for each number of components under consideration, the posterior samples and blend probabilities are available for analysis.
