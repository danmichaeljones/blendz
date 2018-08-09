blendz
======

*Bayesian photometric redshifts of blended sources*

.. image:: https://img.shields.io/badge/astro--ph.CO-arxiv%3A1808.02846-B31B1B.svg?style=flat
    :target: https://arxiv.org/abs/1808.02846

.. image:: https://readthedocs.org/projects/blendz/badge/
    :target: http://blendz.readthedocs.io/en/latest/

.. image:: https://travis-ci.com/danmichaeljones/blendz.svg?token=gRZ3WUjBtLERAoRgdDFa&branch=master
    :target: https://travis-ci.com/danmichaeljones/blendz

.. image:: https://img.shields.io/pypi/v/blendz.svg
    :target: https://pypi.org/project/blendz/

.. image:: https://img.shields.io/github/license/danmichaeljones/blendz.svg
    :target: https://github.com/danmichaeljones/blendz


``blendz`` is a Python module for estimating photometric redshifts of (possibly)
blended sources with an arbitrary number of intrinsic components. Using nested sampling,
``blendz`` gives you samples from the joint posterior distribution of redshift
and magnitude of each component, plus the relative model probability to identify whether
a source is blended.

``blendz`` is easy to install using  `pip <http://www.pip-installer.org/>`_

.. code:: bash

    pip install blendz

and can be run using either simple configuration files

.. code:: python


    pz = blendz.Photoz(config_path='path/to/config.txt')
    pz.sample(2)

or keyword arguments

.. code:: python

    pz = blendz.Photoz(data_path='path/to/data.txt',
                       mag_cols = [1, 2, 3, 4, 5],
                       sigma_cols = [6, 7, 8, 9, 10],
                       ref_band = 2,
                       filters=['sdss/u', 'sdss/g',
                                'sdss/r', 'sdss/i', 'sdss/z'])

    pz.sample(2)

to set the configuration.

You can `read the full documentation <http://blendz.readthedocs.io>`_.

Citation
--------

If you use this code in your research, please attribute `this paper <https://arxiv.org/abs/1808.02846>`_:

.. code-block:: tex

  @article{blendz,
           author = {{Jones}, D.~M. and {Heavens}, A.~F.},
           title = "{Bayesian photometric redshifts of blended sources}",
           journal = {ArXiv e-prints},
           archivePrefix = "arXiv",
           eprint = {1808.02846},
           keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies},
           year = 2018,
           month = aug,
           adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180802846J},
           adsnote = {Provided by the SAO/NASA Astrophysics Data System}
           }
