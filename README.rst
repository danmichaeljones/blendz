blendz
======

*Bayesian photometric redshift of blended sources*

.. image:: https://img.shields.io/badge/astro--ph.CO-arxiv%3A1234.5678-B31B1B.svg?style=flat
    :target: https://arxiv.org/abs/1234.5678

.. image:: https://readthedocs.org/projects/blendz/badge/
    :target: http://blendz.readthedocs.io/en/latest/

.. image:: https://travis-ci.com/danmichaeljones/blendz.svg?token=gRZ3WUjBtLERAoRgdDFa&branch=master
    :target: https://travis-ci.com/danmichaeljones/blendz


``blendz`` is a python module for estimating photometric redshifts of (possibly)
blended sources.



Installation
------------

``blendz`` can be installed from the command line using  `pip <http://www.pip-installer.org/>`_:

.. code:: bash

    >> pip install blendz

To download from source instead, clone `the repository <https://github.com/danmichaeljones/blendz>`_ and install from there:

.. code:: bash

    >> git clone https://github.com/danmichaeljones/blendz.git
    >> cd blendz
    >> python setup.py install

Downloading from source allows you to run the tests, which require ``pytest``.

.. code:: bash

    >> pip install pytest
    >> cd path/to/blendz
    >> pytest


``blendz`` requires ``numpy`` and ``scipy`` to run, two common packages in scientific python code. These
are easily installed using the `Anaconda python distribution <https://www.anaconda.com/download/>`_
if you're not already a python user.

Citation
--------

If you use this code in your research, please attribute `this paper <https://arxiv.org/abs/1234.5678>`_:

.. code-block:: tex

  @article{blendz,
         Author = ... ,
       }
