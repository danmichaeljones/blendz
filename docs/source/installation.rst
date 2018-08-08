.. _install:

Installation
============


Easy installation
--------------------

``blendz`` can be installed from the command line using  `pip <http://www.pip-installer.org/>`_:

.. code:: bash

    pip install blendz

``blendz`` requires ``numpy`` and ``scipy`` to run, two common packages in scientific python code. These
are easily installed using the `Anaconda python distribution <https://www.anaconda.com/download/>`_
if you're not already a python user.

This allows you to use ``blendz`` straight away by installing `Nestle <http://kylebarbary.com/nestle/>`_, a pure Python
implementation of the Nested Sampling algorithm. While this is easier to install than Multinest, photo-z runs
will be slower. If you have a large number of galaxies you want to analyse, you should install Multinest
using the instructions below.


Installing from source
----------------------

To download and install ``blendz`` from source instead, clone `the repository <https://github.com/danmichaeljones/blendz>`_
and install from there:

.. code:: bash

    git clone https://github.com/danmichaeljones/blendz.git
    cd blendz
    python setup.py install

Downloading from source allows you to run the tests, which require ``pytest``.

.. code:: bash

    pip install pytest
    cd path/to/blendz
    pytest


Installing Multinest
----------------------

``blendz`` uses the `PyMultinest <https://johannesbuchner.github.io/PyMultiNest/index.html>`_ library
to run `Multinest <https://ccpforge.cse.rl.ac.uk/gf/project/multinest/>`_. To enable Multinest sampling in ``blendz``,
you need to install both of these libraries yourself.

Detailed instructions to install these `can be found here <https://johannesbuchner.github.io/PyMultiNest/install>`_
with additional details for installing on macOS `available here <http://astrobetter.com/wiki/MultiNest+Installation+Notes>`_.

It is recommended you ensure that you have MPI and ``mpi4py`` working before installing Multinest to enable parallel sampling
which can greatly increase the speed of photo-z runs. Try installing ``mpi4py``:

.. code:: bash

    pip install mpi4py

and test:

.. code:: bash

    mpiexec -n 1 python -m mpi4py.bench helloworld

If you need to install an MPI library, you can do install openMPI on Linux using

.. code:: bash

    sudo apt-get install openmpi-bin libopenmpi-dev

and on macOS using `MacPorts <https://www.macports.org/>`_ by

.. code:: bash

    sudo port install openmpi


Common errors
--------------------------

Could not load Multinest library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you see an error like

.. code:: bash

    ERROR:   Could not load MultiNest library "libmultinest.so"
    ERROR:   You have to build it first, and point the LD_LIBRARY_PATH environment variable to it!

this is because PyMultinest cannot find the Multinest library. If you installed Multinest in the folder

.. code:: bash

    path/to/Multinest

the following command

.. code:: bash

    export LD_LIBRARY_PATH="path/to/MultiNest/lib:$LD_LIBRARY_PATH"

will add Multinest to the path variable so that it can be found. To avoid having to run this every time
you open a new terminal window, you should add this line to your terminal startup file
(`.bashrc` on Linux and `.bash_profile` on macOS). This can be done on Linux using


.. code:: bash

    echo 'export LD_LIBRARY_PATH="path/to/MultiNest/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc

and on macOS using

.. code:: bash

    echo 'export LD_LIBRARY_PATH="path/to/MultiNest/lib:$LD_LIBRARY_PATH"' >> ~/.bash_profile


Intel MKL fatal error
^^^^^^^^^^^^^^^^^^^^^

The following error

.. code:: bash

    Intel MKL FATAL ERROR: Cannot load libmkl_mc.so or libmkl_def.so

seems to be problem related to Anaconda's packaging of the MKL library. Forcing a reinstallation of ``numpy`` by

.. code:: bash

    conda install -f numpy

can sometimes fix it. For more information, see `this discussion on GitHub. <https://github.com/BVLC/caffe/issues/3884/>`_
