.. _photoz:

Running the inference and model comparison
==========================================

Sampling
---------

Once the ``Photoz`` object is instantiated, e.g., with a configuration file,

.. code:: python

    pz = blendz.Photoz(data_path='path/to/data.txt',
                       mag_cols = [1, 2, 3, 4, 5],
                       sigma_cols = [6, 7, 8, 9, 10],
                       ref_band = 2,
                       filters=['sdss/u', 'sdss/g',
                                'sdss/r', 'sdss/i', 'sdss/z'])

the inference can be run for each number of components you'd like to analyse
(e.g., compare between the single components and two-component blend cases), you
need to sample. This can be done by calling the ``sample`` function, which accepts
either an ``int`` or a ``list`` of ``int`` for the components to sample, e.g.,

.. code:: python

    pz.sample([1, 2], save_path='photoz_out.pkl', save_interval=1)

This excerpt also shows the saving feature, which saves ``Photoz`` object to file
every ``save_interval`` sources, and once all sources are analysed. These save files
can be loaded when creating the ``Photoz`` instance by

.. code:: python

    pz = blendz.Photoz(load_state_path='photoz_out.pkl')

Running in parallel
-------------------

The inference can be run in parallel by saving a script to file (e.g., the code above
into a file ``photoz_run.py``) and running with MPI:

.. code:: bash

    mpiexec python photoz_run.py

This requires both MPI and MultiNest be manually installed - see :ref:`install`.


Analyse the inference results
-----------------------------

After the sampling run is complete, the posterior samples can be accessed
using the ``chain`` method, e.g.

.. code:: python

    pz.chain(2, galaxy=0)

returns the samples from the two-component posterior for galaxy 0. If the optional
keyword argument is omitted, a list of chains is returned with one array for each source.

A variety of summery statistic functions are also provided, such as the mean of each parameter

.. code:: python

    pz.mean(2, galaxy=0)

and the maximum *a posteriori* value for each parameter,

.. code:: python

    pz.max(2, galaxy=0)

Again, the ``galaxy`` argument is optional, and omitting it will return an
array of shape ``(num_galaxies, num_components * 2)``.

Model comparison
----------------

The model comparison results can be accessed using the ``logbayes`` function, e.g.,

.. code:: python

    pz.logbayes(2, 1, galaxy=0)

will return the Bayes factor for comparison between the two-component blend and
single source cases. A model comparison prior can be included by multiplying
this value. If the ``galaxy`` argument is omitted, an array of ``float``,
one for each source, is returned.
