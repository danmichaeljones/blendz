Specifying filters
===================


Included filters
-------------------

The ``blendz`` installation includes filter response curves from several instruments. The ``filter_path``
configuration option points to the folder where these are stored by default, and so they can be
specified in the ``filters`` configuration option using the following names:


**SDSS**

Described in `Stoughton et al. (2002) <http://adsabs.harvard.edu/abs/2002AJ....123..485S>`_



======================               =====================
Configuration option                    Description
======================               =====================
``sdss/u``                                 u

``sdss/g``                                  g

``sdss/r``                                  r

``sdss/i``                                  i

``sdss/z``                                   z
======================               =====================




**Viking**

Described in `Edge et al. (2013) <http://adsabs.harvard.edu/abs/2013Msngr.154...32E>`_


======================               =====================
Configuration option                    Description
======================               =====================
``viking/h``                                H

``viking/j``                                J

``viking/k``                              K

``viking/y``                                Y
======================               =====================



**LSST**

Described in `Abell et al.(2009) <http://adsabs.harvard.edu/abs/2009arXiv0912.0201L>`_

======================               =====================
Configuration option                    Description
======================               =====================
``lsst/u``                                 u

``lsst/g``                                  g

``lsst/r``                                  r

``lsst/i``                                  i

``lsst/z``                                   z

``lsst/y``                                   Y
======================               =====================






Loading custom filters
-------------------------

Each custom filter can be specified by a single plaintext file of two columns,
wavelength (Angstroms) in the first, and the filter response in the second,
separated by whitespace, e.g.


.. code:: ini

  912.0  0.0329579004203
  920.0  0.0332336431181
  930.0  0.0335731230922
  940.0  0.033939398051
  950.0  0.0342922864396
  960.0  0.0346317644112
  970.0  0.0349582358084
  980.0  0.0352716948728
  990.0  0.0355717998991
  1000.0  0.0358573156287
  1010.0  0.0361306346606

The ``filter_path`` configuration option should point to the folder where
these files are stored, and each entry in the ``filters`` configuration option
should be the path (including the name and file extension) relative to ``filter_path``, i.e.,
for the following folder layout:


.. code:: ini

  containing_folder/
  ├── filters/
  │   ├── instrument_one/
  │   │   ├── filter_one.txt
  │   │   ├── filter_two.txt
  │   ├── instrument_two/
  │   │   ├── filter_three.txt
  │   │   ├── filter_four.txt

you should set the configuration options to:

.. code:: python

  filter_path = "containing_folder/filters"
  filters = ["instrument_one/filter_one.txt", "instrument_one/filter_two.txt", \
             "instrument_two/filter_three.txt", "instrument_two/filter_four.txt"]

The default filters only don't need file extensions as they are saved in plaintext files without file extensions.
