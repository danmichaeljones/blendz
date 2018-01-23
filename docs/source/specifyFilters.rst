Specifying filters
===================


Included filters
-------------------

The ``blendz`` installation includes filter response curves from several instruments. The ``filter_path``
configuration option points to the folder where these are stored by default, and so they can be
specified in the ``filters`` configuration option using the following names:


**SDSS**

======================               =====================
Configuration option                    Description
======================               =====================
``sdss/u``                                 u

``sdss/g``                                  g

``sdss/r``                                  r

``sdss/i``                                  i

``sdss/z``                                   z
======================               =====================





**Subaru**

======================               ====================================
Configuration option                    Description
======================               ====================================
``subaru/B``                          Subaru Suprime-Cam B filter

``subaru/g``                            Subaru Suprime-Cam g' filter

``subaru/I``                            Subaru Suprime-Cam Ic filter

``subaru/R``                            Subaru Suprime-Cam Rc filter

``subaru/rp``                           Subaru Suprime-Cam r' filter

``subaru/V``                            Subaru Suprime-Cam V filter

``subaru/z``                            Subaru Suprime-Cam z' filter
======================               ====================================






**Viking**

======================               =====================
Configuration option                    Description
======================               =====================
``viking/h``

``viking/j``

``viking/k``

``viking/y``
======================               =====================



**BPZ-Packaged**

=================================               =====================
Configuration option                               Description
=================================               =====================
``bpz/rp_Subaru.res``

``bpz/H_Johnson.res``

``bpz/V_Johnson.res``

``bpz/g_SDSS.res``

``bpz/R_LRIS.res``

``bpz/u_SDSS.res``

``bpz/I_LRIS.res``

``bpz/I_Subaru.res``

``bpz/z_Subaru.res``

``bpz/U1400.res``

``bpz/R_Subaru.res``

``bpz/HST_ACS_WFC_F606W.res``

``bpz/nic3_f110w.res``

``bpz/V_LRIS.res``

``bpz/HST_ACS_WFC_F435W.res``

``bpz/J_Johnson.res``

``bpz/HST_ACS_WFC_F775W.res``

``bpz/U_Johnson.res``

``bpz/nic3_f160w.res``

``bpz/g_Subaru.res``

``bpz/HST_ACS_WFC_F850LP.res``

``bpz/V_Subaru.res``

``bpz/r_SDSS.res``

``bpz/i_SDSS.res``

``bpz/z_SDSS.res``

``bpz/B_Johnson.res``

``bpz/K_Johnson.res``

``bpz/B_Subaru.res``
=================================               =====================




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
  filters = ["instrument_one/filter_one", "instrument_one/filter_two", \
             "instrument_two/filter_three", "instrument_two/filter_four"]
