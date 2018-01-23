Specifying templates
=====================

Templates are specified in ``blendz`` using two different types of file, the template
file itself, and a template set. Template sets are configuration files containing the
name, type and filepath of a collection of templates so that the whole set can be
specified as a single configuration option.


Included templates
-------------------

The ``blendz`` installation includes the 8 templates from
`BPZ <http://www.stsci.edu/~dcoe/BPZ/>`_. The ``template_set_path``
configuration option points to the folder where these are stored by default.
As a result, these can be easily used by setting the ``template_set`` configuration
option to either ``BPZ8`` or ``BPZ6``, where the latter
excludes the two starburst templates added to BPZ by
`Coe et al. 2006 <http://adsabs.harvard.edu/abs/2006AJ....132..926C>`_.

A set of only a single template can also be specified using one of the following options:

========================            ===========================================

``single/El_B2004a``                    ``single/Sbc_B2004a``

``single/Scd_B2004a``                     ``single/Im_B2004a``

``single/SB2_B2004a``                      ``single/SB3_B2004a``

``single/ssp_5Myr_z008``                      ``single/ssp_25Myr_z008``

========================            ===========================================



Loading custom templates
-------------------------

If you want to supply your own templates, you need to create a template set file. This
is a configuration file containing the following information:

- The unique name of every template

- The path to the file specifying the template *relative to the location of the template set file*.

- The galaxy type - this should be either ``early``, ``late`` or ``irr`` (unless you define your own priors).

An example layout of a template set is given below:

.. code:: ini

  [name_of_template_1]
  path = path/to/template1.txt
  type = early

  [name_of_template_2]
  path = path/to/template1.txt
  type = late

When using custom templates, the configuration option ``template_set_path`` should point
to the folder containing your ntemplate set file, and ``template_set`` should be the
full filename, including the file extension.

Each template refered to in the template set file is then specified by a plaintext
file of two columns, wavelength (Angstroms) in the first, and spectral flux
density (in **UNITS**) in the second, separated by whitespace, e.g.

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
