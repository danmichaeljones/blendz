All configuration options
=========================

Below are all of the possible configuration settings. When being set by a `configuration file`,
they should be given as described on **PAGE**, split by ``[Data]`` and ``[Run]``. When being set
as `keyword arguments`, this split is not necessary, but each option should be passed
as the correct type, like the example on **PAGE**.

If you do not set an option, the default value is taken instead. Options with a `N/A` default
value are not optional and must be set by you.


Data options
-------------

=====================        ========================================================                 ==================================================              ========================
Configuration option         Explanation                                                                    Default                                                    Python type
=====================        ========================================================                 ==================================================              ========================
data_path                     Absolute path to the file containing your photometry.                               *N/A*                                                       ``str``


filter_path                   Absolute path to the folder where the files describing
                              your filters are stored.                                                     Filter folder in the included resources.                       ``str``

mag_cols                      List of the columns in the data-file where the                                  *N/A*                                                       ``list`` of ``int``
                              AB-magnitude of the source in each band is. Indices
                              start at zero.

sigma_cols                    List of the columns in the data-file where the error                              *N/A*                                                     ``list`` of ``int``
                              on the AB-magnitude of the source in each band is.
                              Indices start at zero.

spec_z_col                    The column in the data-file where the spectroscopic                           ``None``                                                     ``int`` *or* ``None``
                              redshift of the source is. If it's not present in
                              the data-file, set to ``None``. Indices start
                              at zero.

ref_band                      Index of the filter band that is considered the                                   *N/A*                                                      ``int``
                              reference band, the band the priors are conditioned
                              on, and where the soure magnitude is sampled.
                              Indices start at zero.

filters                       List of paths to the filter files, relative to                                     *N/A*                                                   ``list`` of ``str``
                              ``filter_path``

filter_file_extension         **REMOVE**

zero_point_errors             List of errors on the zero point calibration of                                  *N/A*                                                   ``list`` of ``float``
                              each filter band.
=====================        ========================================================                 ==================================================              ========================



Run options
------------

=====================        ================================================                 ===============================================              ========================
Configuration option         Explanation                                                      Default                                                         Python type
=====================        ================================================                 ===============================================              ========================
z_lo                          Minimum redshift to sample.                                           0                                                             ``float``

z_hi                          Maximum redshift to sample.                                             10                                                            ``float``

z_len                         Length of redshift grid to calculate                                  1000                                                            ``int``
                              functions of redshift on before interpolating.

ref_mag_lo                    Minimum magnitude to sample (numerically, i.e.                        *N/A*                                                           ``float``
                              the *brightest* magnitude).

ref_mag_hi                    Maximum magnitude to sample (numerically, i.e.                        *N/A*                                                           ``float``
                              the *dimmest* magnitude).

template_set_path             Absolute path to the folder containing the                       Template folder in the included resources.                        ``str``
                              template set file, as described in **PAGE**.
                              Templates in the template set file are
                              specified with a path relative to this.

template_set                  File name of the template set file.                               ``BPZ8`` - The set of 8 templates from BPZ                        ``str``
=====================        ================================================                 ===============================================              ========================
