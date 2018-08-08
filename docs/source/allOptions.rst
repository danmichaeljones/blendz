.. _config-options:

All configuration options
=========================

Below are all of the possible configuration settings. When being set by a `configuration file`,
they should be given as described in :ref:`set-config`, split by ``[Data]`` and ``[Run]``. When being set
as `keyword arguments`, this split is not necessary, but each option should be passed
as the correct type.

If you do not set an option, the default value is taken instead. Options with a `N/A` default
value are not optional and must be set by you.


Data options
-------------

=====================        ========================================================                 ==================================================              ========================
Configuration option         Explanation                                                                    Default                                                    Python type
=====================        ========================================================                 ==================================================              ========================
data_path                     Absolute path to the file containing your photometry.                               *N/A*                                                       ``str``

skip_data_rows                Number of columns to ignore at the top of the data                                    0                                                           ``int``
                              file.

data_is_csv                   Flag of whether data is comma-separated. If                                         ``False``                                                     ``bool``
                              ``False``, the data file is assumed to be whitespace
                              separated.

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

ref_band                      Index of the filter band                                                             *N/A*                                                      ``int``
                              (of the list of filters, *not* the data file
                              column) that is considered the
                              reference band, the band the priors are conditioned
                              on, and where the soure magnitude is sampled.
                              Indices start at zero.

filters                       List of paths to the filter files, relative to                                     *N/A*                                                   ``list`` of ``str``
                              ``filter_path``, *with* file extensions. The
                              included filters are saved in files
                              without a file extension.

zero_point_errors             List of errors on the zero point calibration of                                  *N/A*                                                   ``list`` of ``float``
                              each filter band.

magnitude_limit                Value of the survey magnitude limit, fixed for                                     *N/A*                                                       ``float``
                               all galaxies. One of ``magnitude_limit`` or
                               ``magnitude_limit_col`` must be set. If both
                               are set. ``magnitude_limit`` is ignored.

magnitude_limit_col            Value of the survey magnitude limit, set                                         *N/A*                                                         ``int``
                               individually for each
                               galaxy. One of ``magnitude_limit`` or
                               ``magnitude_limit_col`` must be set. If both
                               are set. ``magnitude_limit_col``
                               is preferred.

no_detect_value                Value of the data when an observation was                                                        99.0                                             ``float``
                               made but the source was not detected.

no_observe_value               Value of the data when an observation was                                                        -99.0                                             ``float``
                               not made.

angular_resolution             Angular resolution of the data. Sources with a                                             *N/A*                                                     ``float``
                               smaller angular separation than this are assumed
                               to be blended (for the correlation function).
                               In units of arcseconds.
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

ref_mag_hi                    Fixed maximum magnitude to sample (numerically,
                              i.e. the *dimmest* magnitude). One of                                  *N/A*                                                         ``float``
                              ``ref_mag_hi`` or ``ref_mag_hi_sigma`` must be
                              set. If both are set, ``ref_mag_hi`` is
                              ignored.

ref_mag_hi_sigma              Maximum magnitude to sample (numerically,
                              i.e. the *dimmest* magnitude) in terms of                                   *N/A*                                                         ``float``
                              reference band flux error. One of
                              ``ref_mag_hi`` or ``ref_mag_hi_sigma`` must be
                              set. If both are set, ``ref_mag_hi_sigma`` is
                              preferred.

template_set_path             Absolute path to the folder containing the                       Template folder in the included resources.                        ``str``
                              template set file, as described in
                              :ref:`templates`.
                              Templates in the template set file are
                              specified with a path relative to this.

template_set                  File name of the template set file.                               ``BPZ8`` - The set of 8 templates from BPZ                        ``str``

sort_redshifts                Whether to use redshift sorting to break the                                ``True``                                                ``bool``
                              exchangability of blended posteriors. If
                              ``False``, magnitude sorting is used.

omega_mat                      Omega-matter cosmological parameter.                                     0.3065                                                    ``float``

omega_lam                      Omega-lambda cosmological parameter.                                     0.6935                                                    ``float``

omega_k                        Omega-k cosmological parameter.                                            0.                                                      ``float``

hubble                         Hubble constant in km/s/Mpc.                                               67.9                                                    ``float``

r0                              Constant of proportionality in                                            5.                                                       ``float``
                                correlation function power law.
                                Units of Mpz/h.

gamma                           Exponent parameter in                                                     1.77                                                      ``float``
                                correlation function power law.

prior_params                   Array of prior parameters used in the                                Parameters from Benitez (2000), plus                     ``list`` of ``float``
                               default prior function. Can be set to                                0.6 for the additional magnitude prior
                               ``None`` for being set by the prior                                  parameter.
                               calibration described on :ref:`calibrate`.


=====================        ================================================                 ===============================================              ========================
