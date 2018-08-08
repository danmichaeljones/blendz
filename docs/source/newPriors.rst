.. _new-prior:

Specifying new priors
======================

You can specify new priors by subclassing ``blendz.model.ModelBase``, redefining
the four functions that return the priors and instantiating your new model for ``blendz.photoz``.


Creating a new model class
---------------------------

Your new class should have the following basic layout:

.. code:: python

  from blendz.model import ModelBase

  class MyNewModel(ModelBase):

      #Optional:
      def __init__(self, new_prior_params, **kwargs):
          #Run the setup defined in ModelBase
          super(MyNewModel, self).__init__(**kwargs)
          #Do some other setup with your new_prior_parameters

      #Mandatory:
      def lnPrior(self, redshift, magnitude):
          #Definition of P(z_a, t_a, m_0a) for all t_a
          return 0.

      #Optional:
      def correlationFunction(self, redshifts):
          #Definition of xi({z})
          return 0.

      #Optional:
      def calibrate(self, photometry, cached_likelihood, **kwargs):
          return 0.



A few things to note:

- The ``__init__`` function is optional but allows you to define additional setup tasks that are done when your model is instantiated. It is important you call the superclass ``__init__`` if you define this.

- The ``correlationFunction`` function is also optional. The function ``self.comovingSeparation(z_lo, z_hi)`` defined in ``ModelBase`` may be helpful.

- While ``__init__`` is optional, you **must** redefine ``lnPrior``. This function takes a ``float`` for both the redshift and magnitude, and returns a ``numpy.array`` of the natural log of the prior for each template *type* (not each template). The ``self.possible_types`` attribute is a list of the possible types, where each element is a string with the name of that type. These are automatically read from the template set file supplied at runtime.

- The ``**kwargs`` get passed by ``ModelBase`` to ``Configuration``, allowing you to edit the configuration like other ``blendz`` classes using keyword arguments.

- The ``redshift``, ``magnitude`` and ``component_ref_mag`` arguments passed to natural-log prior functions are floats, while the ``redshifts`` argument in ``correlationFunction`` is a 1D ``numpy`` array.

- The ``calibrate`` function is also optional. This takes as arguments a ``blendz.photometry.Photometry`` object and a ``numpy.array`` of shape ``(num_galaxies, num_templates)`` filled with the likelihood. This function is called by the ``blendz.Photoz.calibrate(**kwargs)`` function, with any keyword arguments passed to the function here. There is no return value for this function; the default model writes the resulting parameters to a configuration file that can be read by ``blendz.Photoz``.

Using the new model
--------------------

The new model can simply be instantiated and passed to ``blendz.Photoz`` as a keyword argument.

.. code:: python

  new_model = MyNewModel(new_prior_params=42, template_set='BPZ6')

  pz = blendz.Photoz(model=new_model)
