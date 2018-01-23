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

      def __init__(self, new_prior_params, **kwargs):
          #Run the setup defined in ModelBase
          super(MyNewModel, self).__init__(**kwargs)
          #Do some other setup with your new_prior_parameters

      def lnTemplatePrior(self, template_type, component_ref_mag):
          #Definition of P(t | m0)
          return 0.

      def lnRedshiftPrior(self, redshift, template_type, component_ref_mag):
          #Definition of P(z | t, m0)
          return 0.

      def correlationFunction(self, redshifts):
          #Definition of xi({z})
          return 0.

      def lnMagnitudePrior(self, magnitude):
          #Definition of P({m0})
          return 0.

A few things to note:

- The ``__init__`` function is optional but allows you to define additional setup tasks that are done when your model is instantiated. It is important you call the superclass ``__init__`` if you define this.

- While ``__init__`` is optional, you **must** redefine ``lnTemplatePrior``, ``lnRedshiftPrior``, ``correlationFunction`` and ``lnMagnitudePrior``. ``ModelBase`` is an `abstract base class <https://docs.python.org/3/library/abc.html>`_ that will raise a ``TypeError`` if you attempt to instantiate your class without redefining these methods.

- The ``**kwargs`` get passed by ``ModelBase`` to ``Configuration``, allowing you to edit the settings of your class (and of the flux responses it contains from ``ModelBase``) using keyword arguments.

- The ``template_type`` argument passed to ``lnTemplatePrior`` and ``lnRedshiftPrior`` is a string specifying the template type as defined in the template set file. While the default priors are only defined for ``"early"``, ``"late"`` and ``"irr"`` types, you can support any types your templates use - you need to define your priors for at least these three if you use the default templates.

- The ``redshift`` and ``component_ref_mag`` arguments passed to ``lnTemplatePrior`` and ``lnRedshiftPrior`` are floats, while the ``redshifts`` argument in ``correlationFunction`` and the ``magnitude`` argument in ``lnMagnitudePrior`` are 1D ``numpy`` arrays.



Using the new model
--------------------

.. code:: python

  new_model = MyNewModel(new_prior_params=42, template_set='BPZ6')

  pz = blendz.Photoz(model=new_model)
