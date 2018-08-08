from setuptools import setup, find_packages

setup(name = 'blendz',
      version = '1.0.0',
      description = 'Bayesian photometric redshifts of blended sources.',
      author = 'Daniel Michael Jones',
      author_email = 'd.jones15@imperial.ac.uk',
      license = 'MIT',
      url = 'http://blendz.readthedocs.io',
      packages = find_packages(),
      install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'nestle',
        'dill',
        'future',
        'emcee',
      ],
      include_package_data = True)#,
#      zip_safe = False)
