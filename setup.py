from setuptools import setup, find_packages

setup(name = 'blendz',
      version = '0.0',
      description = 'None',
      author = 'None',
      author_email = 'danmichaeljones@me.com',
      license = 'None',
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
