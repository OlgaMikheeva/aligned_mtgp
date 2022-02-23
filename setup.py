from setuptools import setup, find_packages

requirements = (
  'numpy>=1.18.0',
  'tensorflow==2.3.0',
  'tensorflow-probability>=0.11.0',
  'gpflow==2.2.1',
  'gpflow_sampling==0.2',
)

extra_requirements = {
  'notebooks': (
    'matplotlib',
    'seaborn',
    'tqdm',
  ),
}

setup(name='amtgp',
      version='0.1',
      license='Creative Commons Attribution-Noncommercial-Share Alike license',
      packages=['amtgp'],
      python_requires='>=3.8',
      install_requires=requirements,
      extras_require=extra_requirements)
