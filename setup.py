from setuptools import setup, find_packages

setup(
  name='dm_project2',
  version='0.1.0',
  packages=find_packages(),
  install_requires=[
    'pandas',
    'numpy',
    'scikit-learn',
    'matplotlib'
  ],
  entry_points={
    'console_scripts': [
      'dm_project2=dm_project2:main'
    ]
  }
)
