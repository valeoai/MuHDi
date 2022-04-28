from setuptools import find_packages
from setuptools import setup

setup(name='MuHDi',
      install_requires=['pyyaml',
                        'easydict',
                        'scipy', 'scikit-image',
                        'future', 'setuptools',
                        'tqdm', 'cffi'
                        ],
      packages=find_packages())
