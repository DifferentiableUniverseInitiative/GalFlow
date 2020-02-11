import datetime
import os
import sys

from setuptools import find_packages
from setuptools import setup
from io import open

# read the contents of the README file
with open('README.md', encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='GalFlow',
    description='GalSim reimplementation in TensorFlow',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='ML4Astro Contributors',
    url='http://github.com/ml4astro/GalFlow',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'galsim'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Astronomy'
    ],
    keywords='astronomy',
    use_scm_version=True,
    setup_requires=['setuptools_scm']
)
