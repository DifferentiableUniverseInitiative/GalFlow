import datetime
import os
import sys

from setuptools import find_packages
from setuptools import setup

setup(
    name='GalFlow',
    description='GalSim reimplementation in TensorFlow',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
)
