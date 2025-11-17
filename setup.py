from setuptools import setup, find_packages
import os

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    jhsfm_requirements = f.read().splitlines()

setup(
    name='jhsfm',
    version='0.0.1',
    packages = find_packages(),
    install_requires = jhsfm_requirements,
)