# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages


long_description = open("README.md").read()

install_requires = []

setup(
    name="pylgn",
    version=1.0,
    url='http://pylgn.readthedocs.io/',
    author='Milad H. Mobarhan',
    author_email='m@milad.no',
    license="GPLv3",
    packages=find_packages(),
    include_package_data=True,
)
