#!/usr/bin/env python
"""
Setup file
"""
from setuptools import setup, find_packages

setup(name='module',
      version='1.0',
      description='My Project',
      author='Pontus Vikstål',
      author_email='vikstal@chalmers.se',
      packages = find_packages(include=['module', 'module.*'])
     )
