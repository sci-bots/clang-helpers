#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from setuptools import setup

import versioneer

setup(name='clang_helpers',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='High-level API using `clang` module to provide static C++ '
      'class introspection.',
      keywords='c++ clang introspection',
      author='Christian Fobel',
      url='https://github.com/wheeler-microfluidics/clang_helpers',
      license='GPL',
      packages=['clang_helpers'],
      install_requires=['path_helpers', 'pydash'])
