import sys

from paver.easy import task, needs
from paver.setuputils import setup, install_distutils_tasks
sys.path.insert(0, '.')
import version


setup(name='clang_helpers',
      version=version.getVersion(),
      description='High-level API using `clang` module to provide static C++ '
      'class introspection.',
      keywords='c++ clang introspection',
      author='Christian Fobel',
      url='https://github.com/wheeler-microfluidics/clang_helpers',
      license='GPL',
      packages=['clang_helpers', 'clang_helpers.clang'],
      package_data={'clang_helpers': ['libclang/*']},
      install_requires=['clang>=3.8', 'path_helpers'])


@task
@needs('generate_setup', 'minilib', 'setuptools.command.sdist')
def sdist():
    """Overrides sdist to make sure that our setup.py is generated."""
    pass


@task
@needs('generate_setup', 'minilib', 'setuptools.command.bdist_wheel')
def bdist_wheel():
    """Overrides bdist_wheel to make sure that our setup.py is generated."""
    pass
