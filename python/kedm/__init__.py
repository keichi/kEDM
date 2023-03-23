"""
Python bindings for kEDM
------------------------

.. currentmodule:: kedm

.. autosummary::
   :toctree: _generate

   edim
   simplex
   simplex_eval
   smap
   smap_eval
   xmap
   get_kokkos_config
"""

from ._kedm import *

from . import _version
__version__ = _version.get_versions()['version']
