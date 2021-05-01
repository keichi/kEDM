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

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
