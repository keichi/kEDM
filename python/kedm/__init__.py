"""
Python bindings for kEDM
------------------------

.. currentmodule:: kedm

.. autosummary::
   :toctree: _generate

   ccm
   edim
   simplex
   eval_simplex
   smap
   eval_smap
   xmap
   get_kokkos_config
"""

from ._kedm import *

from . import _version
__version__ = _version.get_versions()['version']
