"""
Extensions module for advanced MesoHOPS adapters.

This module provides PT-HOPS (Process Tensor) and SBD (Spectrally Bundled Dissipators)
adapters for high-performance quantum dynamics simulations.
"""

from .mesohops_adapters import PT_HopsNoise, SBD_HopsTrajectory
from .spectral_bundling import SpectralBundle, SpectrallyBundledDissipator

__all__ = [
    'PT_HopsNoise',
    'SBD_HopsTrajectory',
    'SpectralBundle',
    'SpectrallyBundledDissipator',
]
