"""
Models module for quantum agrivoltaic simulations.

This module contains model classes for analyzing molecular systems,
biodegradability, environmental factors, sensitivity analysis, 
LCA, techno-economics, and spectroscopy.
"""

try:
    from .agrivoltaic_coupling_model import AgrivoltaicCouplingModel
except ImportError:
    AgrivoltaicCouplingModel = None

try:
    from .biodegradability_analyzer import BiodegradabilityAnalyzer
except ImportError:
    BiodegradabilityAnalyzer = None

try:
    from .eco_design_analyzer import EcoDesignAnalyzer
except ImportError:
    EcoDesignAnalyzer = None

try:
    from .environmental_factors import EnvironmentalFactors
except ImportError:
    EnvironmentalFactors = None

try:
    from .lca_analyzer import LCAAnalyzer
except ImportError:
    LCAAnalyzer = None

try:
    from .multi_scale_transformer import MultiScaleTransformer
except ImportError:
    MultiScaleTransformer = None

try:
    from .quantum_dynamics_simulator import QuantumDynamicsSimulator
except ImportError:
    QuantumDynamicsSimulator = None

try:
    from .sensitivity_analyzer import SensitivityAnalyzer
except ImportError:
    SensitivityAnalyzer = None

try:
    from .simple_quantum_dynamics_simulator import SimpleQuantumDynamicsSimulator
except ImportError:
    SimpleQuantumDynamicsSimulator = None

try:
    from .spectral_optimizer import SpectralOptimizer
except ImportError:
    SpectralOptimizer = None

try:
    from .spectroscopy_2des import Spectroscopy2DES
except ImportError:
    Spectroscopy2DES = None

try:
    from .techno_economic_model import TechnoEconomicModel
except ImportError:
    TechnoEconomicModel = None


__all__ = [
    'AgrivoltaicCouplingModel',
    'BiodegradabilityAnalyzer',
    'EcoDesignAnalyzer',
    'EnvironmentalFactors',
    'LCAAnalyzer',
    'MultiScaleTransformer',
    'QuantumDynamicsSimulator',
    'SensitivityAnalyzer',
    'SimpleQuantumDynamicsSimulator',
    'SpectralOptimizer',
    'Spectroscopy2DES',
    'TechnoEconomicModel',
]
