"""
Quantum Simulations Framework for Agrivoltaics.

This framework provides advanced tools for simulating quantum dynamics
in excitonic systems (FMO complex) and coupling them with agrivoltaic
system performance, eco-design, and techno-economic viability.
"""

__version__ = "1.0.0"

from .core.hops_simulator import HopsSimulator
from .core.constants import *
from .models import *
from .simulations.testing_validation import TestingValidationProtocols
from .utils.logging_config import setup_logging, get_logger

__all__ = [
    'HopsSimulator',
    'TestingValidationProtocols',
    'setup_logging',
    'get_logger',
]
