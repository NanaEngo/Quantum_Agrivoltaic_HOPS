"""
Quantum Simulations Framework for Agrivoltaics.

This framework provides advanced tools for simulating quantum dynamics
in excitonic systems (FMO complex) and coupling them with agrivoltaic
system performance, eco-design, and techno-economic viability.
"""

__version__ = "1.0.0"

from .core.constants import *
from .core.hops_simulator import HopsSimulator
from .models import *
from .simulations.testing_validation import TestingValidationProtocols
from .utils.logging_config import get_logger, setup_logging

__all__ = [
    "HopsSimulator",
    "TestingValidationProtocols",
    "setup_logging",
    "get_logger",
]
