import logging
import os
import sys

import numpy as np

# Add parent of the package directory to path to ensure we're testing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyStandardization")


def test_imports():
    logger.info("Testing top-level imports...")
    try:
        import quantum_simulations_framework as qsf

        logger.info(f"Successfully imported qsf, version: {getattr(qsf, '__version__', 'unknown')}")

        # Test core components
        from quantum_simulations_framework.core import HopsSimulator
        from quantum_simulations_framework.extensions import PT_HopsNoise, SBD_HopsTrajectory
        from quantum_simulations_framework.models import AgrivoltaicCouplingModel, EcoDesignAnalyzer
        from quantum_simulations_framework.simulations import TestingValidationProtocols
        from quantum_simulations_framework.utils import OrcaRunner, setup_logging

        logger.info("All sub-package primary exports verified.")
        return True
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"Import test failed (missing dependency): {e}")
        return False
    # Unexpected exceptions will propagate for visibility.


def test_hops_simulator():
    logger.info("Testing HopsSimulator orchestration...")
    try:
        from quantum_simulations_framework.core import HopsSimulator

        ham = np.diag([1.0, 0.8])
        sim = HopsSimulator(ham, use_mesohops=False)  # Use fallback for speed
        logger.info(f"Simulator type: {sim.simulator_type}")

        t = np.linspace(0, 10, 5)
        results = sim.simulate_dynamics(t)
        logger.info(f"Simulation results keys: {results.keys()}")
        return "populations" in results
    except KeyboardInterrupt:
        raise
    except (
        ImportError,
        RuntimeError,
        AssertionError,
        ValueError,
        TypeError,
        np.linalg.LinAlgError,
    ) as e:
        logger.error(f"HopsSimulator test failed: {e}")
        return False
    # Unexpected exceptions will propagate for visibility.


def test_orca_runner():
    logger.info("Testing OrcaRunner utility...")
    try:
        from quantum_simulations_framework.utils import OrcaRunner

        # Test with a specific path
        runner = OrcaRunner(orca_path="/home/taamangtchu/orca_6_1_0/orca", nprocs=4)
        logger.info(f"OrcaRunner initialized with nprocs={runner.nprocs}")
        # We won't run a real Orca calculation here as it requires the binary
        return True
    except (ImportError, FileNotFoundError, OSError) as e:
        logger.error(f"OrcaRunner test failed (environment): {e}")
        return False
    # Unexpected exceptions will propagate for visibility.


def test_eco_analyzer():
    logger.info("Testing EcoDesignAnalyzer...")
    try:
        from quantum_simulations_framework.models import EcoDesignAnalyzer

        EcoDesignAnalyzer()
        logger.info("EcoDesignAnalyzer initialized.")
        return True
    except (ImportError, RuntimeError, ValueError, TypeError) as e:
        logger.error(f"EcoDesignAnalyzer test failed: {e}")
        return False
    # Unexpected exceptions will propagate for visibility.


if __name__ == "__main__":
    logger.info("=== Quantum Simulations Framework Standardization Verification ===")

    results = {
        "Imports": test_imports(),
        "HopsSimulator": test_hops_simulator(),
        "OrcaRunner": test_orca_runner(),
        "EcoDesignAnalyzer": test_eco_analyzer(),
    }

    logger.info("\nVerification Summary:")
    for test, passed in results.items():
        logger.info(f"- {test}: {'PASS' if passed else 'FAIL'}")

    if all(results.values()):
        logger.info("\nFramework standardization verified successfully.")
        sys.exit(0)
    else:
        logger.error("\nFramework standardization verification failed.")
        sys.exit(1)
