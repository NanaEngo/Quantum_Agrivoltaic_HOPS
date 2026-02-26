import sys
from pathlib import Path

import numpy as np

# Add the parent directory to sys.path to allow importing from the framework
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from core.hamiltonian_factory import create_fmo_hamiltonian
from models.agrivoltaic_coupling_model import AgrivoltaicCouplingModel
from models.environmental_factors import EnvironmentalFactors
from models.spectral_optimizer import SpectralOptimizer


def test_agrivoltaic_coupling_model():
    """Test spectral coupling between OPV and PSU."""
    H, _ = create_fmo_hamiltonian(include_reaction_center=False)
    model = AgrivoltaicCouplingModel(fmo_hamiltonian=H)

    # Standard splitting check
    assert model.n_opv_sites == 4
    assert model.n_psu_sites == 7
    assert model.n_total == 28  # 4 * 7


def test_spectral_optimizer():
    """Test the spectral optimization routine."""
    # Mock spectrum and responses
    lambdas = np.linspace(300, 1100, 100)
    sun = np.ones_like(lambdas)
    opv_res = np.exp(-((lambdas - 600) ** 2) / 10000)
    psu_res = np.exp(-((lambdas - 450) ** 2) / 5000) + np.exp(-((lambdas - 680) ** 2) / 5000)

    optimizer = SpectralOptimizer(
        solar_spectrum=(lambdas, sun), opv_response=opv_res, psu_response=psu_res
    )

    # Run a quick optimization
    results = optimizer.optimize_spectral_splitting(n_filters=1, maxiter=5, popsize=4)

    assert "optimal_pce" in results
    assert "optimal_etr" in results
    assert 0 <= results["optimal_pce"] <= 1.0


def test_environmental_factors():
    """Test environmental data generation and modeling."""
    env = EnvironmentalFactors()

    # Generate dust profile (using internal method name)
    dust = env.dust_accumulation_model(np.arange(30))

    assert len(dust) == 30

    # Test temperature effect
    temp_effect = env.temperature_effects_model(np.array([295, 305]), base_efficiency=0.18)
    assert len(temp_effect) == 2


if __name__ == "__main__":
    pytest.main([__file__])
