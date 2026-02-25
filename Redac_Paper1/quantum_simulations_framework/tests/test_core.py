import os
import sys
import numpy as np
from pathlib import Path

# Add the parent directory to sys.path to allow importing from the framework
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from core.constants import FMO_SITE_ENERGIES_7, FMO_COUPLINGS, DEFAULT_TEMPERATURE
from core.hamiltonian_factory import create_fmo_hamiltonian
from core.hops_simulator import HopsSimulator

def test_constants_validity():
    """Verify that core constants are correctly defined and have expected shapes."""
    assert len(FMO_SITE_ENERGIES_7) == 7
    assert isinstance(FMO_COUPLINGS, dict)
    assert DEFAULT_TEMPERATURE == 295.0

def test_hamiltonian_factory():
    """Verify that the Hamiltonian factory generates valid FMO matrices."""
    H, energies = create_fmo_hamiltonian(include_reaction_center=False)
    assert H.shape == (7, 7)
    assert np.allclose(np.diag(H), energies)
    # Symmetry check
    assert np.allclose(H, H.T)
    
    # Test 8-site (RC) variant
    H8, e8 = create_fmo_hamiltonian(include_reaction_center=True)
    assert H8.shape == (8, 8)

def test_hops_simulator_initialization():
    """Test HopsSimulator initialization and fallback logic."""
    H, _ = create_fmo_hamiltonian(include_reaction_center=False)
    
    # Initialize without MesoHOPS (forced fallback)
    simulator = HopsSimulator(
        H, 
        temperature=300.0, 
        use_mesohops=False
    )
    
    assert simulator.temperature == 300.0
    assert not simulator.use_mesohops

def test_hops_simulator_simulation_stub():
    """Test a basic simulation call (even if stubbed/failed in environment)."""
    H, _ = create_fmo_hamiltonian(include_reaction_center=False)
    simulator = HopsSimulator(H, use_mesohops=False)
    
    time_points = np.linspace(0, 100, 10)
    initial_state = np.zeros(7)
    initial_state[0] = 1.0
    
    results = simulator.simulate_dynamics(time_points, initial_state)
    assert isinstance(results, dict)
    assert 't_axis' in results
    assert 'populations' in results

if __name__ == "__main__":
    pytest.main([__file__])
