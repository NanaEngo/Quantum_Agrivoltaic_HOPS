import os
import sys
import numpy as np
from pathlib import Path

# Add the parent directory to sys.path to allow importing from the framework
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from models.quantum_dynamics_simulator import QuantumDynamicsSimulator, MESOHOPS_AVAILABLE
from models.spectroscopy_2des import Spectroscopy2DES
from core.hamiltonian_factory import create_fmo_hamiltonian

def test_quantum_dynamics_simulator():
    """Test the standalone QuantumDynamicsSimulator."""
    if not MESOHOPS_AVAILABLE:
        pytest.skip("MesoHOPS not installed")
        
    H, _ = create_fmo_hamiltonian(include_reaction_center=False)
    simulator = QuantumDynamicsSimulator(H)
    
    time_points = np.linspace(0, 100, 20)
    initial_state = np.zeros(7, dtype=complex)
    initial_state[0] = 1.0
    
    results = simulator.simulate_dynamics(time_points, initial_state)
    
    assert 'populations' in results
    assert results['populations'].shape == (20, 7)
    # Conservation check (approximate due to dephasing/bath modeling)
    total_pop = np.sum(results['populations'], axis=1)
    assert np.allclose(total_pop, 1.0, atol=0.1)

def test_spectroscopy_2des():
    """Test 2D Electronic Spectroscopy (2DES) simulation."""
    H, _ = create_fmo_hamiltonian(include_reaction_center=False)
    spec = Spectroscopy2DES(system_size=7)
    
    # Run spectrum at waiting time T=0
    results = spec.simulate_2d_spectrum(H, waiting_time=0.0)
    
    assert 'spectrum' in results
    assert results['spectrum'].shape == (128, 128) # Default resolution
    assert 'omega_exc' in results
    assert 'omega_det' in results
    
    # Test plotting path (smoke test)
    fig_path = spec.plot_2d_spectrum(results, output_dir="test_graphics")
    assert os.path.exists(fig_path)
    
    # Cleanup
    if os.path.exists("test_graphics"):
        import shutil
        shutil.rmtree("test_graphics")

if __name__ == "__main__":
    pytest.main([__file__])
