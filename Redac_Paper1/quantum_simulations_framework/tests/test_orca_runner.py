import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow importing from the framework
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from utils.orca_wrapper import OrcaRunner

def test_orca_runner_initialization():
    """Test that OrcaRunner initializes correctly with custom paths."""
    runner = OrcaRunner(work_dir="orca_test_dir")
    assert runner.work_dir == "orca_test_dir"
    assert os.path.exists("orca_test_dir")
    # Cleanup
    if os.path.exists("orca_test_dir") and not os.listdir("orca_test_dir"):
        os.rmdir("orca_test_dir")

def test_orca_input_generation():
    """Test that Orca input files are generated with correct content."""
    runner = OrcaRunner(work_dir="orca_test_input")
    coords = "O 0 0 0\nH 0 0 1\nH 0 1 0"
    file_path = runner.generate_input("test_mol", coords)
    
    assert os.path.exists(file_path)
    with open(file_path, "r") as f:
        content = f.read()
        assert "test_mol" not in content # Name is in filename
        assert "O 0 0 0" in content
        assert "TightSCF" in content
        
    # Cleanup
    os.remove(file_path)
    os.rmdir("orca_test_input")

@pytest.mark.skipif(not os.path.exists("/home/taamangtchu/orca_6_1_0/orca"), 
                    reason="Orca executable not found")
def test_orca_reactivity_descriptors():
    """
    Test actual reactivity descriptor calculation.
    Skipped if Orca is not available in the environment.
    """
    water_coords = """O 0.0000 0.0000 0.0622
H 0.0000 0.7842 -0.4935
H 0.0000 -0.7842 -0.4935"""
    
    runner = OrcaRunner(work_dir="orca_test_run")
    try:
        results = runner.calculate_reactivity_descriptors(
            "H2O_test", water_coords, method="HF", basis="sto-3g"
        )
        assert 'fukui_nucleophilic' in results
        assert 'homo_ev' in results
        assert 'lumo_ev' in results
        assert results['gap_ev'] == results['lumo_ev'] - results['homo_ev']
    finally:
        # Cleanup
        if os.path.exists("orca_test_run"):
            import shutil
            shutil.rmtree("orca_test_run")

if __name__ == "__main__":
    # Allow running directly with python
    pytest.main([__file__])
