import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the parent directory to sys.path to allow importing from the framework
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from utils.csv_data_storage import CSVDataStorage
from utils.figure_generator import FigureGenerator
from utils.logging_config import setup_logging, get_logger

def test_csv_data_storage():
    """Test CSV data storage operations."""
    storage = CSVDataStorage(output_dir="test_data")
    
    # Test saving quantum dynamics
    time_fs = np.linspace(0, 100, 10)
    populations = np.random.rand(10, 7)
    coherences = np.random.rand(10)
    metrics = {'purity': np.random.rand(10)}
    
    csv_path = storage.save_quantum_dynamics_results(time_fs, populations, coherences, metrics)
    
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    assert 'time_fs' in df.columns
    assert 'population_site_1' in df.columns
    
    # Cleanup
    if os.path.exists("test_data"):
        import shutil
        shutil.rmtree("test_data")

def test_figure_generator():
    """Test figure generation and path management."""
    gen = FigureGenerator(figures_dir="test_plots")
    
    # Smoke test: check that it at least exists and has the directory
    assert os.path.isdir(gen.figures_dir)
    
    # Cleanup
    if os.path.exists("test_plots"):
        import shutil
        shutil.rmtree("test_plots")

def test_logging_config():
    """Test logging initialization."""
    import logging
    # This just ensures it doesn't crash
    setup_logging(level=logging.INFO)
    logger = get_logger("test_logger")
    logger.info("Test message")
    assert logger.name == "quantum_simulations_framework.test_logger"

if __name__ == "__main__":
    import logging
    pytest.main([__file__])
