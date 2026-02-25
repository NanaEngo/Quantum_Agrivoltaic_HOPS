import os
import sys
import numpy as np
from pathlib import Path

# Add the parent directory to sys.path to allow importing from the framework
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from models.lca_analyzer import LCAAnalyzer
from models.techno_economic_model import TechnoEconomicModel
from models.eco_design_analyzer import EcoDesignAnalyzer
from models.biodegradability_analyzer import BiodegradabilityAnalyzer
from core.hamiltonian_factory import create_fmo_hamiltonian

def test_lca_analyzer():
    """Test Life Cycle Assessment impact calculations."""
    analyzer = LCAAnalyzer(system_lifetime=25, system_efficiency=0.18)
    
    # Calculate impact for standard parameters
    results = analyzer.calculate_lca_impact(
        manufacturing_energy=1500.0,
        operational_time=25.0,
        material_mass=0.3
    )
    
    assert 'total_carbon_kg_co2eq' in results
    assert 'energy_payback_time_years' in results
    assert results['energy_payback_time_years'] > 0
    assert results['carbon_footprint_gco2eq_per_kwh'] < 100.0

def test_techno_economic_model():
    """Test financial viability calculations."""
    model = TechnoEconomicModel(discount_rate=0.07, system_lifetime=25.0)
    
    results = model.evaluate_project_viability(
        area_hectares=10.0,
        pv_coverage_ratio=0.3,
        pce=0.18,
        etr=0.8
    )
    
    assert 'npv_usd' in results
    assert 'lcoe_usd_per_kwh' in results
    assert 'total_revenue_yr_usd_per_ha' in results
    assert results['total_revenue_yr_usd_per_ha'] > 0

def test_eco_design_analyzer():
    """Test sustainability evaluation and scoring."""
    analyzer = EcoDesignAnalyzer()
    
    # Mock data for Molecule A
    densities = {
        'neutral': np.array([0.1, 0.2, 0.1]),
        'n_plus_1': np.array([0.05, 0.15, 0.05]),
        'n_minus_1': np.array([0.15, 0.25, 0.15])
    }
    
    result = analyzer.evaluate_material_sustainability(
        "Test_Molecule",
        pce=0.15,
        ionization_potential=5.4,
        electron_affinity=3.2,
        electron_densities=densities
    )
    
    assert 'b_index' in result
    assert 'sustainability_score' in result
    assert 0 <= result['sustainability_score'] <= 1.0

def test_biodegradability_analyzer():
    """Test biodegradability scoring specifically."""
    H, _ = create_fmo_hamiltonian(include_reaction_center=False)
    analyzer = BiodegradabilityAnalyzer(molecular_hamiltonian=H, n_electrons=10)
    
    result = analyzer.calculate_biodegradability_score()
    
    assert 0 <= result <= 1.0

if __name__ == "__main__":
    pytest.main([__file__])
