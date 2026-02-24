from models.eco_design_analyzer import EcoDesignAnalyzer
import numpy as np

analyzer = EcoDesignAnalyzer()
example_electron_densities = {
    'neutral': np.array([0.1, 0.15, 0.12, 0.18, 0.14, 0.16, 0.13, 0.17, 0.11, 0.19]),
    'n_plus_1': np.array([0.08, 0.13, 0.10, 0.16, 0.12, 0.14, 0.11, 0.15, 0.09, 0.17]),
    'n_minus_1': np.array([0.12, 0.17, 0.14, 0.20, 0.16, 0.18, 0.15, 0.19, 0.13, 0.21])
}

result_a = analyzer.evaluate_material_sustainability(
    "Molecule A", pce=0.155, ionization_potential=5.4, electron_affinity=3.2,
    electron_densities=example_electron_densities, molecular_weight=2000.0, bde=285.0
)

result_b = analyzer.evaluate_material_sustainability(
    "Molecule B", pce=0.152, ionization_potential=5.6, electron_affinity=3.8,
    electron_densities={
        'neutral': np.array([0.1, 0.12, 0.1, 0.15, 0.12, 0.14, 0.11, 0.16, 0.1, 0.18]),
        'n_plus_1': np.array([0.09, 0.11, 0.09, 0.14, 0.11, 0.13, 0.10, 0.15, 0.09, 0.17]),
        'n_minus_1': np.array([0.11, 0.13, 0.11, 0.16, 0.13, 0.15, 0.12, 0.17, 0.11, 0.19])
    },
    molecular_weight=2000.0, bde=310.0
)

print(f"Molecule A B-index: {result_a['b_index']:.2f} (Paper target: 72)")
print(f"Molecule B B-index: {result_b['b_index']:.2f} (Paper target: 58)")
