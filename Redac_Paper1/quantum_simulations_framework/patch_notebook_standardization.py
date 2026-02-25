import nbformat
import os

def standardize_notebook():
    nb_path = "/media/taamangtchu/MYDATA/Github/Quantum_Agrivoltaic_HOPS/Redac_Paper1/quantum_simulations_framework/quantum_coherence_agrivoltaics_mesohops_complete.ipynb"
    if not os.path.exists(nb_path):
        print(f"Error: {nb_path} not found.")
        return

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == "code":
            source = cell.source
            
            # 1. Clean up imports in Cell 1
            if "Import required libraries" in source and "TechnoEconomicModel" in source:
                source = source.replace("from models.techno_economic_model import TechnoEconomicModel\n", "")
                source = source.replace("from models.spectroscopy_2des import Spectroscopy2DES\n", "")
                # Add consolidated import
                if "import pandas as pd" in source:
                    source = source.replace("import pandas as pd\n", "import pandas as pd\nfrom models import TechnoEconomicModel, Spectroscopy2DES\n")

            # 2. Standardize all framework imports in Cell 5 (or any import cell)
            if "from core.constants import" in source and "FMO_SITE_ENERGIES_7" in source:
                # Remove relative dots
                source = source.replace("from .core.constants", "from core.constants")
                source = source.replace("from .core.hops_simulator", "from core.hops_simulator")
                source = source.replace("from .models.", "from models.")
                source = source.replace("from .simulations.", "from simulations.")
                source = source.replace("from .utils.", "from utils.")
                
                # Consolidate models
                if "from models.biodegradability_analyzer" in source:
                    model_import = """# Import models
from models import (
    BiodegradabilityAnalyzer,
    SensitivityAnalyzer,
    LCAAnalyzer,
    TechnoEconomicModel,
    Spectroscopy2DES,
    MultiScaleTransformer,
    QuantumDynamicsSimulator,
    AgrivoltaicCouplingModel,
    SpectralOptimizer,
    EcoDesignAnalyzer,
    EnvironmentalFactors
)
from simulations import TestingValidationProtocols
from utils import CSVDataStorage
from utils.figure_generator import FigureGenerator
"""
                    # Try to replace the whole block from "# Import models" to the end of fallbacks
                    import re
                    pattern = r"# Import models.*?(?=print\()" # Matches until print
                    source = re.sub(pattern, model_import + "\n", source, flags=re.DOTALL)

            # 3. Fix broken import in Cell 6
            if "from quantum_coherence_agrivoltaics_mesohops import create_fmo_hamiltonian" in source:
                source = source.replace("from quantum_coherence_agrivoltaics_mesohops import create_fmo_hamiltonian", 
                                       "from core.hamiltonian_factory import create_fmo_hamiltonian")

            cell.source = source

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Notebook standardized successfully.")

if __name__ == "__main__":
    standardize_notebook()
