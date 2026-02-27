# Quantum-Enhanced Agrivoltaics: Spectral Bath Engineering via Non-Markovian Dynamics

This repository contains the research framework, simulation codebase, and manuscript source for **"Spectral Bath Engineering for Quantum-Enhanced Agrivoltaics: Advancing Efficiency and Environmental Sustainability via Non-Markovian Dynamics"**. 

Our framework leverages the quantum mechanical nature of photosynthetic energy transfer to optimize agrivoltaic systems, co-locating solar energy generation with high-productivity agricultural production.

## 📄 Manuscript Status

- **Status**: Ready for Submission
- **Target Journal**: *Energy & Environmental Science* (EES)
- **Primary Objective**: Leveraging non-Markovian coherence in the FMO complex to enhance photosynthetic ETR under semi-transparent OPVs.

### Key Validated Metrics
- **Photosynthetic Enhancement**: \SI{25.0}{\percent} ETR increase via vibronic resonance matching.
- **Photovoltaic Performance**: \SI{18.83}{\percent} PCE for the balanced dual-band configuration.
- **System ETR**: \SI{80.51}{\percent} (retaining agricultural productivity under shading).
- **Eco-Compatibility**: PM6-derivative B-index of **101.5** (highly biodegradable).
- **Environmental Stability**: \SI{0.17}{\percent} annual degradation in both PCE and ETR.

---

## 🛠 Technical Stack & Environment

### Computational Engine
- **MesoHOPS (adHOPS) v1.6**: Core non-Markovian dynamics propagation.
- **PT-HOPS/SBD Framework**: Custom Python implementation for process tensor hierarchical equations of motion and spectrally bundled dissipators.
- **ORCA v6.1.0**: DFT calculations using **wB97X-D4** functional and **def2-SVP** basis set.

### Environment Specification
- **Hardware**: AMD Ryzen 5 5500U (6 cores, 12 threads) | \SI{40}{\giga\byte} RAM.
- **Language**: Python 3.12.12
- **Dependencies**: NumPy 2.0.2, SciPy 1.14.1, Matplotlib 3.10.x.

---

## 📂 Project Structure

```text
Redac_Paper1/
├── Q_Agrivoltaics_EES_Main.tex     # Main manuscript LaTeX source
├── Supporting_Info_EES.tex         # Supporting Information source
├── Cover_Letter_EES.tex            # Cover letter for EES
├── references.bib                   # Unified bibliography
├── Graphics/                        # Verified manuscript figures (PDF/PNG)
├── files2/                          # Modular LaTeX sections (Results, Discussion, etc.)
├── simulation_data/                 # Cleaned simulation logs (CSV data archived)
├── CSV_Data_Analysis.md             # Ground truth for all numerical results
├── INSTALLATION.md                  # Setup and dependency guide
└── quantum_simulations_framework/   # Core simulation codebase
    ├── quantum_coherence_agrivoltaics_mesohops_complete.py # Main driver
    ├── adhops/                      # Local MesoHOPS integration
    └── utils/                       # Visualization and patching utilities
```

---

## 🚀 Getting Started

### Installation
Please refer to [INSTALLATION.md](INSTALLATION.md) for detailed environment setup and dependency installation instructions.

### Running Simulations
The primary simulation suite can be executed via the main driver script:
```bash
python quantum_simulations_framework/quantum_coherence_agrivoltaics_mesohops_complete.py
```

### Building the Manuscript
To compile the submission documents, ensure you have a complete TeX distribution (e.g., TeX Live 2025) and use `latexmk`:
```bash
latexmk -pdf Q_Agrivoltaics_EES_Main.tex Supporting_Info_EES.tex Cover_Letter_EES.tex
```

---

## 📑 Core Research Contributions

1. **Spectral Bath Engineering**: A methodology for Identifying quantum-enhanced processes in nature and engineering artificial environments to maximize those resources.
2. **PT-HOPS & SBD**: Scalable non-Markovian dynamics framework achieving \num{10}x speedup over standard HEOM.
3. **Quantum Reactivity Descriptors**: Mechanistic prediction of OPV biodegradability via Fukui functions and chemical hardness.
4. **Pareto-Optimal Cogeneration**: Dual-band splitting strategy (\SI{440.4}{\nano\meter} and \SI{668.4}{\nano\meter}) for simultaneous energy-food maximization.

---

## 🤝 Citation & Contact

If you use this framework or data in your research, please cite our corresponding *Energy & Environmental Science* article.

**Corresponding Author**: Steve Cabrel Teguia Kouam ([steve.teguia@univ-douala.cm](mailto:steve.teguia@univ-douala.cm))