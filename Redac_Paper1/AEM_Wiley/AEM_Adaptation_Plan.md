# Advanced Energy Materials (Wiley) - Paper Adaptation Plan

**Manuscript:** Molecular engineering of semi-transparent non-fullerene acceptors for quantum-enhanced agrivoltaics  
**Target Journal:** Advanced Energy Materials (Impact Factor: 24.4)  
**Document Version:** 4.0 (Reorganized for Efficiency)  
**Last Updated:** March 20, 2026  
**Status:** Ready for Implementation

---

## Quick Navigation

| Section | Purpose | Priority |
|---------|---------|----------|
| [1. Critical Requirements](#1-critical-requirements) | Must-do items before submission | 🔴 HIGH |
| [2. Data Verification](#2-data-verification) | Trace every claim to CSV files | 🔴 HIGH |
| [3. Manuscript Structure](#3-manuscript-structure) | Section-by-section guidance | 🟡 MEDIUM |
| [4. Figures & Tables](#4-figures--tables) | Complete figure specifications | 🟡 MEDIUM |
| [5. Supporting Information](#5-supporting-information) | SI organization | 🟡 MEDIUM |
| [6. Language & Style](#6-language--style) | Writing standards | 🟢 LOW |
| [7. Submission Checklist](#7-submission-checklist) | Final verification | 🔴 HIGH |

---

## 1. Critical Requirements

### 1.1 Nature of This Work

**This is a THEORETICAL/COMPUTATIONAL study:**
- ✅ Quantum dynamics simulations (PT-HOPS/SBD)
- ✅ AI-driven materials discovery (E(n)-GNN)
- ✅ Literature-based OPV parameters
- ✅ Techno-economic modeling
- ❌ **NO** experimental fabrication
- ❌ **NO** experimental measurements

**Language must reflect this:**
- Use: "simulations predict", "based on literature", "projected", "can achieve"
- Avoid: "we fabricated", "we measured", "our devices", "experimental results"

### 1.2 Wavelength Terminology (CRITICAL)

**DISCREPANCY IDENTIFIED - Must Fix:**

| Term | Wavelengths | What It Means |
|------|-------------|---------------|
| **OPV absorption bands** | 440 nm, 668 nm | What the OPV **absorbs** (from `spectral_optimization_*.csv`) |
| **Transmission windows** | 750 nm, 820 nm | What **passes through** to plants (FMO vibronic resonances) |

**Action:** Update ALL manuscript sections to use precise terminology consistently.

### 1.3 No AI-Generated Language

**STRICTLY AVOID:**
```
❌ "In this study, we investigate..."
❌ "The results show that..."
❌ "Furthermore," / "Moreover," / "Additionally" (repetitive)
❌ "It was found that..." (passive voice)
❌ Vague statements without numbers
```

**USE INSTEAD:**
```
✅ "Quantum dynamics simulations yield..."
✅ "PCE = 18.83% (Table 1)"
✅ Direct transitions following logical flow
✅ "We find..." (active voice)
✅ Every claim with specific numerical support
```

### 1.4 Verifiability Requirement

**EVERY statement MUST be:**
1. Traceable to `simulation_data/` CSV files, OR
2. Supported by literature citation, OR
3. Clearly identified as projection ("projected", "potential")

**Data Traceability Matrix:**

| Manuscript Claim | Source File | Verification |
|-----------------|-------------|--------------|
| PCE = 18.83% | `spectral_optimization_*.csv` | Row: `PCE,0.1882733522886984` |
| ETR = 80.51% | `spectral_optimization_*.csv` | Row: `ETR,0.8051448488963987` |
| 50% transfer at ~30 fs | `quantum_dynamics_*.csv` | Site 1: 1.0 → 0.5 at t≈30 fs |
| Peak coherence 0.988 | `quantum_dynamics_*.csv` | Max(coherences) = 0.988 |
| Annual degradation 0.17% | `environmental_effects_*.csv` | (Day0 - Day365) / Day0 |
| B-index = 101.5 | `eco_design_results_*.csv` | Column: `b_index` |

---

## 2. Data Verification

### 2.1 Available Simulation Data

**Location:** `Quantum_Agrivoltaic_HOPS/simulation_data/`

| File | Records | Key Metrics | Use In Manuscript |
|------|---------|-------------|-------------------|
| `quantum_dynamics_*.csv` | 501 (0--1000 fs) | Populations, coherence, QFI, entropy | Figs 3, S6, S7 |
| `spectral_optimization_*.csv` | 12 | PCE, ETR, filter params | Figs 2, 4 |
| `agrivoltaic_results_*.csv` | 2 | System PCE, ETR | Abstract, Results |
| `eco_design_results_*.csv` | 1 | B-index, sustainability | SI Eco-design section |
| `environmental_effects_*.csv` | 365 days | PCE/ETR vs. time, temp, dust | Fig 6 |

### 2.2 Claims Verification Checklist

**Abstract:**
- [ ] "18.8% PCE" → `spectral_optimization_*.csv` ✅
- [ ] "750 and 820 nm" → Clarify as transmission windows ✅
- [ ] "20-50% coherence extension" → Add comparison baseline data ⚠️
- [ ] "25% ETR enhancement" → Add filtered vs. unfiltered comparison ⚠️
- [ ] "18,000 hours stability" → Add "based on literature reports" ⚠️
- [ ] "$0.04-0.06/kWh" → Add LCOE calculation in SI ✅

**Results:**
- [ ] All percentages → CSV files
- [ ] All wavelengths → `spectral_optimization_*.csv`
- [ ] All "enhancement" claims → Baseline comparison included
- [ ] All stability claims → "projected" or "literature-based" qualifier

---

## 3. Manuscript Structure

### 3.1 Section Order & Content

```
Q_Agrivoltaics_AEM_Main.tex
├── Title (sentence case)
├── Abstract (150-250 words, lead with PCE)
├── Keywords (lowercase, comma-separated)
├── Introduction (900-1100 words)
│   └── OPV state-of-the-art → Challenge → Quantum mechanism → Contributions
├── Theory and Methods
│   ├── Open quantum system framework
│   ├── PT-HOPS/SBD formalism
│   ├── AI/ML pipeline (E(n)-GNN)
│   └── FMO parameters
├── OPV Materials Design (NEW)
│   ├── Molecular design strategy (literature-based)
│   ├── Charge transport properties
│   ├── Recombination kinetics
│   └── Proposed device architecture
├── Results
│   ├── Quantum dynamics (coherence, population, QFI)
│   ├── Spectral engineering (transmission windows)
│   ├── Pareto optimization (PCE vs. ETR)
│   ├── Environmental robustness
│   └── Geographic validation
├── Discussion
│   ├── Materials design principles
│   ├── Structure-property relationships
│   ├── Economic analysis (LCOE, revenue)
│   ├── Implementation pathway
│   └── Limitations
├── Conclusion (lead with 18.8% PCE)
├── Acknowledgments
├── Data Availability Statement
├── Conflicts of Interest
├── Author Contributions (lowercase roles)
└── References (60-80, AEM-relevant)
```

### 3.2 Figure Placement

| Figure | Placement | Type | Status |
|--------|-----------|------|--------|
| Fig 1 | After Introduction | Conceptual schematic | ⚠️ Create |
| Fig 2 | After OPV Materials | OPV performance | ⚠️ Create from CSV |
| Fig 3 | After Quantum Results | Coherence dynamics | ⚠️ Create from CSV |
| Fig 4 | After Optimization | Pareto frontier | ⚠️ Create |
| Fig 5 | After AI/ML Results | Pipeline schematic | ⚠️ Create |
| Fig 6 | After Geographic | Environmental robustness | ⚠️ Create from CSV |
| Fig 7 | Optional | Spectral analysis | ⚠️ Create |
| Fig 8 | Optional | Design guidelines | ⚠️ Create |

---

## 4. Figures & Tables

### 4.1 Main Manuscript Figures (6-8 Required)

#### Figure 1: Conceptual Overview (Double Column)
**File:** `Graphics/Figure_1_Conceptual_Overview.png`
**Content:** Agrivoltaic system schematic + molecular structures + energy levels
**Caption:** See `files2/figure_1_conceptual.tex`
**Status:** ⚠️ Create new

#### Figure 2: OPV Performance (Single Column)
**File:** `Graphics/Figure_2_OPV_Performance.png`
**Data Sources:** 
- `spectral_optimization_*.csv` → PCE = 0.1883
- `agrivoltaic_results_*.csv` → System metrics
- Literature → J-V curve shape, EQE (label as "literature-based")
**Caption:** See `files2/figure_2_opv_performance.tex`
**Status:** ⚠️ Generate from CSV + literature

#### Figure 3: Quantum Dynamics (Double Column)
**File:** `Graphics/Figure_3_Quantum_Dynamics.png`
**Data Source:** `quantum_dynamics_*.csv` (501 time steps)
**Plots:**
- (a) Coherence vs. time (filtered vs. broadband)
- (b) Population dynamics (Site 1 → RC)
- (c) Delocalization (IPR)
- (d) QFI evolution
**Status:** ✅ Data available, generate plots

#### Figure 4: Pareto Frontier (Single Column)
**File:** `Graphics/Figure_4_Pareto_Frontier.png`
**Data Source:** `spectral_optimization_*.csv` + framework logs
**Status:** ⚠️ Re-run optimization with population logging OR add disclaimer

#### Figure 5: AI/ML Pipeline (Single Column)
**File:** `Graphics/Figure_5_AI_ML_Pipeline.png`
**Data Source:** Framework configuration + logs
**Status:** ⚠️ Extract timing data from logs

#### Figure 6: Environmental Robustness (Double Column)
**File:** `Graphics/Figure_6_Environmental_Robustness.png`
**Data Source:** `environmental_effects_*.csv` (365 days)
**Plots:**
- (a) Temperature dependence
- (b) Static disorder effects
- (c) Geographic map with 9 zones
- (d) Bath parameter fluctuations
**Status:** ✅ Data available, generate plots

### 4.2 Tables

| Table | Content | Location | Status |
|-------|---------|----------|--------|
| Table 1 | Quantum metrics comparison | Results | ⚠️ Create |
| Table 2 | OPV specifications | Discussion | ⚠️ Create |
| Table 3 | Economic analysis | Discussion | ⚠️ Create |
| Table S1 | Validation tests (12) | SI | ✅ In SI |
| Table S2 | FMO parameters | SI | ✅ In SI |
| Table S3 | Computational performance | SI | ✅ In SI |
| Table S4 | AI/ML hyperparameters | SI | ✅ In SI |

---

## 5. Supporting Information

### 5.1 Organization

```
Supporting_Info_AEM.tex
├── OPV materials synthesis protocols (literature-based)
├── Device fabrication procedures (literature-based)
├── Quantum dynamics methods (PT-HOPS/SBD technical)
├── AI/ML model architecture (E(n)-GNN specs)
├── Stability testing protocols (ISOS standards, literature)
├── Economic analysis details (LCOE calculations)
├── Validation data (12 tests expanded)
├── Figures S1-S8
└── Tables S1-S4
```

### 5.2 SI Figures (8 Total)

| Figure | Content | Data Source | Status |
|--------|---------|-------------|--------|
| S1 | Molecular structures | Literature | ⚠️ Create |
| S2 | J-V and EQE | Literature | ⚠️ Create |
| S3 | Stability testing | Literature + `environmental_effects_*.csv` | ⚠️ Create |
| S4 | AI/ML performance | Framework logs | ⚠️ Create |
| S5 | Pareto analysis | `spectral_optimization_*.csv` | ⚠️ Create |
| S6 | Validation | Framework comparison | ⚠️ Create |
| S7 | Temperature/disorder | `quantum_dynamics_*.csv` | ⚠️ Create |
| S8 | Economics | Calculated | ⚠️ Create |

**All SI figure captions:** See `Supporting_Info_AEM.tex` lines 408-560

---

## 6. Language & Style

### 6.1 Capitalization (Sentence Case)

| Element | Format | Example |
|---------|--------|---------|
| Section titles | Sentence case | `\section{OPV materials design}` |
| Subsections | Sentence case | `\subsection{Charge transport properties}` |
| Figure captions | Sentence case | `\caption{\textbf{Quantum dynamics simulations.}...}` |
| Bold list items | Sentence case + period | `\item \textbf{Record performance.} Text...` |
| Author roles | Lowercase | `Methodology, validation, formal analysis` |
| Proper nouns | Capitalized | FMO, PM6, Y6-BO, DFT, HEOM |

### 6.2 siunitx Macros (Required)

```latex
% Correct usage:
\SI{18.8}{\percent}              % Not: 18.8%
\SI{295}{\kelvin}                % Not: 295 K
\SIlist{750;820}{\nano\meter}    % Multiple values
\SIrange{20}{50}{\percent}       % Range
\num{2847}                       % Not: 2,847
\numrange{470}{3000}             % Number range
\SIrange{0.04}{0.06}{\USD\per\kWh} % Cost with unit
```

### 6.3 physics Macros (Required)

```latex
% Derivatives:
\pdv{\bm{\rho}(t)}{t}            % Partial derivative

% Bra-ket:
\ket{\psi}                       % Ket
\bra{\phi}                       % Bra
\braket{\psi|\phi}               % Inner product
\dyad{n}{m}                      % Outer product

% Operators:
\Tr[\bm{\rho}]                   % Trace
\comm{\mathtt{H}_S}{\bm{\rho}}   % Commutator
```

### 6.4 Equation Punctuation

```latex
% Period after equation (sentence ends):
The efficiency is:
\begin{equation}
\mathrm{PCE} = \frac{J_{\rm SC} V_{\rm OC} \mathrm{FF}}{P_{\rm in}}.
\end{equation}

% Comma after equation (sentence continues):
The Hamiltonian is:
\begin{equation}
\mathtt{H} = \mathtt{H}_S + \mathtt{H}_B + \mathtt{H}_{SB},
\end{equation}
where $\mathtt{H}_S$ is the system Hamiltonian.
```

---

## 7. Submission Checklist

### 7.1 Pre-Submission (1 Week Before)

- [ ] All figures generated from CSV data
- [ ] All captions use sentence case + siunitx
- [ ] All "???" citations replaced
- [ ] Wavelength terminology consistent (440/668 vs. 750/820)
- [ ] All "literature-based" claims have citations
- [ ] All "projected" claims clearly qualified
- [ ] Data availability statement added
- [ ] Verification table in SI (claim → CSV → value)

### 7.2 Formatting (2 Days Before)

- [ ] Reference count: 60-80
- [ ] Word count: 6000-8000 (main text)
- [ ] Figure resolution: 300 dpi minimum
- [ ] Figure format: TIFF/EPS/PDF
- [ ] Graphical abstract: 1000×500 px, PNG/TIFF
- [ ] SI compiled as separate PDF
- [ ] All files named per AEM conventions

### 7.3 Final Review (1 Day Before)

- [ ] No AI-generated language patterns
- [ ] Narrative flow: logical progression
- [ ] Every claim verifiable
- [ ] All co-authors approved
- [ ] Cover letter prepared
- [ ] Suggested reviewers identified (3-5)
- [ ] ORCID IDs collected

### 7.4 Submission Files

| File | Format | Size Limit | Required |
|------|--------|------------|----------|
| Main manuscript | PDF | 50 MB | ✅ |
| Supporting Info | PDF | 100 MB | ✅ |
| Cover letter | PDF | 10 MB | ✅ |
| Graphical abstract | PNG/TIFF | 10 MB | ✅ |
| Author ORCIDs | Text | - | ✅ |
| Suggested reviewers | Text | - | ✅ |

---

## Appendix A: Simulation Data Quick Reference

### A.1 Key Values from CSV Files

```csv
# spectral_optimization_20260225_170559.csv
metric,value
PCE,0.1882733522886984          # → 18.83%
ETR,0.8051448488963987          # → 80.51%
param_1,668.4200374667248       # → OPV absorption center 1
param_4,440.4460195921225       # → OPV absorption center 2

# quantum_dynamics_20260225_170638.csv (first rows)
time_fs,coherences,population_site_1
0.0,1.08e-15,1.0                # Initial: all on Site 1
~30,~2.0,~0.5                   # 50% transfer at ~30 fs
peak,0.988,-                    # Peak coherence

# environmental_effects_20260225_170850.csv
time_days,pce_with_environment
0,0.1687877                     # Day 0 PCE
365,0.1685                      # Day 365 PCE
# Degradation: (0.1688 - 0.1685) / 0.1688 = 0.0017 = 0.17%

# eco_design_results_20260225_170638.csv
material_name,b_index,sustainability_score
PM6 Derivative (Molecule A),101.5,1.12
```

### A.2 Python Script Template for Figure Generation

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load quantum dynamics data
df = pd.read_csv('simulation_data/quantum_dynamics_20260225_170638.csv')

# Plot coherence
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(df['time_fs'], df['coherences'], 'g-', label='Filtered')
ax.set_xlabel('Time (fs)')
ax.set_ylabel('Coherence (l₁-norm)')
ax.legend()
plt.savefig('Graphics/Figure_3_Quantum_Dynamics.png', dpi=300)
```

---

## Appendix B: Cover Letter Template

```latex
Dear Dr. Beller,

We submit "Molecular engineering of semi-transparent non-fullerene acceptors 
for quantum-enhanced agrivoltaics" for consideration in Advanced Energy 
Materials.

This computational study presents:
1. Simulation-predicted \SI{18.8}{\percent} PCE for semi-transparent OPV
2. Quantum-coherent spectral engineering mechanism
3. AI-driven materials discovery (\num{2847} candidates screened)
4. Literature-based stability projections $>$\SI{18000}{hours}
5. Techno-economic modeling: LCOE \$$\num{0.04}$--\$$\num{0.06}/kWh

All authors approve. No conflicts of interest.

Sincerely,
Steve Cabrel Teguia Kouam
```

---

## Appendix C: File Structure

```
AEM_Wiley/
├── Q_Agrivoltaics_AEM_Main.tex          # Main manuscript
├── Supporting_Info_AEM.tex              # Supporting Information
├── Cover_Letter_AEM.tex                 # Cover letter
├── AEM_Adaptation_Plan.md               # This document
├── references.bib                       # References
├── files2/
│   ├── introduction.tex
│   ├── theory_methods.tex
│   ├── opv_materials.tex
│   ├── results.tex
│   ├── discussion.tex
│   ├── conclusion.tex
│   ├── figure_1_conceptual.tex          # Figure captions
│   ├── figure_2_opv_performance.tex
│   ├── figure_3_quantum_dynamics.tex
│   ├── figure_4_pareto_frontier.tex
│   ├── figure_5_ai_ml_pipeline.tex
│   ├── figure_6_environmental_robustness.tex
│   └── figures_main.tex                 # All figures combined
└── Graphics/                            # Create this folder
    ├── Figure_1_Conceptual_Overview.png
    ├── Figure_2_OPV_Performance.png
    ├── Figure_3_Quantum_Dynamics.png
    ├── Figure_4_Pareto_Frontier.png
    ├── Figure_5_AI_ML_Pipeline.png
    ├── Figure_6_Environmental_Robustness.png
    ├── si_figure_s1.png through si_figure_s8.png
    └── Graphical_Abstract_AEM.png
```

---

**Document Version:** 4.0 (Reorganized for Efficiency)  
**Status:** Ready for Implementation  
**Next Action:** Generate figures from CSV data (Section 4)
