# Submission TODO (BMAD Style)

**Goal**: Finalize package for Energy & Environmental Science (EES) submission.
**Priority**: High (P0)
**Status**: Sprint 5 Complete → Final Polish

---

## 📋 1. Technical Refinement (Actionable)
- [ ] **Manuscript Wrapper**: Replace `[REPOSITORY_URL]` in `Q_Agrivoltaics_EES_Main.tex` (Line 111).
- [ ] **Macro Uniformity**: Scan `files2/` to ensure `physics` ($\ket{\psi}$) and `siunitx` (\SI{750}{\nano\meter}) are used consistently.
- [ ] **Placeholder Check**: Verify no `[TBD]` or `[TEMP]` strings remain in `results.tex` or `discussion.tex`.

## 🖼️ 2. Graphics & Visualization (⚠️ Critical Audit)
> [!WARNING]
> Individual figure files (e.g., Fig 1–7) appear missing from the `/Graphics/` directory. 
- [ ] **Asset Recovery**: Locate or re-generate individual high-res figures (PDF/PNG) required for the RSC portal upload.
- [ ] **Graphical Abstract**: Verify `Graphics/Graphical_Abstract_EES.png` exists and meets RSC dimensions (5cm x 5cm) and 600dpi quality.
- [ ] **Figure Callouts**: Sequential check of Fig 1–7 citations in the main text to ensure logical flow.
- [ ] **SI Figures**: Ensure Figures S1–S8 in `Supporting_Info_EES.tex` are correctly referenced.

## 📚 3. Bibliography & Compliance
- [ ] **RSC Formatting**: Verify `references.bib` entries have complete DOIs and use the `unsrt` (numbered) style.
- [ ] **Citation Sync**: Ensure every entry in `references.bib` is actually cited at least once (cleanup unused keys).
- [ ] **SI References**: Ensure SI-specific citations are properly integrated.

## 📜 4. Supporting Information (SI)
- [ ] **Validation Tables**: Full audit of the 12-test validation results presented in SI for numerical accuracy.
- [ ] **Eco-Design Data**: Ensure `BiodegradabilityAnalyzer` output values in SI match the final framework results ($B_{\rm index}$).
- [ ] **Archive Prep**: Bundle `quantum_simulations_framework/` and `quantum_coherence_agrivoltaics_mesohops_complete.ipynb` for easy repository upload.

## 🚀 5. Final Submission Milestones
- [ ] **Cover Letter**: Finalize reviewer suggestions (3-5 names) in `Cover_Letter_EES.tex`.
- [ ] **Full Compile**: Run `pdflatex` + `bibtex` + `pdflatex` (x2) to ensure zero broken links/citations.
- [ ] **Submission**: Upload to RSC portal.

---
**BMAD Note**: Keep it Brief. Make it Meaningful. Ensure it's Actionable. Proceed Directly.
