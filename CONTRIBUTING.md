# Contributing to VQC-ZTI Framework

Thank you for your interest in contributing to the VQC-ZTI Framework. We welcome community extensions, optimization tweaks, and bug fixes that advance reproducible research for hybrid quantum-classical zero-trust network architectures.

## Development Rules & Environment

1. **Environment Alignment**: 
   Ensure your local virtual environment closely mirrors the project dependencies specified in `requirements.txt`:
   * Python (recommended version used in your pipeline script execution)
   * PyTorch
   * PennyLane

2. **Testing and Verification**:
   Before finalizing any script adjustment, verify that code modifications do not break end-to-end model training, the computation of quantum expectation values, or data processing. Run the primary project script to validate pipeline health:
   ```powershell
   .\run_pipeline.ps1
   ```

## Model Integrity & Ablation Alignment

When proposing structural changes to the circuit architecture or embedder logic, please consider how your contribution impacts the five model variants established for the benchmark study:
* **Full Hybrid (FH)** — Complete embedder + VQC + classification head
* **No Head (NH)** — Classification head omitted
* **Shallow Embedder (SE)** — Trimmed embedding architecture
* **No Embedder (NE)** — Processing features directly mapped
* **PQC-only (PQC)** — Parameterized circuit without the hybrid shell

## Commit and Artifact Policies

* **Do Not Commit Raw Datasets or Training Tensors**: Large, intermediate local artifacts, processed raw matrices, or local checkpoint data must remain untracked by Git to keep the branch hierarchy optimized.
* **Stable Formats**: Any modified visual evaluation plots or graphs exported to the `artifacts/` folder should be generated in stable vector or image layers (`.png`, `.svg`, `.pdf`).
