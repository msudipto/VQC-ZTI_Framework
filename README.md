# VQC-ZTI Framework  
**Variational Quantum-Classical Zero-Trust Anomaly Detection and CESNET-Based Security Evaluation**

---

## Abstract  

This repository presents the full experimental and implementation workflow for **VQC-ZTI**, a **variational quantum-classical zero-trust framework** for anomaly detection, risk-adaptive access control, and secure next-generation network evaluation.  
The framework integrates **hybrid quantum-classical learning**, **zero-trust security logic**, and **CESNET-derived aggregated traffic analysis** to support reproducible experimentation on encrypted-flow anomaly detection, micro-segmentation, and policy-aware risk evaluation.  
The implementation is designed for research-grade reproducibility and supports preprocessing, model training, evaluation, ablation studies, and artifact generation for manuscript development and experimental verification.

---

## Repository Structure  

```text
VQC-ZTI-Framework/
├── artifacts/                          # Curated experiment outputs, figures, checkpoints, and final results
├── config/                             # Configuration files for preprocessing, training, evaluation, and experiments
├── data/                               # Dataset placement instructions, local raw/processed data layout, and documentation
├── src/                                # Core source code for preprocessing, training, evaluation, and utilities
│
├── README.md                           # Repository overview and reproducibility guide
├── requirements.txt                    # Python dependency specification
└── run_pipeline.ps1                    # End-to-end Windows PowerShell execution pipeline
```

---

## Environment Configuration  

### Requirements  
- Python ≥ 3.10  
- PyTorch  
- PennyLane  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Windows PowerShell for the provided pipeline script  
- Git LFS for selected large research assets

### Installation  
```bash
python -m venv .venv
source .venv/bin/activate       # (Windows) .venv\Scripts\activate
pip install -r requirements.txt
git lfs install
```

### Repository Setup  
Clone the repository and initialize the local environment:

```bash
git clone https://github.com/<your-username>/VQC-ZTI-Framework.git
cd VQC-ZTI-Framework
```

If the repository contains LFS-tracked files, retrieve them using:

```bash
git lfs pull
```

---

## Experimental Methodology  

### Data Preparation  
The framework is designed to operate on **CESNET-derived aggregated network traffic data** or compatible structured telemetry.  
Raw traffic records are placed locally under the `data/` directory, after which the preprocessing pipeline performs cleaning, feature conversion, normalization, filtering, and experiment-ready dataset construction.

### Preprocessing Workflow  
The preprocessing stage typically performs:

- ingestion of raw CESNET-style aggregated records
- missing-value handling and numeric feature conversion
- optional outlier filtering and feature cleaning
- train/evaluation split construction
- export of processed modeling-ready datasets

All data placement conventions and reproducibility notes are documented in `data/README.md`.

### Training Workflow  
The training pipeline supports **variational quantum-classical anomaly detection** under a zero-trust evaluation setting.  
Depending on configuration, the model pipeline may include:

- classical feature preprocessing
- quantum feature encoding
- variational quantum circuit training
- hybrid optimization over train/eval splits
- metric logging for anomaly detection and classification behavior

### Evaluation and Analysis  
The evaluation workflow is intended to support paper-grade analysis, including:

- model performance measurement across controlled splits
- anomaly detection metrics
- zero-trust risk evaluation behavior
- comparative baselines and ablation studies
- generation of publication-ready outputs and experiment artifacts

### Execution  
For the default Windows workflow, run:

```powershell
.\run_pipeline.ps1
```

This script is intended to coordinate preprocessing, model execution, evaluation, and export of reproducible outputs according to the active repository configuration.

---

## Result Summary  

| Component | Purpose | Research Role |
|----------|---------|---------------|
| **Preprocessing Pipeline** | Converts raw telemetry into experiment-ready inputs | Supports consistent and reproducible dataset formation |
| **Hybrid VQC Model** | Performs anomaly-sensitive quantum-classical learning | Enables evaluation of variational quantum methods for security analytics |
| **Zero-Trust Evaluation Logic** | Maps anomaly evidence into policy-aware risk behavior | Supports access-control and micro-segmentation analysis |
| **Ablation and Artifact Workflow** | Produces structured experiment comparisons and outputs | Facilitates manuscript figures, tables, and reproducible reporting |

---

## Reproducibility Notes  

- All experiment behavior should be governed through version-controlled configuration files under `config/`.
- Random seeds should be fixed where supported to improve repeatability across runs.
- Large curated research assets should be tracked with **Git LFS** when intentionally versioned.
- Temporary caches, bulk intermediate files, logs, and regenerable outputs should remain untracked.
- Dataset version, preprocessing assumptions, split logic, and evaluation settings should be documented for every manuscript-facing experiment.
- Public redistribution of third-party or derived data should only be performed when permitted by the original data source and license terms.

---

## Data Notes  

The repository does **not** assume unrestricted redistribution of external datasets.  
Please refer to [`data/README.md`](data/README.md) for:

- expected local directory structure
- raw vs processed data organization
- CESNET-oriented placement guidance
- data-sharing restrictions
- reproducibility recommendations for paper submission

---

## Artifact Policy  

The `artifacts/` directory is intended for curated research outputs such as:

- final plots used in manuscripts
- selected experiment summaries
- reproducible result bundles
- intentionally shared checkpoints or evaluation outputs

Large temporary runs, logs, caches, and repeated intermediate outputs should not be committed unless explicitly required for archival reproducibility.

---

## Figures and Publication Assets  

This repository is structured to support publication-oriented asset generation, including:

- final experimental plots
- benchmark summary tables
- ablation outputs
- reproducibility-ready result bundles

Where applicable, publication assets should be exported in stable formats such as `.png`, `.pdf`, or `.svg` and stored under curated artifact directories.

---

## Citation  

If you use this repository in academic work, please cite the associated paper and, where appropriate, the software repository itself.

Example repository citation:

```bibtex
@misc{vqc_zti_framework_2026,
  author       = {Mubassir Sudipto and contributors},
  title        = {VQC-ZTI Framework: Variational Quantum-Classical Zero-Trust Anomaly Detection and CESNET-Based Security Evaluation},
  year         = {2026},
  howpublished = {\url{https://github.com/<your-username>/VQC-ZTI-Framework}},
  note         = {Code repository}
}
```

---

## License  

This repository is released under the **MIT License**.  
See the [`LICENSE`](LICENSE) file for complete terms.

---

## Acknowledgment  

This research repository supports ongoing work in **quantum-enhanced cybersecurity**, **zero-trust system design**, and **secure next-generation network evaluation**, with an emphasis on reproducible experimental methodology and publication-oriented artifact generation.

---

**Correspondence:** *Mubassir Sudipto*  
**Email:** [msudipto@iastate.edu](mailto:msudipto@iastate.edu)  
**Affiliation:** Department of Electrical and Computer Engineering, Iowa State University
