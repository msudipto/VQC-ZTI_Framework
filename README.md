---
license: mit
library_name: pytorch
pipeline_tag: tabular-classification
tags:
  - pytorch
  - pennylane
  - quantum-machine-learning
  - quantum-neural-network
  - variational-quantum-classifier
  - hybrid-quantum-classical
  - anomaly-detection
  - network-anomaly-detection
  - cybersecurity
  - zero-trust
  - tactile-internet
  - network-security
  - encrypted-traffic
  - cesnet
  - arxiv:2608.18572
---

# VQC-ZTI
## Variational Quantum Control for Zero Trust Protection of the Tactile Internet

[![arXiv](https://img.shields.io/badge/arXiv-2608.18572-b31b1b.svg)](https://arxiv.org/abs/2608.18572)
[![Hugging Face Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow.svg)](https://huggingface.co/papers/2608.18572)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2608.18572-blue.svg)](https://doi.org/10.48550/arXiv.2608.18572)
[![Conference](https://img.shields.io/badge/IEEE-GLOBECOM%202026-blue.svg)](https://arxiv.org/abs/2608.18572)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Paper:** [VQC-ZTI: Variational Quantum Control for Zero Trust Protection of the Tactile Internet](https://arxiv.org/abs/2608.18572)  
**Hugging Face Paper:** [huggingface.co/papers/2608.18572](https://huggingface.co/papers/2608.18572)  
**Code:** [github.com/msudipto/VQC-ZTI_Framework](https://github.com/msudipto/VQC-ZTI_Framework)  
**DOI:** [10.48550/arXiv.2608.18572](https://doi.org/10.48550/arXiv.2608.18572)  
**Venue:** Accepted at **IEEE Global Communications Conference (GLOBECOM 2026)**  
**Primary arXiv Category:** Cryptography and Security (`cs.CR`)  
**Cross-lists:** Machine Learning (`cs.LG`), Networking and Internet Architecture (`cs.NI`)

---

## Model Description

**VQC-ZTI** is a hybrid quantum-classical anomaly-evidence model developed for a split-plane zero-trust architecture targeting Tactile Internet services.

The framework separates:

1. **Off-path evidence generation**, where encrypted-flow telemetry is processed by a hybrid Variational Quantum Classifier / Quantum Neural Network; and
2. **On-path deterministic enforcement**, where Policy Enforcement Points (PEPs) apply cached grant, restrict, step-up, or deny actions.

The design prevents probabilistic VQC inference from being placed directly in the latency-critical Tactile Internet control path.

The released implementation uses **PyTorch** and **PennyLane** and evaluates the hybrid model on **CESNET-derived aggregated encrypted-flow traffic records**.

### Model Sources

- **Paper:** https://arxiv.org/abs/2608.18572
- **Hugging Face Paper Page:** https://huggingface.co/papers/2608.18572
- **Source Code:** https://github.com/msudipto/VQC-ZTI_Framework
- **DOI:** https://doi.org/10.48550/arXiv.2608.18572

---

## Authors

### Mubassir Serneabat Sudipto
Electrical and Computer Engineering  
Iowa State University  
Ames, Iowa, USA  
Email: [msudipto@iastate.edu](mailto:msudipto@iastate.edu)

### Shakil Ahmed
Computer Science, College of Computing  
Grand Valley State University  
Allendale, Michigan, USA  
Email: [ahmeshak@gvsu.edu](mailto:ahmeshak@gvsu.edu)

### Ashfaq Khokhar
Carl R. Ice College of Engineering  
Kansas State University  
Manhattan, Kansas, USA  
Email: [akhokhar@k-state.edu](mailto:akhokhar@k-state.edu)

---

## Model Architecture

The full-hybrid VQC-ZTI anomaly model contains the following stages:

1. **Feature preprocessing**
   - Payload-independent encrypted-flow statistics
   - Robust feature preparation
   - 12 input features

2. **Classical embedding**
   - A trainable classical embedder maps the processed feature vector into a 12-dimensional representation.

3. **Quantum feature encoding**
   - Number of qubits: **12**
   - Initial state: all-zero state
   - Feature encoding: single-qubit Pauli-Y rotations (`RY`)

4. **Variational Quantum Circuit**
   - Number of variational layers: **2**
   - Trainable single-qubit rotations
   - Nearest-neighbor CNOT entanglement

5. **Quantum measurement**
   - Pauli-Z expectation values are measured from two output qubits.

6. **Classical classification head**
   - Quantum measurements are mapped into two class logits.
   - The class-1 probability is used as the continuous anomaly score.

The complete architecture is trained end-to-end through the PyTorch-PennyLane computational graph.

---

## Input

The model operates on structured, payload-independent network-traffic features derived from CESNET aggregated traffic.

The evaluated implementation uses:

- **12 numerical features**
- packet and byte statistics
- destination diversity
- traffic ratios
- average flow duration
- average time-to-live
- related aggregated flow characteristics

The model is therefore a **tabular binary-classification model**, not a natural-language, image, or generative model.

---

## Output

The model produces a continuous anomaly score:

```text
0 <= anomaly_score <= 1
```

The binary experimental classes are:

```text
0 = normative record
1 = suspicious or high-anomaly record
```

The score represents evidence with respect to the statistical anomaly-labeling procedure used in the study.

**It must not be interpreted as a calibrated probability that a network flow is malicious.**

---

## Dataset and Experimental Data

The evaluation reported in the associated paper uses **4,875 CESNET-derived aggregated traffic records** with **12 payload-independent features**.

Experimental labels are generated through a controlled statistical-anomaly procedure based on robust-scaled feature norms and empirical quantiles.

The labeling scheme separates:

- normative records,
- suspicious records, and
- high-anomaly records.

For binary evaluation, suspicious and high-anomaly records are combined into the anomaly class.

### Important Data Qualification

These labels are **statistical pseudo-labels**, not verified attack annotations.

Consequently:

- reported anomaly-detection performance measures agreement with the constructed statistical benchmark;
- false-positive rate is measured relative to the quantile-derived normative class; and
- the reported results must not be interpreted as verified intrusion-detection performance against confirmed cyberattacks.

The full data preparation and reproducibility workflow is documented in the GitHub repository:

https://github.com/msudipto/VQC-ZTI_Framework

---

## Evaluation Protocols

Three evaluation protocols are used.

### Random Stratified Holdout

- Training records: **3,900**
- Evaluation records: **975**
- Binary-label distribution preserved

### Entity-Group Holdout

- Training records: approximately **3,751-3,971**
- Evaluation records: approximately **904-1,124**
- Training and evaluation entity groups are disjoint

### Temporal Holdout

- Training records: **3,887 earlier records**
- Evaluation records: **988 later records**

These protocols test performance under conventional random splitting as well as entity and chronological distribution shifts.

---

## Training Procedure

The reported full-hybrid QNN configuration uses:

| Parameter | Value |
|---|---:|
| Input features | 12 |
| Qubits | 12 |
| Variational layers | 2 |
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 2 x 10^-3 |
| Random seeds | 42-46 |
| Training runs | 5 |
| Quantum execution | Analytic expectation values |
| Quantum framework | PennyLane |
| ML framework | PyTorch |

Training uses a **class-weighted negative log-likelihood objective**.

Class weights are determined from the training partition and normalized to unit mean. The classical embedder, variational quantum circuit, and classical classification head are jointly optimized.

### Quantum Execution Setting

The reported experiments use **noiseless analytic expectation values**.

The reported results therefore do not include:

- finite-shot uncertainty,
- quantum-hardware noise,
- physical-device connectivity constraints,
- transpilation overhead,
- hardware queueing delay, or
- other physical quantum-computing effects.

---

## Evaluation Results

The full-hybrid QNN is compared against:

- ExtraTrees
- Random Forest
- Logistic Regression
- hybrid-model ablations

### Full-Hybrid QNN Results

| Evaluation Split | ROC-AUC | Accuracy | FPR |
|---|---:|---:|---:|
| Entity-Group | **0.9974 +/- 0.0009** | **0.9762 +/- 0.0028** | **0.0241 +/- 0.0042** |
| Random | **0.9981 +/- 0.0004** | **0.9756 +/- 0.0042** | **0.0270 +/- 0.0050** |
| Temporal | **0.9941 +/- 0.0018** | **0.9747 +/- 0.0044** | **0.0248 +/- 0.0052** |

Across the three protocols, the full-hybrid QNN achieves the highest reported mean AUC and accuracy and the lowest mean FPR among the evaluated models.

### Comparison with ExtraTrees

Relative to ExtraTrees, mean FPR is reduced by:

| Evaluation Split | FPR Reduction |
|---|---:|
| Random | **44.6%** |
| Entity-Group | **49.6%** |
| Temporal | **67.9%** |

These comparisons do **not** demonstrate quantum computational advantage because the classical baselines are not parameter- or compute-matched to the complete hybrid neural architecture.

---

## Ablation Study

The associated study evaluates five model configurations:

| Variant | Description |
|---|---|
| **Full Hybrid (FH)** | Complete classical embedder + VQC + classification head |
| **No Head (NH)** | Classification head removed |
| **Shallow Embedder (SE)** | Reduced classical embedding architecture |
| **No Embedder (NE)** | Processed features encoded directly |
| **PQC-only (PQC)** | Restricted parameterized quantum circuit without the complete hybrid structure |

The full-hybrid configuration performs best across all reported evaluation protocols.

The ablation results indicate that the reported performance depends on the **complete hybrid quantum-classical pipeline**, rather than on the parameterized quantum circuit alone.

---

## Zero-Trust Integration

The anomaly model is designed as an **evidence source**, not as a direct access-control mechanism.

VQC-ZTI separates two operational planes:

### Evidence Plane

The asynchronous evidence plane performs:

```text
Encrypted-Flow Telemetry
        |
Feature Preparation
        |
Classical Embedding
        |
Quantum Encoding
        |
VQC / QNN Inference
        |
Classical Post-Processing
        |
Risk Fusion
        |
Policy Computation
```

### Enforcement Plane

The latency-critical enforcement plane uses:

```text
Request
   |
Policy Enforcement Point (PEP)
   |
Cached Deterministic Policy
   |
Grant / Restrict / Step-Up / Deny
   |
Protected Tactile Internet Service
```

VQC execution is therefore **not an intermediate processing hop for the current Tactile Internet transaction**.

Delayed anomaly scoring can postpone a future policy update, but does not directly add quantum-processing delay to the current enforcement decision.

---

## Intended Uses

VQC-ZTI is intended for:

- research on hybrid quantum-classical machine learning;
- network anomaly-detection experimentation;
- zero-trust architecture research;
- Tactile Internet security research;
- quantum-enhanced cybersecurity experimentation;
- comparative classical/quantum model evaluation;
- reproducibility studies;
- controlled anomaly-scoring experiments;
- hybrid-model ablation studies; and
- research on off-path security-evidence generation.

---

## Out-of-Scope Uses

The model should **not** currently be treated as:

- a production intrusion-detection system;
- a verified malware or attack detector;
- a calibrated probability-of-compromise estimator;
- an autonomous access-control authority;
- proof of quantum advantage;
- evidence of performance on physical quantum hardware;
- evidence of end-to-end Tactile Internet latency compliance; or
- a replacement for deployment-grade network-security validation.

Security-critical decisions should not be based solely on this experimental model.

---

## Limitations

The evaluation has several important validity boundaries.

1. **Statistical labels**
   - Labels are quantile-derived statistical anomalies rather than verified attack annotations.

2. **Dataset scope**
   - CESNET aggregated traffic does not represent a hardware-in-the-loop Tactile Internet deployment.

3. **Quantum simulation**
   - Experiments use noiseless analytic quantum simulation.

4. **No finite-shot evaluation**
   - Shot noise and sampling uncertainty are excluded.

5. **No physical quantum hardware**
   - Hardware noise, topology, transpilation, execution latency, and queueing are not evaluated.

6. **No demonstrated quantum advantage**
   - Classical baselines are not parameter- or compute-matched to the complete hybrid architecture.

7. **Zero-trust policy validation**
   - Risk-fusion functions and policy thresholds have not yet been calibrated against operational access-control outcomes.

8. **Latency validation**
   - Prototype component measurements do not establish end-to-end Tactile Internet latency compliance.

Future work includes verified attack labels, capacity-matched neural baselines, finite-shot and noisy quantum execution, physical quantum devices, controlled attack experiments, and hardware-in-the-loop PDP/PEP evaluation.

---

## Reproducibility

The complete experimental framework is available at:

**GitHub:**  
https://github.com/msudipto/VQC-ZTI_Framework

The repository provides:

- preprocessing code;
- experiment configuration;
- PyTorch-PennyLane training workflow;
- classical baselines;
- hybrid-model ablations;
- evaluation scripts;
- result generation;
- manuscript figures;
- experiment artifacts; and
- reproducibility documentation.

### Repository Installation

```bash
git clone https://github.com/msudipto/VQC-ZTI_Framework.git
cd VQC-ZTI_Framework

python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
.\run_pipeline.ps1
```

On compatible Unix-like environments, activate the environment with:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

See the GitHub repository for the current execution and data-placement instructions.

---

## Downloading from Hugging Face

After replacing `YOUR_HF_USERNAME` with the owner of this model repository, the model repository can be cloned using:

```bash
git clone https://huggingface.co/YOUR_HF_USERNAME/VQC-ZTI
```

or downloaded with the Hugging Face CLI:

```bash
hf download YOUR_HF_USERNAME/VQC-ZTI
```

Model loading depends on the checkpoint format included with the release. Refer to the accompanying repository files and the official VQC-ZTI GitHub implementation for the exact reconstruction and inference procedure.

---

## Paper

The model and experimental framework are described in:

**Mubassir Serneabat Sudipto, Shakil Ahmed, and Ashfaq Khokhar.**  
**“VQC-ZTI: Variational Quantum Control for Zero Trust Protection of the Tactile Internet.”**  
Accepted at **IEEE Global Communications Conference (GLOBECOM 2026)**.  
arXiv:2608.18572, 2026.

- **arXiv:** https://arxiv.org/abs/2608.18572
- **Hugging Face Papers:** https://huggingface.co/papers/2608.18572
- **DOI:** https://doi.org/10.48550/arXiv.2608.18572
- **Code:** https://github.com/msudipto/VQC-ZTI_Framework

---

## Citation

If you use VQC-ZTI, its implementation, experimental methodology, model checkpoints, or reported results in academic work, please cite the associated paper.

### Paper Citation

```bibtex
@misc{sudipto2026vqczti,
  title         = {{VQC-ZTI}: Variational Quantum Control for Zero Trust Protection of the Tactile Internet},
  author        = {Sudipto, Mubassir Serneabat and Ahmed, Shakil and Khokhar, Ashfaq},
  year          = {2026},
  eprint        = {2608.18572},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  doi           = {10.48550/arXiv.2608.18572},
  url           = {https://arxiv.org/abs/2608.18572},
  note          = {Accepted at IEEE Global Communications Conference (GLOBECOM 2026)}
}
```

### Software Repository Citation

If you specifically use or extend the accompanying software implementation, you may additionally cite:

```bibtex
@misc{vqc_zti_framework_2026,
  author       = {Sudipto, Mubassir Serneabat and Ahmed, Shakil and Khokhar, Ashfaq},
  title        = {{VQC-ZTI Framework}: Variational Quantum-Classical Zero-Trust Anomaly Detection and CESNET-Based Security Evaluation},
  year         = {2026},
  howpublished = {\url{https://github.com/msudipto/VQC-ZTI_Framework}},
  note         = {Code repository},
  url          = {https://github.com/msudipto/VQC-ZTI_Framework}
}
```

---

## License

The VQC-ZTI software implementation is released under the **MIT License**.

See the accompanying `LICENSE` file and the official GitHub repository for complete terms:

https://github.com/msudipto/VQC-ZTI_Framework

---

## Acknowledgment

This research supports ongoing work in:

- quantum-enhanced cybersecurity;
- hybrid quantum-classical machine learning;
- zero-trust system design;
- network anomaly detection;
- Tactile Internet security; and
- secure next-generation network evaluation.

The project emphasizes reproducible experimental methodology and careful separation between simulated anomaly-evidence performance and claims of operational security effectiveness or quantum advantage.

---

## Contact

**Mubassir Serneabat Sudipto**  
Iowa State University  
[msudipto@iastate.edu](mailto:msudipto@iastate.edu)

**Shakil Ahmed**  
Grand Valley State University  
[ahmeshak@gvsu.edu](mailto:ahmeshak@gvsu.edu)

**Ashfaq Khokhar**  
Kansas State University  
[akhokhar@k-state.edu](mailto:akhokhar@k-state.edu)

---

**Paper:** https://arxiv.org/abs/2608.18572  
**Hugging Face Paper:** https://huggingface.co/papers/2608.18572  
**GitHub:** https://github.com/msudipto/VQC-ZTI_Framework
