---
name: Bug report
about: Report execution crashes, preprocessing failures, or graph execution discrepancies in VQC-ZTI
title: '[BUG] '
labels: 'bug'
assignees: ''

---

**Describe the bug**
A clear and concise description of the training crash, script failure, or artifact generation issue.

**To Reproduce**
Steps to reproduce the behavior:
1. Run pipeline command `.\run_pipeline.ps1` or explicit python command: `python -m src...`
2. Configuration profiles used in `config/`
3. Error outcome or mismatch observed

**Expected behavior**
A clear description of what should happen (e.g., successful epoch completion, balanced class distribution, artifact generation).

**Pipeline Context & Error Logs (please complete if applicable):**
 - Python Script / Entry Point: [e.g., training loop, ablation script, baseline evaluator]
 - Holdout Protocol Used: [e.g., Random Stratified, Entity-Group, Temporal]
 - PyTorch / PennyLane console trackback errors:
   ```text
   Paste tracebacks or console logs here
   ```

**Additional context**
Add any other details about memory allocation limits, missing raw dataset tensors, or script anomalies.
