## Description
Provide a concise overview of the script optimizations, circuit layers adjustments, or zero-trust telemetry preprocessing rules introduced by this PR.

## Related Issue
Fixes # (link your custom Bug Report or Feature Request issue number here)

## Change Type
- [ ] Bug fix (non-breaking change fixing training crashes, feature scaling math, or graph pipeline blocks)
- [ ] New feature (non-breaking extension adding alternative quantum ansatz architectures or metric scoring functions)
- [ ] Refactoring / Documentation alignment

## Architecture & Ablation Impact Checklist
Please verify how your modifications behave relative to the framework's baseline evaluation layout:
- [ ] My modifications preserve the end-to-end training of the **Full Hybrid (FH)** model pipeline.
- [ ] I have verified that this change does not break or cause runtime errors across the relevant ablation variants (**NH**, **SE**, **NE**, **PQC**).
- [ ] The model successfully handles the **12 payload-independent aggregate-flow features** from the CESNET dataset without feature-dimension mismatch.

## Local Pipeline Execution Verification
- [ ] I have executed the main PowerShell tracking wrapper (`.\run_pipeline.ps1`) locally, and it completes training/evaluation cycles without error flags.
- [ ] Any updated visualization benchmarks exported to the `artifacts/` folder render properly in stable vector/raster structures (`.png`, `.svg`, `.pdf`).

## Git and Artifact Containment Integrity
- [ ] I have **NOT** included heavy training weights checkpoints, duplicate processing caches, or raw tabular dataset blocks in this tracking commit.
