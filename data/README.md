# Data Directory

This directory documents how datasets should be organized for local experimentation and reproducible evaluation.

## Purpose

The `data/` directory is intended for:

- dataset placement instructions
- lightweight public samples, if any
- local raw and processed data organization
- reproducibility notes for manuscript preparation

This repository should **not** be used to redistribute third-party datasets unless redistribution is explicitly permitted by the original data provider.

## Recommended Layout

```text
data/
├── README.md
├── samples/                # Optional lightweight public sample files
├── raw/                    # Original downloaded or collected data (ignored by Git)
├── interim/                # Temporary intermediate files (ignored by Git)
└── processed/              # Generated modeling-ready datasets (ignored by Git)
```

## Expected Workflow

1. Place original source data under `data/raw/`
2. Run preprocessing scripts from the repository
3. Store intermediate outputs under `data/interim/`
4. Write model-ready outputs to `data/processed/`
5. Track only lightweight examples or explicitly shareable files

## CESNET-Oriented Organization

For CESNET-derived experiments, a typical raw-data layout may look like:

```text
data/raw/cesnet/
├── ip_addresses_full/
│   └── agg_10_minutes/
├── institutions/
│   └── agg_10_minutes/
├── institution_subnets/
│   └── agg_10_minutes/
└── times/
    └── times_10_minutes.csv
```

Adjust the exact layout as required by your preprocessing configuration files in `config/`.

## Processed Outputs

A typical processed output may be generated under a path such as:

```text
data/processed/cesnet_vqczti.npz
```

or another configuration-driven filename used by the training and evaluation pipeline.

Processed files are usually ignored by Git because they can be large, regenerable, or derived from third-party data sources with separate usage restrictions.

## Data Sharing Policy

Please follow these rules when using this repository:

- do not commit private, licensed, or restricted raw datasets
- do not assume derived data can be redistributed without checking source terms
- do not publish large processed datasets unless redistribution is clearly allowed
- use Git LFS only for intentionally shared, permitted large files
- keep public sample files small and clearly documented

## Reproducibility Guidance

For paper submission and archival purposes, document:

- dataset name and version
- acquisition date
- preprocessing configuration used
- feature selection and transformation assumptions
- train/evaluation split protocol
- random seeds
- any filtering, aggregation, or label-generation logic applied

## Notes for Collaborators

If you are reproducing the experiments on a new machine:

- create the `raw/`, `interim/`, and `processed/` folders locally if missing
- place the source dataset under `data/raw/`
- verify that config paths match your local directory structure
- run the preprocessing step before training or evaluation

## Dataset Attribution

If your experiments use CESNET-derived data or any other third-party source, cite the original dataset or publication in your manuscript and respect the source license and terms of use.
