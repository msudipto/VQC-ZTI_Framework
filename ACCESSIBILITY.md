# Accessibility Statement

We want the VQC-ZTI Framework to be accessible to everyone, including researchers, students, and practitioners with disabilities. This document outlines our current accessibility focus and how you can report barriers.

## Priorities

Our priority is ensuring that our research codebases, dataset descriptions, and automation scripts are readable and usable by all developers. We focus on:
* **Text Readability**: Structuring project logs, feature definitions, and setup files to clear, screen-reader friendly standards.
* **Vector Graphics**: Exporting evaluation plots and charts inside the `artifacts/` folder using stable vector file structures like SVG and PDF to support high-contrast scaling.
* **Keyboard-Only Operation**: Ensuring that our main pipeline commands can be entirely run and tracked via standard keyboard terminal interactions.

## Contributor expectations

When contributing new variational layers, feature scaling pipelines, or script revisions:
* Use high-contrast color palettes for code execution figures and charts to assist color-blind reviewers.
* Document any changes to configuration keys or features in clear, semantic plain text within documentation files or script headers.

## Reporting accessibility issues

If you encounter an accessibility barrier while running our scripts, reading the framework logs, or inspecting the publication charts, please report it privately via email:

* **Mubassir Serneabat Sudipto** — msudipto@iastate.edu

Please describe the barrier you encountered, the relevant script or file path, and any assistive technology you were using.

### How we respond

We will acknowledge your report within 5 business days and work to implement a clear text workaround or graphical update in the primary development branch.

## Supported environments

This framework is tested on standard terminal environments (Windows PowerShell and Unix Bash). We recommend using high-contrast terminal themes and vector-capable document viewers to inspect output charts under the `artifacts/` folder.
