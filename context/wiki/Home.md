# Agnostic-Inference-Engine Wiki

An engineering and research notebook for building statistical inference tools
from first principles.

The documentation has two layers:

- **English** contains the stable public description of the system, its
  architecture, decisions, and reproducible results.
- **Русский** contains the live investigation: working plans, derivations,
  failed versions, experiments, and the current state of the tools.

## Documentation map

```text
Agnostic-Inference-Engine Wiki
|
|-- Foundations
|   |-- Statistics Map
|   |-- Philosophy of Inference
|   `-- Research and Engineering Methodology
|
|-- Architecture
|   |-- Inference Core
|   |-- Experimental Stands
|   `-- Data and Reproducibility
|
|-- Engineering Decisions
|   `-- Architecture Decision Records (ADR)
|
|-- Research
|   |-- Parametric Inference
|   |-- Model Checking
|   |-- Nonparametric Estimation
|   |-- Regression
|   `-- Systems and PostgreSQL
|
|-- Experiments
|   |-- Synthetic Ground Truth
|   |-- Real Server Latency
|   `-- Regression Stands
|
`-- Russian Lab / Русская лаборатория
    |-- Theory Lab
    |-- Active Plans
    |-- Research Tracks
    `-- Tool Registry
```

## English

- [Foundations](Foundations.md)
- [Research and Engineering Methodology](Research-and-Engineering-Methodology.md)
- [Architecture](Architecture.md)
- [Engineering Decisions](Engineering-Decisions.md)
- [Research](Research.md)
- [Experiments](Experiments.md)

## Русская лаборатория

- [Главная](Home-RU.md)
- [Журнал понимания](Theory-Lab.md)
- [Карта исследований](Research-RU.md)
- [Что уже лежит в ящике](TOOLS-001.md)

## Current state

Python is the Ground Truth and prototyping layer. The current work covers
empirical distributions, bootstrap, maximum likelihood, model checking, KDE,
kernel regression, and linear regression. The intended production core will be
implemented in Rust or C++ after the contracts stabilize.
