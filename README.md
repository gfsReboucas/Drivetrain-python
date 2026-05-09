# Drivetrain Python

Python reimplementation of selected drivetrain modelling, scaling, and validation tools from the MATLAB [Drivetrain](https://github.com/gfsReboucas/Drivetrain) repository.

The goal is not to clone the MATLAB code line by line. The goal is to turn the original research code into a maintainable scientific Python package for gearbox geometry, ISO-style safety calculations, drivetrain scaling, and reduced-order dynamic models.

## Why This Project Exists

Wind turbine drivetrain models often sit between two extremes:

- detailed commercial multibody models that are hard to automate or share;
- simplified analytical models that are easier to run but lose traceability.

This project aims to build the middle layer: transparent, testable Python models that preserve the core physics and can later connect to open-source simulation tools.

## Current Status

This repository currently contains an early Python port started in 2020. It is useful as a reference, but it needs modernization before it should be treated as a reliable package.

Known limitations:

- targets older Python and scientific-library APIs;
- still has some commercial-tool assumptions from the MATLAB code path;
- has an initial package layout and smoke tests, but still needs CI and validation reports;
- needs numerical regression checks against the MATLAB implementation.

## Scope

In scope:

- gear and gear-set geometry;
- shafts, bearings, materials, and carriers;
- NREL 5 MW reference gearbox data;
- ISO 6336 pitting and bending calculations where formulas are available;
- drivetrain scaling studies;
- reduced-order torsional and lateral dynamic formulations;
- validation notebooks and reproducible examples.

Out of scope for the first milestone:

- Simpack automation;
- KISSsoft automation;
- proprietary file formats that require unavailable commercial software.

## Proposed Free Simulation Integrations

Future integration candidates:

- [Project Chrono / PyChrono](https://projectchrono.org/) for open-source multibody dynamics;
- [Exudyn](https://exudyn.readthedocs.io/) for Python-based rigid and flexible multibody systems;
- [OpenModelica](https://openmodelica.org/) for Modelica system models;
- [OpenFAST](https://www.nrel.gov/wind/nwtc/openfast) for wind-turbine aero-servo-elastic context and load cases.

## Development Setup

Recommended Anaconda environment:

```powershell
conda env create -f environment.yml
conda activate drivetrain-python
```

If `conda` is not on PATH, open an Anaconda Prompt or run `conda init powershell` from an Anaconda shell.

## Basic Usage

Use package imports from `drivetrain`:

```python
from drivetrain import NREL_5MW
from drivetrain.Gear import GearSet
from drivetrain.iso6336 import ISO_6336
```

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the migration plan and [docs/github-project.md](docs/github-project.md) for a GitHub issue backlog.

## Portfolio Angle

This is a niche project, so popularity is not the main hiring signal. The strong signal is the combination of:

- mechanical engineering domain knowledge;
- scientific Python;
- numerical validation;
- software architecture around old research code;
- clear project management and documentation.

See [docs/portfolio-positioning.md](docs/portfolio-positioning.md) for how to present it.
