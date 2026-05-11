# Roadmap

This roadmap turns the old Python experiment into a maintainable scientific Python project.

## Milestone 0: Stabilize the Existing Port

Goal: make the current repository importable and runnable in a pinned environment.

- Pin Python 3.11 and scientific dependencies.
- Remove deprecated imports such as `scipy.array` and `numpy.asscalar`.
- Remove or isolate commercial COM imports from import-time code paths.
- Add smoke tests for `NREL_5MW`, `GearSet.example_01_ISO6336`, and the simplest dynamic model.
- Preserve numerical behavior before refactoring.

Deliverable: old examples run without commercial software.

## Milestone 1: Package the Core Domain Model

Goal: create a clean package structure.

- Move top-level modules into a `drivetrain/` package.
- Split geometry, components, ISO calculations, scaling, and dynamics into clear modules.
- Replace broad mutable classes with typed dataclasses where practical.
- Add examples under `examples/`.
- Add tests under `tests/`.

Deliverable: `import drivetrain` works and core models have unit tests.

## Milestone 2: Validate Against MATLAB

Goal: prove the Python results match the MATLAB reference where expected.

- Export MATLAB reference values for NREL 5 MW geometry.
- Compare gear geometry, masses, inertias, mesh stiffness, gear ratios, and ISO 6336 safety factors.
- Add toleranced regression tests.
- Write a validation report with tables and plots.

Deliverable: reproducible validation notebook and automated regression tests.

## Milestone 3: Dynamics

Goal: port and validate reduced-order dynamics.

- Start with torsional 2DOF.
- Port Kahraman 1994 formulation.
- Port Lin and Parker 1999 formulation after core matrices are tested.
- Add base linear-system utilities for state-space assembly, modal truncation, and frequency response.
- Defer drivetrain time-response workflows, Newmark, Wilson, and Bathe integration helpers until deterministic reference tests are added for each solver.
- Defer formulation-specific force vectors, DOF descriptions, and export/disp helpers until the higher-order formulations expose stable matrix layouts.

Deliverable: modal frequencies and mode shapes validated against MATLAB and notebook examples.

## Milestone 4: Scaling Studies

Goal: make the original drivetrain scaling research workflow reproducible in Python.

- Implement scaling factors as explicit data objects.
- Recreate single-stage and full-drivetrain scaling studies.
- Track safety-factor and modal-frequency preservation.
- Add plotting functions for comparison studies.

Deliverable: example notebook showing a complete scaling workflow.

## Milestone 5: Open-Source Simulation Adapters

Goal: evaluate free alternatives to Simpack and KISSsoft integrations.

- Keep adapters optional and outside the core package.
- Prototype PyChrono or Exudyn model export for a simple drivetrain.
- Prototype OpenFAST load import/export workflows.
- Keep ISO calculations native in Python before relying on external tools.

Deliverable: documented experimental adapter, not required for core package use.
