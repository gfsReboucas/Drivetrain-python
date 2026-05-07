# GitHub Project Backlog

Use this as the initial GitHub Project board. Suggested columns:

- Backlog
- Ready
- In Progress
- Review
- Done

## Labels

- `type:cleanup`
- `type:feature`
- `type:test`
- `type:docs`
- `type:validation`
- `area:geometry`
- `area:iso-6336`
- `area:dynamics`
- `area:scaling`
- `area:packaging`
- `good first issue`

## Initial Issues

### 1. Make the old Python port importable

Labels: `type:cleanup`, `area:packaging`

Replace deprecated imports and isolate optional COM dependencies so importing the package does not require Simpack, KISSsoft, or Windows COM objects.

Acceptance criteria:

- `python -c "from Drivetrain import NREL_5MW"` works in the conda environment.
- No commercial software is required for import.
- Existing examples fail only on known numerical issues, not missing imports.

### 2. Add a reproducible conda environment

Labels: `type:cleanup`, `area:packaging`, `good first issue`

Validate `environment.yml` on Windows with Anaconda.

Acceptance criteria:

- Environment creates successfully.
- `python --version` reports Python 3.11.
- `numpy`, `scipy`, `matplotlib`, and `pytest` import successfully.

### 3. Create a package layout

Labels: `type:cleanup`, `area:packaging`

Move the current modules into a proper `drivetrain/` package without changing behavior.

Acceptance criteria:

- `import drivetrain` works.
- Existing examples are updated.
- Old module names are either migrated or temporarily shimmed.

### 4. Add smoke tests for the NREL 5 MW model

Labels: `type:test`, `area:geometry`

Create tests that instantiate the NREL 5 MW reference drivetrain and check basic dimensions, ratios, and array shapes.

Acceptance criteria:

- `pytest` runs at least one NREL smoke test.
- Tests avoid commercial integrations.

### 5. Validate gear geometry against MATLAB

Labels: `type:validation`, `area:geometry`

Export reference values from MATLAB and compare Python values for gear diameters, masses, inertias, contact ratios, and mesh stiffness.

Acceptance criteria:

- Reference data is stored in a text format such as CSV or JSON.
- Tolerances are documented.
- Regression tests compare Python output to the reference data.

### 6. Port ISO 6336 pitting tests

Labels: `type:test`, `area:iso-6336`, `type:validation`

Build tests around the ISO/TR 6336-30 example already referenced in the MATLAB and Python code.

Acceptance criteria:

- One parallel gear-set benchmark is automated.
- Numerical tolerance is explicit.
- Test output includes the expected safety-factor range.

### 7. Separate commercial integrations from core logic

Labels: `type:cleanup`, `area:packaging`

Move KISSsoft and Simpack related functionality behind optional adapter modules.

Acceptance criteria:

- Core package imports on machines without Simpack or KISSsoft.
- Adapter imports fail with clear messages when optional dependencies are unavailable.

### 8. Validate torsional 2DOF modal model

Labels: `type:test`, `area:dynamics`, `type:validation`

Use the analytical test from the old dynamic-formulation branch as the first dynamics regression test.

Acceptance criteria:

- Natural frequencies match the analytical solution.
- Rigid-body mode handling is documented.

### 9. Port Kahraman 1994 stage model

Labels: `type:feature`, `area:dynamics`

Port the `general_Kahraman_94` branch idea into tested production code.

Acceptance criteria:

- Fixed-ring planetary stage analytical test passes.
- Matrix dimensions and coordinate definitions are documented.

### 10. Write the first validation notebook

Labels: `type:docs`, `type:validation`

Create a notebook that explains the NREL 5 MW model, prints key drivetrain properties, and compares at least one result against MATLAB reference data.

Acceptance criteria:

- Notebook runs from a clean conda environment.
- Output is understandable without reading the MATLAB source.

### 11. Port drivetrain comparison and export utilities

Labels: `type:feature`, `area:packaging`, `area:geometry`

Port MATLAB display/comparison/export behavior such as drivetrain and stage comparison tables, `export2struct`, `save_as_struct`, and matrix export, using Python-native outputs such as dictionaries, pandas DataFrames, JSON, or CSV.

Acceptance criteria:

- A Python user can compare reference and scaled drivetrains in tabular form.
- Exported data is serializable without custom class instances.
- Tests cover at least one reference-vs-scaled comparison.

### 12. Port missing shaft calculations

Labels: `type:feature`, `area:geometry`, `type:validation`

Python has basic shaft geometry and matrices, but MATLAB `Shaft.m` also includes critical speed, scaled shaft sizing, fatigue/yield safety factors, damping matrix variants, and broader validation tests.

Acceptance criteria:

- `critical_speed`, `scaled_version`, `safety_factors`, and damping-matrix behavior are implemented or explicitly scoped out.
- Unit tests cover torsional stiffness, mass/inertia, and one shaft safety-factor case.
- Results are compared against MATLAB reference values where available.

### 13. Port bearing fatigue and load-distribution analysis

Labels: `type:feature`, `type:validation`

Python has bearing stiffness/damping association, but MATLAB `Bearing.m` includes dynamic equivalent load, fatigue damage, load-duration distribution displays, histograms, Weibull fitting, and data-analysis helpers.

Acceptance criteria:

- Dynamic equivalent load and damage calculation are implemented independently of Simpack.
- Plotting/data-analysis helpers operate on plain Python data structures or pandas objects.
- At least one fatigue/damage result is regression-tested.

### 14. Port gear and gear-set helper methods

Labels: `type:feature`, `area:geometry`

MATLAB `Gear.m` and `Gear_Set.m` include helper methods not yet present or incomplete in Python: work pitch diameter, mass recalculation, reference-circle plotting data, ISO rounding helper, gear-dimension conversion, gear-set comparison, `scale_by`, `scaled_version`, speed mapping, mesh-stiffness profile, total mass/inertia, center-distance solving, and kinematic tree visualization.

Acceptance criteria:

- Pure geometry/scaling helpers are ported before visualization helpers.
- Mesh stiffness and center-distance calculations have tests.
- Plotting helpers return data or axes without requiring interactive UI.

### 15. Complete native ISO 6336 implementation

Labels: `type:feature`, `area:iso-6336`, `type:validation`

The Python `ISO_6336.py` currently covers a subset of pitting calculations. MATLAB has richer native `MATISO_6336.m` behavior, including bending factors, life factors, spectra helpers, dynamic factor variants, load-spectrum examples, and many derived ISO factors.

Acceptance criteria:

- Native Python ISO pitting coverage is reconciled with MATLAB `MATISO_6336.m`.
- Bending safety factor support is implemented or separated into a documented follow-up.
- ISO factors have focused tests against MATLAB/reference examples.

### 16. Port load-duration and stress-spectrum utilities

Labels: `type:feature`, `area:iso-6336`, `type:validation`

MATLAB `ISO_6336.m` includes load-duration distribution (`LDD`), pitting stress bins, torque/speed spectra helpers, S-N curve helpers, Weibull summaries, and plotting routines. These are useful without Simpack if driven by CSV or pandas time series.

Acceptance criteria:

- LDD and pitting stress bin functions accept arrays or pandas Series.
- Tests cover a small synthetic load/speed time series.
- Plotting is optional and does not block headless test runs.

### 17. Port dynamic response utilities

Labels: `type:feature`, `area:dynamics`, `type:validation`

MATLAB `Dynamic_Formulation.m` includes damped modal analysis, state matrices, modal truncation, Newmark/Wilson/Bathe integration, FRF, frequency response, PI-controlled time response, and load-vector helpers. Python currently has only partial modal logic.

Acceptance criteria:

- Newmark integration is ported first with the MATLAB textbook examples as tests.
- FRF and frequency response are added after matrix assembly is validated.
- Damped modal analysis and state matrices have shape and regression tests.

### 18. Complete Kahraman 1994 dynamics port

Labels: `type:feature`, `area:dynamics`, `type:validation`

Python has an early `Kahraman_94` class, but MATLAB has a fuller implementation with stage inertia, faulty inertia, damping, stiffness, sun/planet/ring stiffness components, fault support, DOF descriptions, and validation tests.

Acceptance criteria:

- Stage-level matrix methods are implemented with documented DOF ordering.
- Fixed-ring planetary test from the old Python branch is automated.
- Fault-related methods are either ported or split into a follow-up issue.

### 19. Complete Lin and Parker 1999 dynamics port

Labels: `type:feature`, `area:dynamics`, `type:validation`

Python `Lin_Parker_99` is visibly incomplete. MATLAB `Lin_Parker_99.m` includes inertia, damping, stiffness, centripetal vector, external load vector, coordinate changes, stage matrices, tests, and derived `KK` behavior.

Acceptance criteria:

- Python `Lin_Parker_99` can instantiate for NREL 5 MW without fallback to the 2DOF model.
- Stage matrix dimensions and coordinate transforms are tested.
- Modal frequencies are compared against MATLAB reference output.

### 20. Port Eritenel and Parker 2009 dynamics formulation

Labels: `type:feature`, `area:dynamics`

MATLAB includes `Eritenel_Parker_09.m`, but Python has no equivalent. This should be handled after the Kahraman and Lin/Parker ports are stable.

Acceptance criteria:

- Scope and references are documented before implementation.
- Stage inertia and stiffness matrices are ported with tests.
- Numerical comparisons against MATLAB are added.

### 21. Port carrier plotting/export helpers

Labels: `type:feature`, `area:geometry`

Python has a basic `Carrier`, while MATLAB `Carrier.m` also includes export, 2D/3D plotting, rectangle rendering, and reference-circle geometry.

Acceptance criteria:

- Geometry helpers return arrays suitable for plotting.
- Export returns serializable dictionaries.
- Tests cover mass/inertia and reference-circle output shapes.

### 22. Port additional drivetrain reference models

Labels: `type:feature`, `area:geometry`, `type:validation`

MATLAB includes reference drivetrain class folders such as `@DTU_10MW`, `@GRC_750kW`, `@NREL_5MW`, and `@WTDB_5MW`. Python currently exposes only `NREL_5MW`.

Acceptance criteria:

- Add Python reference classes or data builders for non-commercial model data.
- Each model has smoke tests for stage layout, ratios, and basic geometry.
- Simpack model files remain out of scope for core tests.

### 23. Add optional adapters for commercial-tool workflows

Labels: `type:feature`, `area:packaging`

MATLAB includes `SimpackCOM`, `KISSsoftCOM`, `KSISO_6336`, and several Simpack/KISSsoft file workflows. These should not be core dependencies, but the Python package should make room for optional adapters or free alternatives later.

Acceptance criteria:

- Commercial integrations are isolated behind optional adapter modules.
- Missing commercial software produces clear runtime messages, not import failures.
- Free alternatives such as PyChrono, Exudyn, OpenModelica, or OpenFAST are evaluated in separate design notes before implementation.

## Deferred Issues

### D1. Port drivetrain scaling workflow from MATLAB

Labels: `type:feature`, `area:scaling`

Port the non-commercial scaling methods from MATLAB `Drivetrain.m` and `scaling_factor.m`, including `scale_all`, `scale_aspect`, `scaled_version`, `scaled_sweep`, `scaled_safety_factors`, and the `scaling_factor` helper behavior.

Reason for deferral:

Scaling should come after the geometry, ISO calculations, and dynamic models have stronger validation coverage. Otherwise it will be hard to tell whether scaling failures come from the scaling logic or from unvalidated underlying calculations.

Acceptance criteria:

- Scaling factors are represented by an explicit Python object or typed mapping.
- Full-drivetrain and stage-level scaling can be run without Simpack or KISSsoft.
- At least one scaling result is regression-tested against exported MATLAB reference data.
