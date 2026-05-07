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
