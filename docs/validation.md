# Validation Notes

## NREL 5 MW Gear Geometry

The fixture in `tests/data/nrel_5mw_matlab_gear_geometry.json` stores gear
geometry reference values for the three NREL 5 MW gearbox stages.

The values are derived from the MATLAB repository sources:

- `@NREL_5MW/NREL_5MW.m`
- `Gear.m`
- `Gear_Set.m`

The fixture covers:

- reference, tip, base, root, and mean diameters
- gear masses
- gear mass moments of inertia
- transverse, overlap, and total contact ratios
- single-tooth and mesh-stiffness factors
- mean mesh stiffness

The regression test uses the tolerances stored in the fixture metadata:

- relative tolerance: `1e-10`
- absolute tolerance: `1e-8`

The Python test does not require MATLAB at runtime. MATLAB remains the source
definition for the fixture, while the committed JSON keeps CI and local testing
reproducible on machines without MATLAB.

## Torsional 2DOF Modal Model

The torsional 2DOF validation test uses a two-inertia analytical system with
one equivalent shaft stiffness. The generalized eigenproblem has:

- one rigid-body mode at `0 Hz`
- one elastic torsional mode

For rotor inertia `J_R`, reflected generator inertia `J_G U^2`, and equivalent
shaft stiffness `k_eq`, the non-zero natural frequency is:

```text
f = sqrt(k_eq * (1 / J_R + 1 / (J_G U^2))) / (2 pi)
```

The rigid-body mode shape is expected to be equal angular motion of both
coordinates. The implementation normalizes each mode by its largest component,
so the rigid-body mode is checked as `[1, 1]`.
