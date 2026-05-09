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
