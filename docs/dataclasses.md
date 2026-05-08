# Dataclass Migration Notes

The core package uses Python `dataclasses` for small engineering containers
whose constructor inputs are clear and mostly passive:

- `Material`
- `Rack`
- `Bearing`
- `Shaft`
- `Carrier`

These classes still compute derived engineering quantities in `__post_init__`
so the public attributes used by the existing calculations remain available.
For example, `Shaft` keeps the existing `Shaft(dd, LL)` call style while
deriving area, volume, inertia, and mass after initialization.

Calculation-heavy classes such as `Gear`, `GearSet`, `Drivetrain`, dynamic
formulations, and `ISO_6336` are intentionally left as regular classes for
now. They currently combine input handling, derived geometry, numerical
methods, and validation assumptions. Converting them safely should happen
after their modules are split and their inputs are better separated from
derived outputs.

Pydantic is deferred because the current package is a numerical modeling core,
not a file/API boundary. It may be useful later for validating JSON, YAML, CSV,
or web/API inputs before those inputs are converted into the internal
dataclass-based model objects.
