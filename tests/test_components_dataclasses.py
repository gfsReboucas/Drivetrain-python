from dataclasses import is_dataclass

import numpy as np

from drivetrain.Gear import Carrier
from drivetrain.components import Bearing, Material, Rack, Shaft
from drivetrain.components.bearings import Bearing as BearingModuleImport
from drivetrain.components.materials import Material as MaterialModuleImport
from drivetrain.components.racks import Rack as RackModuleImport
from drivetrain.components.shafts import Shaft as ShaftModuleImport


def test_core_component_types_are_dataclasses():
    assert is_dataclass(Material)
    assert is_dataclass(Rack)
    assert is_dataclass(Bearing)
    assert is_dataclass(Shaft)
    assert is_dataclass(Carrier)


def test_component_package_reexports_focused_modules():
    assert MaterialModuleImport is Material
    assert RackModuleImport is Rack
    assert BearingModuleImport is Bearing
    assert ShaftModuleImport is Shaft


def test_material_derives_shear_modulus():
    steel = Material()

    assert steel.E == 206.0e9
    np.testing.assert_allclose(steel.G, (steel.E / 2.0) / (1.0 + steel.nu))


def test_rack_preserves_module_rounding_behavior():
    rack = Rack(type="D", m=6.5, alpha_P=20.0)

    assert rack.type == "D"
    assert rack.m == 6.0
    np.testing.assert_allclose(rack.p, np.pi * rack.m)


def test_bearing_preserves_matrix_behavior():
    bearing = Bearing(
        stiffness=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        damping=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        name="BRG",
        type="test",
        OD=10.0,
        ID=5.0,
        B=2.0,
    )

    np.testing.assert_allclose(bearing.stiffness_matrix(), np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    np.testing.assert_allclose(bearing.damping_matrix(), np.diag([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    assert bearing.name == "BRG"
    assert bearing.type == "test"


def test_shaft_preserves_positional_constructor_and_derived_values():
    shaft = Shaft(700.0, 2.0e3)

    assert shaft.d == 700.0
    assert shaft.L == 2.0e3
    np.testing.assert_allclose(shaft.A, (np.pi / 4.0) * (0.7**2))
    assert shaft.mass > 0.0


def test_carrier_derives_geometry_and_inertia():
    carrier = Carrier(863.0, 491.0)

    assert carrier.a_w == 863.0
    assert carrier.b_g == 491.0
    np.testing.assert_allclose(carrier.d_a, 2.6 * carrier.a_w)
    assert carrier.mass > 0.0
