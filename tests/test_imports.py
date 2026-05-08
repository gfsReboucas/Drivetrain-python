def test_package_imports():
    import drivetrain

    assert drivetrain.Material().E > 0.0
    assert drivetrain.GearSet is not None
    assert drivetrain.ISO_6336 is not None
    assert drivetrain.torsional_2DOF is not None
    assert drivetrain.NREL_5MW is not None


def test_legacy_module_imports_remain_available():
    import components
    import Drivetrain
    import dynamic_formulation
    import Gear
    import ISO_6336

    assert components.Material().E > 0.0
    assert Gear.GearSet is not None
    assert ISO_6336.ISO_6336 is not None
    assert dynamic_formulation.torsional_2DOF is not None
    assert Drivetrain.NREL_5MW is not None