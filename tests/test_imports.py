def test_core_modules_import():
    import components
    import Gear
    import ISO_6336
    import dynamic_formulation
    import Drivetrain

    assert components.Material().E > 0.0
    assert Gear.GearSet is not None
    assert ISO_6336.ISO_6336 is not None
    assert dynamic_formulation.torsional_2DOF is not None
    assert Drivetrain.NREL_5MW is not None
