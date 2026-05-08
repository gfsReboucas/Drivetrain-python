def test_package_imports():
    import drivetrain

    assert drivetrain.Material().E > 0.0
    assert drivetrain.GearSet is not None
    assert drivetrain.ISO_6336 is not None
    assert drivetrain.torsional_2DOF is not None
    assert drivetrain.NREL_5MW is not None
