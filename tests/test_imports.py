def test_package_imports():
    import drivetrain

    assert drivetrain.Material().E > 0.0
    assert drivetrain.GearSet is not None
    assert drivetrain.torsional_2DOF is not None
    assert drivetrain.NREL_5MW is not None


def test_iso_6336_package_import():
    from drivetrain.iso6336 import ISO_6336 as PackageImport

    assert PackageImport is not None
