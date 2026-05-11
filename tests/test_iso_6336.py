import numpy as np

from drivetrain.gears import GearSet
from drivetrain.iso6336 import ISO_6336


def test_iso_6336_example_01_pitting_safety_factor():
    gset = GearSet.example_01_ISO6336()
    torque = 9.0e3
    speed = 360.0
    power = torque * speed * (np.pi / 30.0) * 1.0e-3

    calculation = ISO_6336(
        gset,
        K_A=1.0,
        L_h=50.0e3,
        S_Hmin=1.0,
        S_Fmin=1.0,
    )
    safety_factor = calculation.Pitting(
        P=power,
        n_1=speed,
        R_a=1.0,
        nu_40=320.0,
        line=2,
        C_a=70.0,
    )

    np.testing.assert_allclose(
        safety_factor,
        [1.08047178, 1.08591609],
        rtol=1e-8,
    )
    assert np.all(safety_factor > 1.0)
