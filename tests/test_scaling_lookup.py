import numpy as np

from drivetrain import GearSet, Shaft


def test_shaft_apply_lambda_uses_exact_scaling_keys():
    shaft = Shaft(100.0, 200.0)

    scaled = shaft.apply_lambda({"d": 2.0, "L": 3.0})

    assert scaled.d == 200.0
    assert scaled.L == 600.0


def test_shaft_apply_lambda_ignores_unrelated_substring_keys():
    shaft = Shaft(100.0, 200.0)

    scaled = shaft.apply_lambda({"diameter": 2.0, "Length": 3.0})

    assert scaled.d == shaft.d
    assert scaled.L == shaft.L


def test_gear_set_apply_lambda_uses_exact_scaling_keys():
    gear_set = GearSet(m_n=4.0, b=20.0, a_w=100.0, shaft=Shaft(50.0, 100.0))

    scaled = gear_set.apply_lambda({"m_n": 2.0, "b": 3.0, "d": 4.0, "L": 5.0})

    assert scaled.m_n == 8.0
    np.testing.assert_allclose(scaled.b, 60.0)
    assert scaled.output_shaft.d == 200.0
    assert scaled.output_shaft.L == 500.0


def test_gear_set_apply_lambda_ignores_unrelated_substring_keys():
    gear_set = GearSet(m_n=4.0, b=20.0, a_w=100.0, shaft=Shaft(50.0, 100.0))

    scaled = gear_set.apply_lambda({"not_m_n": 2.0, "bearing_width": 3.0, "diameter": 4.0})

    assert scaled.m_n == gear_set.m_n
    np.testing.assert_allclose(scaled.b, gear_set.b)
    assert scaled.output_shaft.d == gear_set.output_shaft.d
    assert scaled.output_shaft.L == gear_set.output_shaft.L
