import numpy as np
import pytest

from drivetrain.gears import GearSet


def _parallel_stage():
    return GearSet(
        configuration="parallel",
        z=np.array([20.0, 80.0]),
        a_w=60.0,
        b=20.0,
        x=np.zeros(2),
        k=np.zeros(2),
        bore_ratio=0.5*np.ones(2),
    )


def _planetary_stage():
    return GearSet(
        configuration="planetary",
        z=np.array([20.0, 40.0, 100.0]),
        a_w=100.0,
        b=20.0,
        x=np.zeros(3),
        k=np.zeros(3),
        bore_ratio=0.5*np.ones(3),
    )


def test_parallel_set_gear_speed_matches_matlab_ratio_convention():
    gear_set = _parallel_stage()

    speeds, fixed_index, input_index = gear_set.set_gear_speed([1.0, np.nan])

    assert np.isnan(fixed_index)
    assert input_index == 0
    np.testing.assert_allclose(speeds, [1.0, 0.25])


def test_parallel_body_speed_ratios_are_input_referenced():
    gear_set = _parallel_stage()

    ratios = gear_set.body_speed_ratios()

    np.testing.assert_allclose(ratios["pinion"], 1.0)
    np.testing.assert_allclose(ratios["wheel"], 0.25)


def test_planetary_set_gear_speed_matches_fixed_ring_matlab_case():
    gear_set = _planetary_stage()

    speeds, fixed_index, input_index = gear_set.set_gear_speed([1.0, np.nan, 0.0, np.nan])

    assert fixed_index == 2
    assert input_index == 0
    np.testing.assert_allclose(speeds, [1.0, -0.25, 0.0, 1.0/6.0])


def test_planetary_body_speed_ratios_are_input_referenced():
    gear_set = _planetary_stage()

    ratios = gear_set.body_speed_ratios()

    np.testing.assert_allclose(ratios["sun"], 1.0)
    np.testing.assert_allclose(ratios["planet"], -0.25)
    np.testing.assert_allclose(ratios["ring"], 0.0)
    np.testing.assert_allclose(ratios["carrier"], 1.0/6.0)


def test_planetary_set_gear_speed_matches_fixed_carrier_matlab_case():
    gear_set = _planetary_stage()

    speeds, fixed_index, input_index = gear_set.set_gear_speed([1.0, np.nan, np.nan, 0.0])

    assert fixed_index == 3
    assert input_index == 0
    np.testing.assert_allclose(speeds, [1.0, -0.5, -0.2, 0.0])


def test_set_gear_speed_rejects_ambiguous_planetary_inputs():
    gear_set = _planetary_stage()

    with pytest.raises(ValueError, match="fixed zero-speed"):
        gear_set.set_gear_speed([1.0, np.nan, np.nan, np.nan])

    with pytest.raises(ValueError, match="exactly one nonzero input"):
        gear_set.set_gear_speed([1.0, 2.0, 0.0, np.nan])
