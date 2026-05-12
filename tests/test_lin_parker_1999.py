import numpy as np
import pytest

from drivetrain.dynamics import Lin_Parker_99
from drivetrain.models import NREL_5MW


def test_lin_parker_99_public_class_instantiates_for_nrel_5mw():
    drivetrain = NREL_5MW(dynamic_model=Lin_Parker_99)

    assert drivetrain.f_n.shape == (42,)
    assert drivetrain.mode_shape.shape == (42, 42)
    assert np.isfinite(drivetrain.f_n).sum() >= 39


def test_lin_parker_99_stage_matrix_dimensions_for_nrel_stages():
    planetary = NREL_5MW.gear_set(0)
    parallel = NREL_5MW.gear_set(2)

    planetary_inertia = Lin_Parker_99.stage_inertia_matrix(planetary)
    planetary_gyroscopic = Lin_Parker_99.stage_gyroscopic_matrix(planetary)
    planetary_stiffness = Lin_Parker_99.stage_stiffness_matrix(planetary)
    parallel_inertia = Lin_Parker_99.stage_inertia_matrix(parallel)
    parallel_gyroscopic = Lin_Parker_99.stage_gyroscopic_matrix(parallel)
    parallel_stiffness = Lin_Parker_99.stage_stiffness_matrix(parallel)

    assert planetary_inertia.shape == (18, 18)
    assert planetary_gyroscopic.shape == (18, 18)
    assert planetary_stiffness["K_b"].shape == (18, 18)
    assert planetary_stiffness["K_m"].shape == (18, 18)
    assert planetary_stiffness["K_Omega"].shape == (18, 18)
    assert parallel_inertia.shape == (9, 9)
    assert parallel_gyroscopic.shape == (9, 9)
    assert parallel_stiffness["K_b"].shape == (9, 9)
    assert parallel_stiffness["K_m"].shape == (9, 9)
    assert parallel_stiffness["K_Omega"].shape == (9, 9)


def test_lin_parker_99_stage_coordinate_change_dimensions():
    planetary = NREL_5MW.gear_set(0)
    parallel = NREL_5MW.gear_set(2)

    planetary_transform = Lin_Parker_99.stage_coordinate_change(planetary)
    parallel_transform = Lin_Parker_99.stage_coordinate_change(parallel)

    assert planetary_transform.shape == (18, 15)
    assert parallel_transform.shape == (6, 6)


def test_lin_parker_99_stage_output_shaft_inclusion_is_explicit():
    stage = NREL_5MW.gear_set(2)

    inertia_without_shaft = Lin_Parker_99.stage_inertia_matrix(stage)
    inertia_with_shaft = Lin_Parker_99.stage_inertia_matrix(stage, include_output_shaft=True)
    stiffness_without_shaft = Lin_Parker_99.stage_stiffness_matrix(stage)
    stiffness_with_shaft = Lin_Parker_99.stage_stiffness_matrix(stage, include_output_shaft=True)

    assert np.count_nonzero(inertia_without_shaft[-6:, -6:]) < np.count_nonzero(
        inertia_with_shaft[-6:, -6:]
    )
    assert np.count_nonzero(stiffness_without_shaft["K_b"][-6:, -6:]) < np.count_nonzero(
        stiffness_with_shaft["K_b"][-6:, -6:]
    )


def test_lin_parker_99_full_model_shaft_inclusion_is_explicit():
    drivetrain = NREL_5MW()

    with_shafts = Lin_Parker_99(drivetrain)
    with pytest.warns(RuntimeWarning, match="invalid value encountered in sqrt"):
        without_shafts = Lin_Parker_99(drivetrain, include_shafts=False)

    assert np.count_nonzero(without_shafts.M) < np.count_nonzero(with_shafts.M)
    assert np.count_nonzero(without_shafts.K_b) < np.count_nonzero(with_shafts.K_b)


def test_lin_parker_99_gyroscopic_matrices_are_skew_symmetric():
    drivetrain = NREL_5MW()
    dynamic_model = Lin_Parker_99(drivetrain)

    np.testing.assert_allclose(dynamic_model.G, -dynamic_model.G.T)
    assert dynamic_model.G.shape == dynamic_model.M.shape

    for stage_index in range(3):
        gyroscopic = Lin_Parker_99.stage_gyroscopic_matrix(NREL_5MW.gear_set(stage_index))
        np.testing.assert_allclose(gyroscopic, -gyroscopic.T)


def test_lin_parker_99_stage_stiffness_matrices_are_symmetric():
    for stage_index in range(3):
        stiffness = Lin_Parker_99.stage_stiffness_matrix(NREL_5MW.gear_set(stage_index))

        np.testing.assert_allclose(stiffness["K_b"], stiffness["K_b"].T)
        np.testing.assert_allclose(stiffness["K_m"], stiffness["K_m"].T)
        np.testing.assert_allclose(stiffness["K_Omega"], stiffness["K_Omega"].T)


def test_lin_parker_99_stage_total_stiffness_has_no_significant_negative_eigenvalues():
    for stage_index in range(3):
        stiffness = Lin_Parker_99.stage_stiffness_matrix(NREL_5MW.gear_set(stage_index))
        stage_total = stiffness["K_b"] + stiffness["K_m"] + stiffness["K_Omega"]
        eigenvalues = np.linalg.eigvalsh((stage_total + stage_total.T)/2)

        assert eigenvalues[0] >= -1.0e-5
