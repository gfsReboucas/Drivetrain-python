import numpy as np

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
    planetary_stiffness = Lin_Parker_99.stage_stiffness_matrix(planetary)
    parallel_inertia = Lin_Parker_99.stage_inertia_matrix(parallel)
    parallel_stiffness = Lin_Parker_99.stage_stiffness_matrix(parallel)

    assert planetary_inertia.shape == (18, 18)
    assert planetary_stiffness["K_b"].shape == (18, 18)
    assert planetary_stiffness["K_m"].shape == (18, 18)
    assert planetary_stiffness["K_Omega"].shape == (18, 18)
    assert parallel_inertia.shape == (9, 9)
    assert parallel_stiffness["K_b"].shape == (9, 9)
    assert parallel_stiffness["K_m"].shape == (9, 9)
    assert parallel_stiffness["K_Omega"].shape == (9, 9)


def test_lin_parker_99_stage_stiffness_matrices_are_symmetric():
    for stage_index in range(3):
        stiffness = Lin_Parker_99.stage_stiffness_matrix(NREL_5MW.gear_set(stage_index))

        np.testing.assert_allclose(stiffness["K_b"], stiffness["K_b"].T)
        np.testing.assert_allclose(stiffness["K_m"], stiffness["K_m"].T)
        np.testing.assert_allclose(stiffness["K_Omega"], stiffness["K_Omega"].T)
