import numpy as np
import pytest
import scipy.linalg as la

from drivetrain.dynamics import Lin_Parker_99
from drivetrain.models import NREL_5MW


def _bearing_matrix(x_stiffness, y_stiffness, torsional_stiffness=0.0):
    return np.diag([x_stiffness, y_stiffness, torsional_stiffness])


def _body_inertia_matrix(mass, polar_inertia, base_radius):
    return np.diag([mass, mass, polar_inertia/(base_radius**2)])


def _body_gyroscopic_matrix(mass):
    return np.array(
        [
            [0.0, -2.0*mass, 0.0],
            [2.0*mass, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )


def _clean_numerical_noise(matrix):
    matrix = matrix.copy()
    matrix[abs(matrix) <= 1.0e-4] = 0.0
    return matrix


def _stage_transform_to_assembled_coordinates(stage, matrix):
    transform = Lin_Parker_99.stage_coordinate_change(stage)
    return transform.T @ matrix @ transform


def _with_output_interface(matrix):
    return la.block_diag(matrix, np.zeros((3, 3)))


def _paper_stage_inertia_matrix(stage):
    if stage.configuration == "parallel":
        pinion_radius = stage.d_b[0]*1.0e-3/2
        wheel_radius = stage.d_b[1]*1.0e-3/2
        raw = la.block_diag(
            _body_inertia_matrix(stage.mass[0], stage.J_x[0], pinion_radius),
            _body_inertia_matrix(stage.mass[1], stage.J_x[1], wheel_radius),
        )
    elif stage.configuration == "planetary":
        carrier_radius = stage.a_w*1.0e-3
        ring_radius = stage.d_b[2]*1.0e-3/2
        sun_radius = stage.d_b[0]*1.0e-3/2
        planet_radius = stage.d_b[1]*1.0e-3/2
        raw = la.block_diag(
            _body_inertia_matrix(stage.carrier.mass, stage.carrier.J_x, carrier_radius),
            _body_inertia_matrix(stage.mass[2], stage.J_x[2], ring_radius),
            _body_inertia_matrix(stage.mass[0], stage.J_x[0], sun_radius),
            *[
                _body_inertia_matrix(stage.mass[1], stage.J_x[1], planet_radius)
                for _ in range(stage.N_p)
            ],
        )
    else:
        raise ValueError(stage.configuration)

    return _with_output_interface(_stage_transform_to_assembled_coordinates(stage, raw))


def _paper_stage_gyroscopic_matrix(stage):
    if stage.configuration == "parallel":
        raw = la.block_diag(
            _body_gyroscopic_matrix(stage.mass[0]),
            _body_gyroscopic_matrix(stage.mass[1]),
        )
    elif stage.configuration == "planetary":
        raw = la.block_diag(
            _body_gyroscopic_matrix(stage.carrier.mass),
            _body_gyroscopic_matrix(stage.mass[2]),
            _body_gyroscopic_matrix(stage.mass[0]),
            *[_body_gyroscopic_matrix(stage.mass[1]) for _ in range(stage.N_p)],
        )
    else:
        raise ValueError(stage.configuration)

    return _with_output_interface(_stage_transform_to_assembled_coordinates(stage, raw))


def _mesh_blocks(stage):
    alpha_n = np.radians(stage.alpha_n)

    def psi(index):
        return (index - 1)*(2*np.pi/stage.N_p)

    def psi_s(index):
        return psi(index) - alpha_n

    def psi_r(index):
        return psi(index) + alpha_n

    def ring_ring(k_mesh, index):
        return k_mesh*np.array(
            [
                [np.sin(psi_r(index))**2, -np.sin(psi_r(index))*np.cos(psi_r(index)), -np.sin(psi_r(index))],
                [-np.sin(psi_r(index))*np.cos(psi_r(index)), np.cos(psi_r(index))**2, np.cos(psi_r(index))],
                [-np.sin(psi_r(index)), np.cos(psi_r(index)), 1.0],
            ]
        )

    def ring_planet(k_mesh, index):
        return k_mesh*np.array(
            [
                [-np.sin(psi_r(index))*np.sin(alpha_n), np.sin(psi_r(index))*np.cos(alpha_n), np.sin(psi_r(index))],
                [np.cos(psi_r(index))*np.sin(alpha_n), -np.cos(psi_r(index))*np.cos(alpha_n), -np.cos(psi_r(index))],
                [np.sin(alpha_n), -np.cos(alpha_n), -1.0],
            ]
        )

    def sun_sun(k_mesh, index):
        return k_mesh*np.array(
            [
                [np.sin(psi_s(index))**2, -np.cos(psi_s(index))*np.sin(psi_s(index)), -np.sin(psi_s(index))],
                [-np.cos(psi_s(index))*np.sin(psi_s(index)), np.cos(psi_s(index))**2, np.cos(psi_s(index))],
                [-np.sin(psi_s(index)), np.cos(psi_s(index)), 1.0],
            ]
        )

    def sun_planet(k_mesh, index):
        return k_mesh*np.array(
            [
                [np.sin(psi_s(index))*np.sin(alpha_n), np.sin(psi_s(index))*np.cos(alpha_n), -np.sin(psi_s(index))],
                [-np.cos(psi_s(index))*np.sin(alpha_n), -np.cos(psi_s(index))*np.cos(alpha_n), np.cos(psi_s(index))],
                [-np.sin(alpha_n), -np.cos(alpha_n), 1.0],
            ]
        )

    def planet_sun(k_mesh):
        line_of_action = np.array([np.sin(alpha_n), np.cos(alpha_n), -1.0])
        return k_mesh*np.outer(line_of_action, line_of_action)

    def planet_ring(k_mesh):
        line_of_action = np.array([np.sin(alpha_n), -np.cos(alpha_n), -1.0])
        return k_mesh*np.outer(line_of_action, line_of_action)

    def carrier_planet(k_mesh, index):
        return k_mesh*np.array(
            [
                [-np.cos(psi(index)), np.sin(psi(index)), 0.0],
                [-np.sin(psi(index)), -np.cos(psi(index)), 0.0],
                [0.0, -1.0, 0.0],
            ]
        )

    return {
        "ring_ring": ring_ring,
        "ring_planet": ring_planet,
        "sun_sun": sun_sun,
        "sun_planet": sun_planet,
        "planet_sun": planet_sun,
        "planet_ring": planet_ring,
        "carrier_planet": carrier_planet,
    }


def _paper_parallel_stage_stiffness_matrices(stage):
    pinion_radius = stage.d_b[0]*1.0e-3/2
    wheel_radius = stage.d_b[1]*1.0e-3/2
    mesh = _mesh_blocks(stage)

    pinion_bearing = stage.bearing[3:].parallel_association()
    pinion_bearing_matrix = _bearing_matrix(
        pinion_bearing.k_y,
        pinion_bearing.k_z,
        pinion_bearing.k_alpha/(pinion_radius**2),
    )
    raw_bearing = la.block_diag(np.zeros((3, 3)), pinion_bearing_matrix)

    wheel_bearing = stage.bearing[:3].parallel_association()
    wheel_support_matrix = _bearing_matrix(
        wheel_bearing.k_y,
        wheel_bearing.k_z,
        wheel_bearing.k_alpha/(wheel_radius**2),
    )
    mesh_coupling = mesh["sun_planet"](stage.k_mesh, 1)
    raw_mesh = np.block(
        [
            [mesh["planet_sun"](stage.k_mesh) + wheel_support_matrix, mesh_coupling.T],
            [mesh_coupling, mesh["sun_sun"](stage.k_mesh, 1)],
        ]
    )
    raw_centripetal = la.block_diag(
        _bearing_matrix(stage.mass[0], stage.mass[0]),
        _bearing_matrix(stage.mass[1], stage.mass[1]),
    )

    return {
        "K_b": raw_bearing,
        "K_m": raw_mesh,
        "K_Omega": raw_centripetal,
    }


def _paper_planetary_stage_stiffness_matrices(stage):
    carrier_radius = stage.a_w*1.0e-3
    planet_radius = stage.d_b[1]*1.0e-3/2
    mesh = _mesh_blocks(stage)
    n_planet_dof = 3*stage.N_p

    carrier_bearing = stage.bearing[2:4].parallel_association()
    carrier_bearing_matrix = _bearing_matrix(
        carrier_bearing.k_y,
        carrier_bearing.k_z,
        carrier_bearing.k_alpha/(carrier_radius**2),
    )
    raw_bearing = la.block_diag(
        carrier_bearing_matrix,
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        np.zeros((n_planet_dof, n_planet_dof)),
    )

    planet_bearing = stage.bearing[:2].parallel_association()
    planet_bearing_matrix = _bearing_matrix(
        planet_bearing.k_y,
        planet_bearing.k_z,
        planet_bearing.k_alpha/(planet_radius**2),
    )
    k_sp = stage.sub_set("sun-planet").k_mesh
    k_pr = stage.sub_set("planet-ring").k_mesh

    carrier_planet_row = np.zeros((3, n_planet_dof))
    ring_planet_row = np.zeros_like(carrier_planet_row)
    sun_planet_row = np.zeros_like(carrier_planet_row)
    carrier_sum = np.zeros((3, 3))

    for planet_index in range(stage.N_p):
        index = planet_index + 1
        planet_dofs = slice(3*planet_index, 3*(planet_index + 1))
        carrier_planet_coupling = mesh["carrier_planet"](1.0, index)

        carrier_planet_row[:, planet_dofs] = (
            carrier_planet_coupling @ planet_bearing_matrix
        )
        ring_planet_row[:, planet_dofs] = mesh["ring_planet"](k_pr, index)
        sun_planet_row[:, planet_dofs] = mesh["sun_planet"](k_sp, index)
        carrier_sum += (
            carrier_planet_coupling
            @ planet_bearing_matrix
            @ carrier_planet_coupling.T
        )

    ring_sum = sum(mesh["ring_ring"](k_pr, index + 1) for index in range(stage.N_p))
    sun_sum = sum(mesh["sun_sun"](k_sp, index + 1) for index in range(stage.N_p))
    body_diagonal = la.block_diag(carrier_sum, ring_sum, sun_sum)
    body_planet_coupling = np.vstack(
        (carrier_planet_row, ring_planet_row, sun_planet_row)
    )
    planet_diagonal = planet_bearing_matrix + mesh["planet_ring"](k_pr) + mesh["planet_sun"](k_sp)
    planet_block = la.block_diag(*[planet_diagonal for _ in range(stage.N_p)])

    raw_mesh = np.block(
        [
            [body_diagonal, body_planet_coupling],
            [body_planet_coupling.T, planet_block],
        ]
    )
    raw_centripetal = la.block_diag(
        _bearing_matrix(stage.carrier.mass, stage.carrier.mass),
        _bearing_matrix(stage.mass[2], stage.mass[2]),
        _bearing_matrix(stage.mass[0], stage.mass[0]),
        *[_bearing_matrix(stage.mass[1], stage.mass[1]) for _ in range(stage.N_p)],
    )

    return {
        "K_b": raw_bearing,
        "K_m": raw_mesh,
        "K_Omega": raw_centripetal,
    }


def _paper_stage_stiffness_matrices(stage):
    if stage.configuration == "parallel":
        raw_stiffness = _paper_parallel_stage_stiffness_matrices(stage)
    elif stage.configuration == "planetary":
        raw_stiffness = _paper_planetary_stage_stiffness_matrices(stage)
    else:
        raise ValueError(stage.configuration)

    return {
        key: _clean_numerical_noise(
            _with_output_interface(
                _stage_transform_to_assembled_coordinates(stage, matrix)
            )
        )
        for key, matrix in raw_stiffness.items()
    }


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


def test_lin_parker_99_stage_inertia_and_gyroscopic_matrices_match_lp99_forms():
    # Lin and Parker write each body block as diag(m, m, J/r^2) and each
    # gyroscopic block as [[0, -2m, 0], [2m, 0, 0], [0, 0, 0]] before the
    # stage coordinate transformation.
    for stage_index in range(3):
        stage = NREL_5MW.gear_set(stage_index)

        np.testing.assert_allclose(
            Lin_Parker_99.stage_inertia_matrix(stage),
            _paper_stage_inertia_matrix(stage),
        )
        np.testing.assert_allclose(
            Lin_Parker_99.stage_gyroscopic_matrix(stage),
            _paper_stage_gyroscopic_matrix(stage),
        )


def test_lin_parker_99_stage_stiffness_matrices_match_lp99_assembly():
    # The expected matrices are assembled in the paper's raw body coordinates,
    # then mapped through stage_coordinate_change to match the implementation.
    for stage_index in range(3):
        stage = NREL_5MW.gear_set(stage_index)
        actual = Lin_Parker_99.stage_stiffness_matrix(stage)
        expected = _paper_stage_stiffness_matrices(stage)

        for key in ("K_b", "K_m", "K_Omega"):
            np.testing.assert_allclose(actual[key], expected[key], atol=1.0e-4)


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
