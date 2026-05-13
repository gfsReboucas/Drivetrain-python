import numpy as np
import scipy.linalg as la

from drivetrain.components import Material, Shaft


def _modal_frequencies(stiffness, mass):
    eigenvalues, _ = la.eig(stiffness, mass, right=True)
    eigenvalues = np.real_if_close(eigenvalues)
    eigenvalues = np.real(eigenvalues)
    eigenvalues[np.abs(eigenvalues) < 1e-8] = 0.0
    frequencies = np.sqrt(np.clip(eigenvalues, 0.0, None)) / (2.0 * np.pi)
    return np.sort(frequencies)


def _full_shaft_coordinate_transform():
    transform = np.zeros((12, 12))
    mapping = {
        0: (0, 1.0),
        1: (6, 1.0),
        2: (3, 1.0),
        3: (9, 1.0),
        4: (1, 1.0),
        5: (5, -1.0),
        6: (7, 1.0),
        7: (11, -1.0),
        8: (2, 1.0),
        9: (4, 1.0),
        10: (8, 1.0),
        11: (10, 1.0),
    }
    for component_index, (full_index, sign) in mapping.items():
        transform[component_index, full_index] = sign
    return transform


def _lin_parker_shaft_projection():
    transform = np.zeros((12, 6))
    for full_index, lp_index in [(1, 0), (2, 1), (3, 2), (7, 3), (8, 4), (9, 5)]:
        transform[full_index, lp_index] = 1.0
    return transform


def _component_shaft_gyroscopic_matrix(shaft):
    material = Material()
    length_m = shaft.L * 1.0e-3
    coupling = np.array(
        [
            [36.0, 3.0*length_m, -36.0, 3.0*length_m],
            [3.0*length_m, 4.0*length_m**2, -3.0*length_m, -length_m**2],
            [-36.0, -3.0*length_m, 36.0, -3.0*length_m],
            [3.0*length_m, -length_m**2, -3.0*length_m, 4.0*length_m**2],
        ]
    )
    coupling *= material.rho * shaft.I_x / (30.0 * length_m)

    gyroscopic = np.zeros((12, 12))
    y_dofs = slice(4, 8)
    z_dofs = slice(8, 12)
    gyroscopic[y_dofs, z_dofs] = coupling
    gyroscopic[z_dofs, y_dofs] = -coupling.T
    return gyroscopic


def test_shaft_torsional_stiffness_matches_closed_form():
    shaft = Shaft(50.0, 1000.0)
    material = Material()
    length_m = shaft.L * 1.0e-3

    expected = material.G * shaft.I_x / length_m

    np.testing.assert_allclose(shaft.stiffness("torsional"), expected, rtol=1e-12)


def test_shaft_mass_and_inertia_match_closed_form():
    shaft = Shaft(50.0, 1000.0)
    material = Material()
    diameter_m = shaft.d * 1.0e-3
    length_m = shaft.L * 1.0e-3

    expected_area = (np.pi / 4.0) * diameter_m**2
    expected_mass = material.rho * expected_area * length_m
    expected_polar_area_inertia = (np.pi / 2.0) * (diameter_m / 2.0) ** 4
    expected_mass_inertia = (expected_mass / 2.0) * (diameter_m / 2.0) ** 2

    np.testing.assert_allclose(shaft.A, expected_area, rtol=1e-12)
    np.testing.assert_allclose(shaft.mass, expected_mass, rtol=1e-12)
    np.testing.assert_allclose(shaft.I_x, expected_polar_area_inertia, rtol=1e-12)
    np.testing.assert_allclose(shaft.J_x, expected_mass_inertia, rtol=1e-12)


def test_shaft_critical_speed_matches_matlab_formula():
    shaft = Shaft(50.0, 1000.0)
    material = Material()
    length_m = shaft.L * 1.0e-3

    expected_omega = np.sqrt(material.E * shaft.I_y / (shaft.mass / length_m)) * (
        np.pi / length_m
    ) ** 2
    expected_frequency = expected_omega / (2.0 * np.pi)

    np.testing.assert_allclose(shaft.critical_speed(), expected_frequency, rtol=1e-12)


def test_shaft_safety_factors_match_existing_fatigue_calculation():
    shaft = Shaft(100.0, 1000.0)
    material = Material()
    torque = 1000.0

    expected = shaft.fatigue_yield_safety(
        material.S_ut * 1.0e-6,
        material.S_y * 1.0e-6,
        K_f=1.0,
        K_fs=1.0,
        T_m=torque,
    )

    actual = shaft.safety_factors(K_f=1.0, K_fs=1.0, T_m=torque)

    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    assert actual[0] > 1.0
    assert actual[1] > 1.0


def test_shaft_damping_matrix_defaults_to_one_percent_stiffness():
    shaft = Shaft(50.0, 1000.0)

    np.testing.assert_allclose(
        shaft.damping_matrix("torsional"),
        0.01 * shaft.stiffness_matrix("torsional"),
        rtol=1e-12,
    )


def test_shaft_axial_and_torsional_element_matrices_match_closed_form():
    shaft = Shaft(50.0, 1000.0)
    material = Material()
    length_m = shaft.L * 1.0e-3
    two_node_consistent_mass = np.array([[2.0, 1.0], [1.0, 2.0]])
    two_node_stiffness = np.array([[1.0, -1.0], [-1.0, 1.0]])

    expected_axial_mass = shaft.mass/6.0 * two_node_consistent_mass
    expected_axial_stiffness = material.E*shaft.A/length_m * two_node_stiffness
    expected_torsional_mass = material.rho*length_m*shaft.I_x/6.0 * two_node_consistent_mass
    expected_torsional_stiffness = material.G*shaft.I_x/length_m * two_node_stiffness

    np.testing.assert_allclose(shaft.inertia_matrix("axial"), expected_axial_mass, rtol=1e-12)
    np.testing.assert_allclose(
        shaft.stiffness_matrix("axial"), expected_axial_stiffness, rtol=1e-12
    )
    np.testing.assert_allclose(
        shaft.inertia_matrix("torsional"), expected_torsional_mass, rtol=1e-12
    )
    np.testing.assert_allclose(
        shaft.stiffness_matrix("torsional"), expected_torsional_stiffness, rtol=1e-12
    )


def test_shaft_bending_element_matrices_match_euler_bernoulli_reference():
    shaft = Shaft(50.0, 1000.0)
    material = Material()
    length_m = shaft.L * 1.0e-3

    expected_mass = material.rho*length_m*shaft.A/420.0 * np.array(
        [
            [156.0, 22.0*length_m, 54.0, -13.0*length_m],
            [22.0*length_m, 4.0*length_m**2, 13.0*length_m, -3.0*length_m**2],
            [54.0, 13.0*length_m, 156.0, -22.0*length_m],
            [-13.0*length_m, -3.0*length_m**2, -22.0*length_m, 4.0*length_m**2],
        ]
    )
    expected_stiffness = material.E*shaft.I_y/length_m**3 * np.array(
        [
            [12.0, 6.0*length_m, -12.0, 6.0*length_m],
            [6.0*length_m, 4.0*length_m**2, -6.0*length_m, 2.0*length_m**2],
            [-12.0, -6.0*length_m, 12.0, -6.0*length_m],
            [6.0*length_m, 2.0*length_m**2, -6.0*length_m, 4.0*length_m**2],
        ]
    )

    np.testing.assert_allclose(shaft.inertia_matrix("bending"), expected_mass, rtol=1e-12)
    np.testing.assert_allclose(
        shaft.stiffness_matrix("bending"), expected_stiffness, rtol=1e-12
    )


def test_shaft_full_matrices_match_documented_coordinate_transform():
    shaft = Shaft(50.0, 1000.0)
    component_mass = la.block_diag(
        shaft.inertia_matrix("axial"),
        shaft.inertia_matrix("torsional"),
        shaft.inertia_matrix("bending"),
        shaft.inertia_matrix("bending"),
    )
    component_stiffness = la.block_diag(
        shaft.stiffness_matrix("axial"),
        shaft.stiffness_matrix("torsional"),
        shaft.stiffness_matrix("bending"),
        shaft.stiffness_matrix("bending"),
    )
    transform = _full_shaft_coordinate_transform()

    np.testing.assert_allclose(
        shaft.inertia_matrix("full"),
        transform.T @ component_mass @ transform,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        shaft.stiffness_matrix("full"),
        transform.T @ component_stiffness @ transform,
        rtol=1e-12,
        atol=1e-6,
    )


def test_shaft_lin_parker_projection_matches_documented_coordinates():
    shaft = Shaft(50.0, 1000.0)
    transform = _lin_parker_shaft_projection()

    np.testing.assert_allclose(
        shaft.inertia_matrix("Lin_Parker_99"),
        transform.T @ shaft.inertia_matrix("full") @ transform,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        shaft.stiffness_matrix("Lin_Parker_99"),
        transform.T @ shaft.stiffness_matrix("full") @ transform,
        rtol=1e-12,
        atol=1e-6,
    )


def test_shaft_full_gyroscopic_matrix_matches_nelson_mcvaugh_reference():
    shaft = Shaft(50.0, 1000.0)
    transform = _full_shaft_coordinate_transform()
    expected = transform.T @ _component_shaft_gyroscopic_matrix(shaft) @ transform

    actual = shaft.gyroscopic_matrix("full")

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual, -actual.T, rtol=1e-12, atol=1e-12)


def test_shaft_gyroscopic_matrix_scales_with_spin_speed():
    shaft = Shaft(50.0, 1000.0)
    unit_spin = shaft.gyroscopic_matrix("full")

    np.testing.assert_allclose(
        shaft.gyroscopic_matrix("full", spin_speed=3.5),
        3.5 * unit_spin,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        shaft.gyroscopic_matrix("full", spin_speed=0.0),
        np.zeros((12, 12)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_shaft_lin_parker_gyroscopic_projection_matches_documented_coordinates():
    shaft = Shaft(50.0, 1000.0)
    transform = _lin_parker_shaft_projection()
    expected = transform.T @ shaft.gyroscopic_matrix("full") @ transform

    actual = shaft.gyroscopic_matrix("Lin_Parker_99")

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual, -actual.T, rtol=1e-12, atol=1e-12)


def test_shaft_full_matrix_modes_match_decoupled_matrix_reference():
    shaft = Shaft(50.0, 1000.0)
    expected = np.sort(
        np.concatenate(
            [
                _modal_frequencies(shaft.stiffness_matrix("axial"), shaft.inertia_matrix("axial")),
                _modal_frequencies(
                    shaft.stiffness_matrix("torsional"), shaft.inertia_matrix("torsional")
                ),
                _modal_frequencies(shaft.stiffness_matrix("bending"), shaft.inertia_matrix("bending")),
                _modal_frequencies(shaft.stiffness_matrix("bending"), shaft.inertia_matrix("bending")),
            ]
        )
    )

    actual = _modal_frequencies(
        shaft.stiffness_matrix("full"),
        shaft.inertia_matrix("full"),
    )

    np.testing.assert_allclose(actual[:6], np.zeros(6), atol=1e-5)
    np.testing.assert_allclose(actual[6:], expected[6:], rtol=1e-10, atol=1e-8)


def test_shaft_full_matrix_modes_match_matlab_reported_values():
    shaft = Shaft(50.0, 1000.0)

    actual = _modal_frequencies(
        shaft.stiffness_matrix("full"),
        shaft.inertia_matrix("full"),
    )

    np.testing.assert_allclose(
        actual,
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            273.81,
            273.81,
            935.24,
            935.24,
            1753.8,
            2827.9,
        ],
        atol=0.02,
    )
