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
