import numpy as np

from drivetrain.dynamics import DynamicModel, model


def test_dynamic_model_is_clear_base_class_alias():
    assert DynamicModel is model


def test_state_matrix_uses_zero_damping_by_default():
    dynamic_model = model(None)
    dynamic_model.M = np.diag([2.0, 4.0])
    dynamic_model.K = np.array([[6.0, -2.0], [-2.0, 4.0]])

    actual = dynamic_model.state_matrix()

    expected = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-3.0, 1.0, 0.0, 0.0],
            [0.5, -1.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(actual, expected)


def test_frf_matches_single_dof_dynamic_stiffness_solution():
    dynamic_model = model(None)
    dynamic_model.M = np.array([[2.0]])
    dynamic_model.D = np.array([[0.5]])
    dynamic_model.K = np.array([[18.0]])

    receptance, mobility, accelerance = dynamic_model.frf(1.0, load=np.array([3.0]))

    omega = 2.0*np.pi
    expected_receptance = 3.0/(18.0 - 2.0*omega**2 + 1j*0.5*omega)

    np.testing.assert_allclose(receptance, [[expected_receptance]])
    np.testing.assert_allclose(mobility, [[1j*omega*expected_receptance]])
    np.testing.assert_allclose(accelerance, [[-omega**2*expected_receptance]])


def test_modal_truncation_projects_full_matrices_onto_selected_modes():
    mass = np.diag([2.0, 3.0])
    damping = np.diag([0.2, 0.3])
    mesh_stiffness = np.diag([20.0, 30.0])
    bearing_stiffness = np.array([[4.0, 1.0], [1.0, 6.0]])
    load = np.eye(2)
    modes = np.array([[1.0, 1.0], [1.0, -1.0]])

    reduced = model.modal_truncation(
        mass,
        damping,
        mesh_stiffness,
        bearing_stiffness,
        load,
        modes,
        mode_indices=[1],
    )

    phi = modes[:, [1]]
    np.testing.assert_allclose(reduced["M"], phi.T @ mass @ phi)
    np.testing.assert_allclose(reduced["D"], phi.T @ damping @ phi)
    np.testing.assert_allclose(reduced["K_m"], phi.T @ mesh_stiffness @ phi)
    np.testing.assert_allclose(reduced["K_b"], phi.T @ bearing_stiffness @ phi)
    np.testing.assert_allclose(reduced["b"], phi.T @ load)
    np.testing.assert_allclose(reduced["Phi"], phi)
