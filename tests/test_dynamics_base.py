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


def test_newmark_average_matches_matlab_reference_example():
    expected_position = np.array(
        [
            [0.0, 0.00673, 0.0505, 0.189, 0.485, 0.961, 1.58, 2.23, 2.76, 3.00, 2.85, 2.28, 1.40],
            [0.0, 0.364, 1.35, 2.68, 4.00, 4.95, 5.34, 5.13, 4.48, 3.64, 2.90, 2.44, 2.31],
        ]
    )
    mass = np.diag([2.0, 1.0])
    stiffness = np.array([[6.0, -2.0], [-2.0, 4.0]])
    damping = np.zeros((2, 2))
    time = np.arange(13)*0.28
    load = np.tile(np.array([[0.0], [10.0]]), (1, time.size))

    solution = DynamicModel.newmark(
        time,
        x0=np.zeros(2),
        v0=np.zeros(2),
        M=mass,
        D=damping,
        K=stiffness,
        load=load,
        option="average",
    )

    assert solution["solver"] == "Newmark"
    assert solution["alpha"] == 0.25
    np.testing.assert_allclose(solution["t"], time)
    np.testing.assert_allclose(solution["x"], expected_position, atol=5e-3)


def test_newmark_linear_single_dof_starts_from_consistent_acceleration():
    time = np.linspace(0.0, 1.0, 11)
    load = np.zeros(time.size)
    load[1:6] = [5.0, 8.6602, 10.0, 8.6603, 5.0]

    solution = DynamicModel.newmark(
        time,
        x0=np.array([0.0]),
        v0=np.array([0.0]),
        M=np.array([[0.2533]]),
        D=np.array([[0.1592]]),
        K=np.array([[10.0]]),
        load=load,
        option="linear",
    )

    np.testing.assert_allclose(solution["alpha"], 1.0/6.0)
    np.testing.assert_allclose(solution["a"][:, 0], [0.0])
    assert solution["x"].shape == (1, time.size)
    assert solution["v"].shape == (1, time.size)
    assert solution["a"].shape == (1, time.size)
