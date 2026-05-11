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


def test_time_response_defaults_to_scipy_solve_ivp():
    time = np.linspace(0.0, 1.0, 11)
    load = np.ones(time.size)

    solution = DynamicModel.time_response(
        time,
        x0=np.array([0.0]),
        v0=np.array([0.0]),
        M=np.array([[2.0]]),
        D=np.array([[0.0]]),
        K=np.array([[0.0]]),
        load=load,
    )

    expected_acceleration = np.full((1, time.size), 0.5)
    expected_velocity = time[np.newaxis, :]*0.5
    expected_position = 0.25*time[np.newaxis, :]**2

    assert solution["solver"] == "solve_ivp"
    assert solution["method"] == "Radau"
    np.testing.assert_allclose(solution["a"], expected_acceleration, atol=1e-9)
    np.testing.assert_allclose(solution["v"], expected_velocity, atol=1e-9)
    np.testing.assert_allclose(solution["x"], expected_position, atol=1e-9)


def test_time_response_dispatches_to_fixed_step_helpers():
    time = np.linspace(0.0, 0.2, 3)
    mass = np.array([[2.0]])
    damping = np.array([[0.0]])
    stiffness = np.array([[0.0]])
    load = np.ones(time.size)

    newmark = DynamicModel.time_response(
        time,
        x0=np.array([0.0]),
        v0=np.array([0.0]),
        M=mass,
        D=damping,
        K=stiffness,
        load=load,
        method="newmark",
    )
    bathe = DynamicModel.time_response(
        time,
        x0=np.array([0.0]),
        v0=np.array([0.0]),
        M=mass,
        D=damping,
        K=stiffness,
        load=load,
        method="bathe",
    )

    assert newmark["solver"] == "Newmark"
    assert bathe["solver"] == "Bathe"


def test_wilson_free_mass_constant_force_matches_exact_response():
    time = np.linspace(0.0, 1.0, 11)
    load = np.ones(time.size)

    solution = DynamicModel.wilson(
        time,
        x0=np.array([0.0]),
        v0=np.array([0.0]),
        M=np.array([[2.0]]),
        D=np.array([[0.0]]),
        K=np.array([[0.0]]),
        load=load,
    )

    expected_acceleration = np.full((1, time.size), 0.5)
    expected_velocity = time[np.newaxis, :]*0.5
    expected_position = 0.25*time[np.newaxis, :]**2

    assert solution["solver"] == "Wilson"
    assert solution["theta"] == 1.42
    np.testing.assert_allclose(solution["a"], expected_acceleration, atol=1e-14)
    np.testing.assert_allclose(solution["v"], expected_velocity, atol=1e-14)
    np.testing.assert_allclose(solution["x"], expected_position, atol=1e-14)


def test_wilson_accepts_matlab_five_dof_reference_problem_shape():
    n = 5
    mass_per_floor = 208.6
    bending_stiffness = 5.469e10
    height = 120.0
    load_value = 1.0e3
    time = np.arange(0.0, 2.0 + 0.1, 0.1)

    mass = mass_per_floor*np.diag([1.0, 1.0, 1.0, 1.0, 0.5])
    stiffness_diagonal = np.diag([18.83, 14.65, 14.06, 9.878, 1.608])
    stiffness_upper = np.zeros((n, n))
    stiffness_upper[0, 1] = -11.90
    stiffness_upper[0, 2] = 4.773
    stiffness_upper[0, 3] = -1.193
    stiffness_upper[0, 4] = 0.1989
    stiffness_upper[1, 2] = -10.71
    stiffness_upper[1, 3] = 4.177
    stiffness_upper[1, 4] = -0.6961
    stiffness_upper[2, 3] = -9.514
    stiffness_upper[2, 4] = 2.586
    stiffness_upper[3, 4] = -3.646
    stiffness = (stiffness_upper + stiffness_upper.T + stiffness_diagonal)*bending_stiffness/(height**3)

    load = np.zeros((n, time.size))
    load[-1, 1:] = load_value

    solution = DynamicModel.wilson(
        time,
        x0=np.zeros(n),
        v0=np.zeros(n),
        M=mass,
        D=np.zeros((n, n)),
        K=stiffness,
        load=load,
    )

    assert solution["x"].shape == (n, time.size)
    assert solution["v"].shape == (n, time.size)
    assert solution["a"].shape == (n, time.size)
    assert np.all(np.isfinite(solution["x"]))


def test_bathe_matches_matlab_reference_example():
    expected_position = np.array(
        [
            [0.0, 0.00458, 0.0445, 0.183, 0.486, 0.979, 1.62, 2.28, 2.81, 3.03, 2.83, 2.21, 1.28],
            [0.0, 0.373, 1.38, 2.73, 4.04, 4.97, 5.31, 5.06, 4.38, 3.55, 2.85, 2.46, 2.40],
        ]
    )
    mass = np.diag([2.0, 1.0])
    stiffness = np.array([[6.0, -2.0], [-2.0, 4.0]])
    damping = np.zeros((2, 2))
    time = np.arange(13)*0.28
    load = np.tile(np.array([[0.0], [10.0]]), (1, time.size))

    solution = DynamicModel.bathe(
        time,
        x0=np.zeros(2),
        v0=np.zeros(2),
        M=mass,
        D=damping,
        K=stiffness,
        load=load,
    )

    assert solution["solver"] == "Bathe"
    np.testing.assert_allclose(solution["t"], time)
    np.testing.assert_allclose(solution["x"], expected_position, atol=5e-3)
