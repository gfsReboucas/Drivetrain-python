"""Base classes and utilities for drivetrain dynamic formulations."""

import numpy as np
import scipy.linalg as la


class model:
    def __init__(self, dtrain):
        self.drivetrain = dtrain # Drivetrain()
        
        # self.x = 0
        self.M = 0
        self.K = 0
        # self.F = 0
        
        # self.n_DOF = 0
        
        # self.f_n = 0
        # self.mode_shape = 0
        
    def modal_analysis(self):
        
        eig_val, mode_shape = la.eig(self.K, self.M, right = True)

        if(not any(np.iscomplex(eig_val))):
            eig_val = np.real(eig_val)
        else:
            print('At least one complex eigenvalue detected during the calculation of the symmetric undamped eigenvalue problem.')
        
        # lambda to omega_n:
        omega_n = np.sqrt(eig_val)
        # omega_n to Hz:
        f_n = omega_n/(2.0*np.pi)
        
        idx = np.argsort(f_n)
        f_n = f_n[idx]
        mode_shape = mode_shape[:, idx]
        
        for i in range(len(f_n)):
            j = np.argmax(abs(mode_shape[:, i]))
            mode_shape[:, i] = mode_shape[:, i]/mode_shape[j, i]

        return {
                'f_n': f_n,
                'mode_shape': mode_shape
                }

    def damping_matrix(self):
        """Return the model damping matrix, defaulting to zero damping."""
        damping = getattr(self, "D", None)
        if damping is None or np.ndim(damping) == 0:
            return np.zeros_like(np.asarray(self.M, dtype=float))
        return np.asarray(damping)

    def centripetal_force_vector(self):
        """Return the formulation centripetal force vector extension point."""
        return np.zeros(self._matrix_size())

    def external_load_vector(self):
        """Return the formulation external load influence matrix."""
        return np.eye(self._matrix_size())

    def state_matrix(self, M=None, D=None, K=None):
        """Assemble the first-order state matrix for M x'' + D x' + K x = F."""
        mass = self._square_matrix(self.M if M is None else M, "M")
        damping = self._square_matrix(self.damping_matrix() if D is None else D, "D")
        stiffness = self._square_matrix(self.K if K is None else K, "K")
        self._require_same_shape(mass, damping, stiffness)

        n = mass.shape[0]
        return np.block(
            [
                [np.zeros((n, n)), np.eye(n)],
                [-la.solve(mass, stiffness), -la.solve(mass, damping)],
            ]
        )

    def frf(self, freq, load=None, M=None, D=None, K=None):
        """Calculate receptance, mobility, and accelerance FRFs.

        Frequencies are provided in Hz. A scalar frequency returns 2-D response
        matrices; multiple frequencies return arrays with frequency as axis 0.
        """
        mass = self._square_matrix(self.M if M is None else M, "M")
        damping = self._square_matrix(self.damping_matrix() if D is None else D, "D")
        stiffness = self._square_matrix(self.K if K is None else K, "K")
        self._require_same_shape(mass, damping, stiffness)

        load_matrix = self.external_load_vector() if load is None else load
        load_matrix = np.asarray(load_matrix)
        if load_matrix.ndim == 1:
            load_matrix = load_matrix[:, np.newaxis]
        if load_matrix.shape[0] != mass.shape[0]:
            raise ValueError("load must have one row per dynamic DOF")

        scalar_frequency = np.ndim(freq) == 0
        frequencies = np.atleast_1d(freq).astype(float)

        receptance = np.empty((frequencies.size, mass.shape[0], load_matrix.shape[1]), dtype=complex)
        mobility = np.empty_like(receptance)
        accelerance = np.empty_like(receptance)

        for i, frequency in enumerate(frequencies):
            omega = 2.0*np.pi*frequency
            dynamic_stiffness = stiffness - omega**2*mass + 1j*omega*damping
            receptance[i] = la.solve(dynamic_stiffness, load_matrix)
            mobility[i] = 1j*omega*receptance[i]
            accelerance[i] = 1j*omega*mobility[i]

        if scalar_frequency:
            return receptance[0], mobility[0], accelerance[0]
        return receptance, mobility, accelerance

    @staticmethod
    def modal_truncation(M, D, K_m, K_b, b, Phi, mode_indices, precision=12):
        """Project full-order matrices onto selected zero-based modal indices."""
        phi = np.asarray(Phi)[:, mode_indices]
        eps = 10.0**(-precision)

        def clean(matrix):
            matrix = np.asarray(matrix)
            matrix[np.abs(matrix) < eps] = 0.0
            return matrix

        return {
            "M": clean(phi.T @ np.asarray(M) @ phi),
            "D": clean(phi.T @ np.asarray(D) @ phi),
            "K_m": clean(phi.T @ np.asarray(K_m) @ phi),
            "K_b": clean(phi.T @ np.asarray(K_b) @ phi),
            "b": clean(phi.T @ np.asarray(b)),
            "Phi": phi,
        }

    @staticmethod
    def newmark(time, x0, v0, M, D, K, load, option="average"):
        """Solve a linear second-order system with the Newmark method.

        The system is M x'' + D x' + K x = load. ``option`` can be
        ``"average"`` for the average-acceleration method or ``"linear"`` for
        the linear-acceleration method.
        """
        delta = 0.5
        option_key = option.lower()
        if option_key == "average":
            alpha = 0.25
        elif option_key == "linear":
            alpha = 1.0/6.0
        else:
            raise ValueError("option must be 'average' or 'linear'")

        time, x0, v0, mass, damping, stiffness, load = model._prepare_time_integration_inputs(
            time, x0, v0, M, D, K, load
        )

        dt = time[1] - time[0]
        n = mass.shape[0]
        nt = time.size
        position = np.zeros((n, nt))
        velocity = np.zeros_like(position)
        acceleration = np.zeros_like(position)

        a1 = 1.0/(alpha*dt**2)
        a2 = delta/(alpha*dt)
        a3 = 1.0/(alpha*dt)
        a4 = 1.0/(2.0*alpha) - 1.0
        a5 = delta/alpha - 1.0
        a6 = (dt/2.0)*(delta/alpha - 2.0)
        a7 = dt*(1.0 - delta)
        a8 = delta*dt

        effective_stiffness = a1*mass + a2*damping + stiffness

        position[:, 0] = x0
        velocity[:, 0] = v0
        acceleration[:, 0] = la.solve(
            mass,
            load[:, 0] - damping @ velocity[:, 0] - stiffness @ position[:, 0],
        )

        for i in range(nt - 1):
            effective_load = (
                load[:, i + 1]
                + mass @ (a1*position[:, i] + a3*velocity[:, i] + a4*acceleration[:, i])
                + damping @ (a2*position[:, i] + a5*velocity[:, i] + a6*acceleration[:, i])
            )

            position[:, i + 1] = la.solve(effective_stiffness, effective_load)
            acceleration[:, i + 1] = (
                a1*(position[:, i + 1] - position[:, i])
                - a3*velocity[:, i]
                - a4*acceleration[:, i]
            )
            velocity[:, i + 1] = (
                velocity[:, i]
                + a7*acceleration[:, i]
                + a8*acceleration[:, i + 1]
            )

        return {
            "solver": "Newmark",
            "delta": delta,
            "alpha": alpha,
            "t": time,
            "x": position,
            "v": velocity,
            "a": acceleration,
        }

    @staticmethod
    def wilson(time, x0, v0, M, D, K, load, theta=1.42):
        """Solve a linear second-order system with the Wilson-theta method."""
        time, x0, v0, mass, damping, stiffness, load = model._prepare_time_integration_inputs(
            time, x0, v0, M, D, K, load
        )

        dt = time[1] - time[0]
        n = mass.shape[0]
        nt = time.size

        position = np.zeros((n, nt))
        velocity = np.zeros_like(position)
        acceleration = np.zeros_like(position)

        load_increment = np.diff(load, axis=1)

        r1 = 6.0/(theta*dt)**2
        r2 = 3.0/(theta*dt)
        r3 = 6.0/(theta*dt)
        r4 = theta*dt/2.0
        r5 = dt/2.0
        r6 = dt**2/2.0
        r7 = dt**2/6.0

        effective_stiffness = r1*mass + r2*damping + stiffness
        a_matrix = r3*mass + 3.0*damping
        b_matrix = 3.0*mass + r4*damping

        position[:, 0] = x0
        velocity[:, 0] = v0
        acceleration[:, 0] = la.solve(
            mass,
            load[:, 0] - damping @ velocity[:, 0] - stiffness @ position[:, 0],
        )

        for i in range(nt - 1):
            effective_load_increment = (
                theta*load_increment[:, i]
                + a_matrix @ velocity[:, i]
                + b_matrix @ acceleration[:, i]
            )

            position_increment_theta = la.solve(effective_stiffness, effective_load_increment)
            acceleration_increment_theta = (
                r1*position_increment_theta
                - r3*velocity[:, i]
                - 3.0*acceleration[:, i]
            )

            acceleration_increment = acceleration_increment_theta/theta
            velocity_increment = dt*acceleration[:, i] + r5*acceleration_increment
            position_increment = (
                dt*velocity[:, i]
                + r6*acceleration[:, i]
                + r7*acceleration_increment
            )

            position[:, i + 1] = position[:, i] + position_increment
            velocity[:, i + 1] = velocity[:, i] + velocity_increment
            acceleration[:, i + 1] = acceleration[:, i] + acceleration_increment

        return {
            "solver": "Wilson",
            "theta": theta,
            "t": time,
            "x": position,
            "v": velocity,
            "a": acceleration,
        }

    def _matrix_size(self):
        return self._square_matrix(self.M, "M").shape[0]

    @staticmethod
    def _prepare_time_integration_inputs(time, x0, v0, M, D, K, load):
        time = np.asarray(time, dtype=float)
        if time.ndim != 1 or time.size < 2:
            raise ValueError("time must be a one-dimensional array with at least two points")

        dt = time[1] - time[0]
        if not np.allclose(np.diff(time), dt):
            raise ValueError("time spacing must be constant")

        mass = model._square_matrix(M, "M")
        damping = model._square_matrix(D, "D")
        stiffness = model._square_matrix(K, "K")
        model._require_same_shape(mass, damping, stiffness)

        x0 = np.atleast_1d(np.asarray(x0, dtype=float))
        v0 = np.atleast_1d(np.asarray(v0, dtype=float))
        if x0.shape != (mass.shape[0],) or v0.shape != (mass.shape[0],):
            raise ValueError("x0 and v0 must have one value per dynamic DOF")

        load = np.asarray(load, dtype=float)
        if load.ndim == 1:
            load = load[np.newaxis, :]
        if load.shape != (mass.shape[0], time.size):
            raise ValueError("load must have shape (n_dof, n_time)")

        return time, x0, v0, mass, damping, stiffness, load

    @staticmethod
    def _square_matrix(matrix, name):
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be a square matrix")
        return matrix

    @staticmethod
    def _require_same_shape(*matrices):
        shapes = {matrix.shape for matrix in matrices}
        if len(shapes) != 1:
            raise ValueError("M, D, and K must have the same shape")


DynamicModel = model

###############################################################################
