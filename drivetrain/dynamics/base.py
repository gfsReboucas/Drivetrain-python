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

    def _matrix_size(self):
        return self._square_matrix(self.M, "M").shape[0]

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
