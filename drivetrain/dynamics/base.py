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

###############################################################################
