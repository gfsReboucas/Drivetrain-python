"""Two-degree-of-freedom torsional drivetrain formulation."""

import numpy as np

from .base import model


class torsional_2DOF(model):
    def __init__(self, dtrain):
        super().__init__(dtrain)
        self.n_DOF = 2

        self.M = self.__inertia_matrix()
        self.K = self.__stiffness_matrix()
        
        modA = self.modal_analysis()
        self.f_n = modA['f_n']
        self.mode_shape = modA['mode_shape']
        
    def __inertia_matrix(self):
        DT = self.drivetrain
        
        J_R = DT.J_Rotor # [kg-m^2], Rotor inertia
        J_G = DT.J_Gen   # [kg-m^2], Generator inertia

        U = DT.u[-1]
        
        M = np.diag([J_R, J_G*U**2])
        
        return M

    def __stiffness_matrix(self):
        DT = self.drivetrain
        U = DT.u[-1]
        
        k_LSS = DT.main_shaft.stiffness('torsional')
        k_HSS = DT.stage[-1].output_shaft.stiffness('torsional')
            
        k = (k_LSS*k_HSS*U**2)/(k_LSS + k_HSS*U**2)
        
        K = k*np.array([[ 1.0, -1.0],
                     [-1.0,  1.0]])
        
        return K

###############################################################################
