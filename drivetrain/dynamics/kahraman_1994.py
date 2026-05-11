"""Kahraman 1994 drivetrain dynamic formulation."""

import numpy as np

from .base import model


class Kahraman_94(model):
    def __init__(self, dtrain):
        super().__init__(dtrain)
        
        # number of DOFs for each stage:
        self.n_DOF = self.__calc_NDOF()
        
        self.M = self.__inertia_matrix()
        self.K = self.__stiffness_matrix()
        
        modA = self.modal_analysis()
        self.f_n = modA['f_n']
        self.mode_shape = modA['mode_shape']
    
    def __calc_NDOF(self):
        stage = self.drivetrain.stage
        Np = [0, 2]
        
        for i in range(len(stage)):
            Np.append(Np[-1] + sum([stage[i].N_p + 1 if(stage[i].configuration == 'parallel')
                                    else stage[i].N_p + 2]))
        
        return Np
        
    def __inertia_matrix(self):
        
        DT = self.drivetrain
        
        N = self.n_DOF

        M = np.zeros((N[-1], N[-1]))

        M[0 , 0 ] = DT.J_Rotor # [kg-m^2], Rotor inertia
        M[-1, -1] = DT.J_Gen   # [kg-m^2], Generator inertia
        
        i = 0
        sub_range = slice(N[i], N[i + 1])
        M[sub_range, 
          sub_range] += DT.main_shaft.inertia_matrix('torsional')
        
        for i in range(DT.N_st):
            sub_range = slice(N[i + 1] - 1, N[i + 2])
            M[sub_range, 
              sub_range] += Kahraman_94.__stage_inertia_matrix(DT.stage[i])
    
        return M

    @staticmethod
    def __stage_inertia_matrix(stage):
        
        if(stage.configuration == 'parallel'):
            J_p = stage.J_x[0]
            J_w = stage.J_x[1]
            
            M = np.diag([J_w, J_p, 0.0])
            
        elif(stage.configuration == 'planetary'):
            J_c = stage.carrier.J_x
            J_s = stage.J_x[0]
            J_p = stage.J_x[1]
            
            d = [J_c]
            [d.append(J_p) for i in range(stage.N_p)]
            d.append(J_s)
            d.append(0.0)
            
            M = np.diag(d)
        
        M[-2:, -2:] += stage.output_shaft.inertia_matrix('torsional')
        
        return M

    def __stiffness_matrix(self):
        
        DT = self.drivetrain
        N = self.n_DOF
                
        K = np.zeros((N[-1], N[-1]))

        i = 0
        sub_range = slice(N[i], N[i + 1])
        K[sub_range, 
          sub_range] += DT.main_shaft.stiffness_matrix('torsional')
        
        for i in range(DT.N_st):
            sub_range = slice(N[i + 1] - 1, N[i + 2])
            K[sub_range, 
              sub_range] += Kahraman_94.__stage_stiffness_matrix(DT.stage[i])

        return K
    
    @staticmethod
    def __stage_stiffness_matrix(stage):
        if(stage.configuration == 'parallel'):
            N = 3
            K = np.zeros((N, N))

            r_p = stage.d[0]*1.0e-3/2.0
            r_w = stage.d[1]*1.0e-3/2.0
            
            k = stage.k_mesh
            
            K[0:2, 0:2] = k*np.array([[    r_w**2, r_p*r_w],
                                   [r_p*r_w   , r_p**2]])
            
        elif(stage.configuration == 'planetary'):
            N = stage.N_p + 3
            K = np.zeros((N, N))
            
            k_1 = stage.sub_set('planet-ring').k_mesh
            k_2 = stage.sub_set('sun-planet').k_mesh

            r_c = stage.a_w*1.0e-3
            r_s = stage.d[0]*1.0e-3/2.0
            r_p = stage.d[1]*1.0e-3/2.0
            
            d = [stage.N_p*r_c*(k_1 + k_2)]
            [d.append((k_1 + k_2)*r_p**2) for i in range(stage.N_p)]
            d.append(stage.N_p*k_2*r_s**2)
            d.append(0.0)

            pla_lin     = np.ones(stage.N_p + 1)*r_c*r_p*(k_1 - k_2)
            pla_lin[-1] = -3.0*k_2*r_s*r_c
            pla_col     = np.ones(stage.N_p    )*k_2*r_p*r_s
            
            i = stage.N_p + 1
            i1 = i + 1
            K[0, 1:i1] = pla_lin
            K[1:i, -2] = pla_col 
            K += K.T
            K += np.diag(d)

        K[-2:, -2:] += stage.output_shaft.stiffness_matrix('torsional')

        return K

###############################################################################
