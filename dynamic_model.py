# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:04:56 2020

@author: geraldod
"""
from numpy import pi, array, diag, argsort, sqrt, iscomplex, real, zeros, eye, ones
from scipy.linalg import eig, cholesky, inv
# import Drivetrain

class dynamic_model:
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
        
        # Cholesky decomposition:
        L = cholesky(self.M, lower = True)
        
        # Mass normalized stiffness matrix:
        K_tilde = inv(L)*self.K*inv(L.T)
       
        # correcting numeric erros and make the problem symmetric:
        # K_tilde = (K_tilde + K_tilde.T)/2.0
        
        eig_val, mode_shape = eig(K_tilde)
        
        if(not any(iscomplex(eig_val))):
            eig_val = real(eig_val)
        else:
            print('At least one complex eigenvalue detected during the calculation of the symmetric undamped eigenvalue problem.')
        
        # lambda to omega_n:
        omega_n = sqrt(eig_val)
        # omega_n to Hz:
        f_n = omega_n/(2.0*pi)
        
        idx = argsort(f_n)
        f_n = f_n[idx]
        mode_shape = mode_shape[:, idx]
        
        return {
                'f_n': f_n,
                'mode_shape': mode_shape
                }

###############################################################################

class torsional_2DOF(dynamic_model):
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
        
        M = diag([J_R, J_G*U**2])
        
        return M

    def __stiffness_matrix(self):
        DT = self.drivetrain
        U = DT.u[-1]
        
        k_LSS = DT.main_shaft.stiffness('torsional')
        k_HSS = DT.stage[-1].output_shaft.stiffness('torsional')
            
        k = (k_LSS*k_HSS*U**2)/(k_LSS + k_HSS*U**2)
        
        K = k*array([[ 1.0, -1.0],
                     [-1.0,  1.0]])
        
        return K

###############################################################################
class Kahraman_94(dynamic_model):
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
        
        M = zeros((N[-1], N[-1]))
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
            m_p = stage.mass[0]
            m_w = stage.mass[1]
            
            r_p = stage.d[0]*1.0e-3/2.0
            r_w = stage.d[1]*1.0e-3/2.0
            
            M = diag([m_w*r_w**2, m_p*r_p**2, 0.0])
            
        elif(stage.configuration == 'planetary'):
            m_c = stage.carrier.mass
            m_s = stage.mass[0]
            m_p = stage.mass[1]
            
            r_c = stage.a_w*1.0e-3
            r_s = stage.d[0]*1.0e-3/2.0
            r_p = stage.d[1]*1.0e-3/2.0
            
            d = [m_c*r_c**2]
            [d.append(m_p*r_p**2) for i in range(stage.N_p)]
            d.append(m_s*r_s**2)
            d.append(0.0)
            
            M = diag(d)
        
        M[-2:, -2:] += stage.output_shaft.inertia_matrix('torsional')
        
        return M

    def __stiffness_matrix(self):
        
        DT = self.drivetrain
        N = self.n_DOF
                
        K = zeros((N[-1], N[-1]))

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
            K = zeros((N, N))

            r_p = stage.d[0]*1.0e-3/2.0
            r_w = stage.d[1]*1.0e-3/2.0
            
            k = stage.k_mesh
            
            K[0:2, 0:2] = k*array([[    r_w**2, r_p*r_w],
                                   [r_p*r_w   , r_p**2]])
            
        elif(stage.configuration == 'planetary'):
            N = stage.N_p + 3
            K = zeros((N, N))
            
            k_1 = stage.sub_set('planet-ring').k_mesh
            k_2 = stage.sub_set('sun-planet').k_mesh

            r_c = stage.a_w*1.0e-3
            r_s = stage.d[0]*1.0e-3/2.0
            r_p = stage.d[1]*1.0e-3/2.0
            
            d = [stage.N_p*r_c*(k_1 + k_2)]
            [d.append((k_1 + k_2)*r_p**2) for i in range(stage.N_p)]
            d.append(stage.N_p*k_2*r_s**2)
            d.append(0.0)

            pla_lin     = ones(stage.N_p + 1)*r_c*r_p*(k_1 - k_2)
            pla_lin[-1] = -3.0*k_2*r_s*r_c
            pla_col     = ones(stage.N_p    )*k_2*r_p*r_s
            
            i = stage.N_p + 1
            i1 = i + 1
            K[0, 1:i1] = pla_lin
            K[1:i, -2] = pla_col 
            K += K.T
            K += diag(d)

        K[-2:, -2:] += stage.output_shaft.stiffness_matrix('torsional')

        return K
            
###############################################################################
if(__name__ == '__main__'):
    pass
