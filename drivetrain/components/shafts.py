"""Cylindrical shaft geometry and matrix calculations."""

from dataclasses import InitVar, dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import scipy.linalg as la

from .materials import Material
from .utils import scaling_factor

@dataclass
class Shaft:
    '''
    This class implements some geometrical concepts and parameters for 
    cylindrical shafts.
        
    References:
        [1]  Budynas, R., Nisbett, J. (2015). Shigley's Mechanical Engineering 
        Design. 10th ed. New York: McGraw-Hill
        
        [2] Neto M. A., Amaro A., Roseiro L., Cirne J., Leal R. (2015) Finite 
        Element Method for Beams. In: Engineering Computation of Structures: 
        The Finite Element Method. Springer, Cham 
        https://doi.org/10.1007/978-3-319-17710-6_4

        [3] Paz M., Kim Y.H. (2019) Dynamic Analysis of Three-Dimensional Frames. 
        In: Structural Dynamics. Springer, Cham
        https://doi.org/10.1007/978-3-319-94743-3_13
        
    written by:
        Geraldo Rebouças
        - gfs.reboucas@gmail.com
        - https://gfsreboucas.github.io
    '''
    
    dd: InitVar[float] = 700.0
    LL: InitVar[float] = 2.0e3

    def __post_init__(self, dd, LL):
        # main attributes: 
        self.d = dd # [mm], diameter
        self.L = LL # [mm], length
        
        # secondary attributes:
        # [m**2], Cross section area:
        self.A    = (np.pi/4.0)*(self.d*1.0e-3)**2
        # [m**3], Volume:
        self.V    = self.A*self.L*1.0e-3
        # [kg], mass:
        self.mass = Material().rho*self.V
        # [m**4], Area moment of inertia (x axis, rot.):
        self.I_x  = (np.pi/2.0)*(self.d*1.0e-3/2.0)**4
        # [m**4], Area moment of inertia (y axis):
        self.I_y  = self.I_x/2.0
        # [m**4], Area moment of inertia (z axis):
        self.I_z  = self.I_y
        # [kg-m**2], Mass moment of inertia (x axis, rot.):
        self.J_x  = (self.mass/2.0)*(self.d*1.0e-3/2.0)**2
        # [kg-m**2], Mass moment of inertia (y axis):
        self.J_y  = (self.mass/12.0)*(3.0*(self.d/2.0)**2 + self.L**2)*1.0e-6
        # [kg-m**2], Mass moment of inertia (z axis):
        self.J_z  = self.J_y
        
    def __repr__(self):
        val = ('Diameter,                               d   = {:.5e} mm\n'.format(self.d) +
               'Length,                                 L   = {:.5e} mm\n'.format(self.L) +
               'Mass,                                   m   = {:.5e} kg\n'.format(self.mass) +
               'Area moment of inertia, (x axis, rot.), I_x = {:.5e} m**4\n'.format(self.I_x) +
               'Mass moment of inertia, (x axis, rot.), J_x = {:.5e} kg-m**2\n'.format(self.J_x))
        
        return val
    
    def rectangle(self, C = (0, 0), color = 'r'):
       
        ax = plt.gca()
        rect = Rectangle(C, self.L, self.d, color = color, edgecolor = 'k', \
                         linestyle = '-', facecolor = color)
        
        ax.add_patch(rect)
    
    def apply_lambda(self, gamma):
        dd = self.d*scaling_factor('d', gamma)
        LL = self.L*scaling_factor('L', gamma)
        
        return Shaft(dd, LL)
    
    def stiffness(self, option):
        E = Material().E
        G = Material().G
        L = self.L*1.0e-3
        
        k = -1.0
        
        if(option == 'axial'):
            k = E*self.A/L
        elif(option == 'torsional'):
            k = G*self.I_x/L
        elif(option == 'bending'):
            k = E*self.I_y/(L**3)
        elif(option == 'full'):
            k = np.array([self.stiffness('axial'),
                       self.stiffness('torsional'),
                       self.stiffness('bending')])
        else:
            print('Option [{}] is NOT valid.'.format(option))
            
        return k
    
    def inertia_matrix(self, option):
        rho = Material().rho
        
        L = self.L*1.0e-3
        
        M = -1.0
        
        if(option == 'axial'):
            M = np.eye(2)*2.0
            M[0, 1] = 1.0
            M[1, 0] = 1.0
            
            M = M*(self.mass/6.0)
        elif(option == 'torsional'):
            M = np.eye(2)*2.0
            M[0, 1] = 1.0
            M[1, 0] = 1.0
            
            M = M*(rho*L*self.I_x/6.0)
        elif(option == 'bending'): # plane x-z
            M = np.array([[ 156    ,  22 * L   ,   54    , -13 * L   ],
                       [  22 * L,   4 * L**2,   13 * L, - 3 * L**2],
                       [  54    ,  13 * L   ,  156    , -22 * L   ],
                       [- 13 * L, - 3 * L**2, - 22 * L,   4 * L**2]])
            M = M*(rho*L*self.A/420)
        elif(option == 'full'):
            M_a = self.inertia_matrix('axial')
            M_t = self.inertia_matrix('torsional')
            M_b = self.inertia_matrix('bending')
            
            M = la.block_diag(M_a, M_t, M_b, M_b)
            
            R = np.zeros((12, 12))
            R[ 1 - 1,  1 - 1] =  1
            R[ 2 - 1,  7 - 1] =  1
            R[ 3 - 1,  4 - 1] =  1
            R[ 4 - 1, 10 - 1] =  1
            R[ 9 - 1,  3 - 1] =  1
            R[10 - 1,  5 - 1] =  1
            R[11 - 1,  9 - 1] =  1
            R[12 - 1, 11 - 1] =  1
            R[ 5 - 1,  2 - 1] =  1
            R[ 6 - 1,  6 - 1] = -1
            R[ 7 - 1,  8 - 1] =  1
            R[ 8 - 1, 12 - 1] = -1
            
            M = R.T @ M @ R
        elif(option == 'Lin_Parker_99'):
            M = self.inertia_matrix('full')
            
            R = np.zeros((12, 6))
            R[2  - 1, 1 - 1] = 1
            R[3  - 1, 2 - 1] = 1
            R[4  - 1, 3 - 1] = 1
            R[8  - 1, 4 - 1] = 1
            R[9  - 1, 5 - 1] = 1
            R[10 - 1, 6 - 1] = 1
            
            M = R.T @ M @ R
        else:
            print('Option [{}] is NOT valid.'.format(option))

        return M
    
    def stiffness_matrix(self, option):
        steel = Material()
        E   = steel.E
        G   = steel.G
        
        L = self.L*1.0e-3
        
        K = -1.0
        
        if(option == 'axial'):
            K = np.eye(2)
            K[0, 1] = -1.0
            K[1, 0] = -1.0
            
            K = K*(E*self.A/L)
        elif(option == 'torsional'):
            K = np.eye(2)
            K[0, 1] = -1.0
            K[1, 0] = -1.0
            
            K = K*(G*self.I_x/L)
        elif(option == 'bending'): # plane x-z
            K = np.array([[ 12    ,  6 * L   , -12    ,  6 * L   ],
                       [  6 * L,  4 * L**2, - 6 * L,  2 * L**2],
                       [-12    , -6 * L   ,  12    , -6 * L   ],
                       [  6 * L,  2 * L**2, - 6 * L,  4 * L**2]])
            
            K = K*(E*self.I_y/(L**3))
        elif(option == 'full'):
            K_a = self.stiffness_matrix('axial')
            K_t = self.stiffness_matrix('torsional')
            K_b = self.stiffness_matrix('bending')
            
            K = la.block_diag(K_a, K_t, K_b, K_b)
            
            R = np.zeros((12, 12))
            R[ 1 - 1,  1 - 1] =  1
            R[ 2 - 1,  7 - 1] =  1
            R[ 3 - 1,  4 - 1] =  1
            R[ 4 - 1, 10 - 1] =  1
            R[ 9 - 1,  3 - 1] =  1
            R[10 - 1,  5 - 1] =  1
            R[11 - 1,  9 - 1] =  1
            R[12 - 1, 11 - 1] =  1
            R[ 5 - 1,  2 - 1] =  1
            R[ 6 - 1,  6 - 1] = -1
            R[ 7 - 1,  8 - 1] =  1
            R[ 8 - 1, 12 - 1] = -1
            
            K = R.T @ K @ R
        elif(option == 'Lin_Parker_99'):
            K = self.stiffness_matrix('full')
            
            R = np.zeros((12, 6))
            R[ 2 - 1, 1 - 1] = 1
            R[ 3 - 1, 2 - 1] = 1
            R[ 4 - 1, 3 - 1] = 1
            R[ 8 - 1, 4 - 1] = 1
            R[ 9 - 1, 5 - 1] = 1
            R[10 - 1, 6 - 1] = 1
            
            K = R.T @ K @ R
        else:
            print('Option [{}] is NOT valid.'.format(option))

        # if(not allclose(K, K.T)):
        #     K = (K + K.T)/2
        
        return K

    def damping_matrix(self, option, beta=0.01):
        return beta*self.stiffness_matrix(option)

    def critical_speed(self):
        '''
        Returns the shaft first critical speed, [Hz].

        This follows the MATLAB Shaft.critical_speed implementation for a
        uniform circular shaft using the first bending mode approximation.
        '''
        E = Material().E
        L = self.L*1.0e-3
        omega_1 = np.sqrt(E*self.I_y/(self.mass/L))*(np.pi/L)**2
        return omega_1/(2.0*np.pi)

    def safety_factors(self, K_f, K_fs, T_m):
        '''
        Calculates shaft fatigue and yielding safety factors.

        This is the MATLAB-compatible public wrapper. The lower-level
        fatigue_yield_safety method expects strengths in MPa.
        '''
        material = Material()
        return self.fatigue_yield_safety(
            material.S_ut*1.0e-6,
            material.S_y*1.0e-6,
            K_f,
            K_fs,
            T_m,
        )
    
    def fatigue_yield_safety(self, S_ut, S_y, K_f, K_fs, T_m):
        '''
        Calculates the safety factors for fatigue and yielding for a circular 
        shaft according to [1].
        
        Input parameters:
            - S_ut: Tensile strength, [MPa]
            - S_y:  Yield strength, [MPa]
            - K_f:  Fatigue stress-concentration factor for bending
            - K_fs: Fatigue stress-concentration factor for torsion
            - T_m: Midrange torque, [N-m]
        
        Assumptions:
            - Solid shaft with round cross section.
            - Axial loads are neglected.
            - Using the distortion energy failure theory
        '''
        
        dd = self.d
        # Endurance limit:
        if(S_ut <= 1400): # [MPa]
            S_e_prime = 0.5*S_ut
        else:
            S_e_prime = 700.0
        
        # Surface factor:
        # k_a =  1.58*S_ut**(-0.085) # Ground
        k_a =  4.51*S_ut**(-0.265) # Machined or cold-drawn
        # k_a =  57.7*S_ut**(-0.718) # Hot-rolled
        # k_a = 272.0*S_ut**(-0.995) # As-forged
        
        # Size factor:
        if((0.11*25.4 <= dd) or (dd <= 2.0*25.4)): # [mm]
            k_b = (dd/7.62)**(-0.107)
        elif((2.0*25.4 <= dd) or (dd <= 10.0*25.4)):
            k_b = 1.51*dd**(-0.157)
        
        # Loading factor:
        # k_c = 1.0  # Bending
        # k_c = 0.85 # Axial
        k_c = 0.59 # Torsion
        
        # Temperature factor: 70 <= T_F <= 1000 Fahrenheit
        T_F = 0.0
        k_d = 0.975 + (0.432e-3)*T_F - (0.115e-5)*T_F**2 + (0.104e-8)*T_F**3 - (0.595e-12)*T_F**4
        
        # Reliability factor:
        # z_a = 1.288 # 90%
        z_a = 1.645 # 95%
        # z_a = 2.326 # 99%
        
        k_e = 1.0 - 0.08*z_a
        
        # Miscellaneous-Effects factor:
        k_f = 1.0
        
        # Endurance limit at the critical location of a machine part in the 
        # geometry and condition of use:
        S_e = k_a*k_b*k_c*k_d*k_e*k_f*S_e_prime
        
        M_m = 0.0 # Midrange bending moment
        M_a = 0.0 # Alternating bending moment
        # T_m = 0.0 # Midrange torque
        T_a = 0.0 # Alternating torque
        
        dd = self.d*1.0e-3
        S_e = S_e*1.0e6
        S_y = S_y*1.0e6
        
        # Distortion energy theory and ASME fatigue criteria:
        val = (16.0/(np.pi*dd**3))*np.sqrt(4.0*(K_f*M_a/S_e)**2 + 3.0*(K_fs*T_a/S_e)**2 + \
                                     4.0*(K_f*M_m/S_y)**2 + 3.0*(K_fs*T_m/S_y)**2)
        n = 1.0/val # n is proportional to d**3 T**-1
        
        # Yielding factor of safety:
        sigma_Max = np.sqrt(3.0*(16.0*T_m/(np.pi*dd**3)))
        
        n_y = S_y/sigma_Max
        
        return n, n_y
