# -*- coding: utf-8 -*-
"""
Written by:
    Geraldo Rebouças
    - Geraldo.Reboucas@ntnu.no OR
    - gfs.reboucas@gmail.com

    Postdoctoral Fellow at:
        Norwegian University of Science and Technology, NTNU
        Department of Marine Technology, IMT
        Marine System Dynamics and Vibration Lab, MD Lab
        https://www.ntnu.edu/imt/lab/md-lab

@author: geraldod
"""

# import sys
from numpy import pi, sin, tan, radians, isscalar, mean, eye, allclose, diag, \
                  sqrt, zeros
from scipy import interpolate, array
from scipy.stats import hmean
from scipy.linalg import block_diag
from matplotlib.pyplot import gca
from matplotlib.patches import Rectangle

###############################################################################
def check_key(key, dic):
    '''
    check if key is a part of any key from 

    Parameters
    ----------
    key : string
        DESCRIPTION.
    dic : dict
        DESCRIPTION.

    Returns
    -------
    val : float
        DESCRIPTION.

    '''
    val = 1.0
    for k, v in dic.items():
        if(key in k):
            val = v
        
    return val

###############################################################################

class Material:
    '''
    Simple class to store some properties of materials used to manufacture
    gears.
    '''
    def __init__(self):
        self.E          =  206.0e9  # [Pa],     Young's modulus
        self.nu         =    0.3    # [-],      Poisson's ratio
        self.sigma_Hlim = 1500.0e6  # [Pa],     Allowable contact stress number
        self.rho        =    7.83e3 # [kg/m**3], Density
        self.S_ut       =  700.0e6  # [Pa],     Tensile strength
        self.S_y        =  490.0e6  # [Pa],     Yield strength
        
        # % [Pa],     Shear modulus
        self.G = (self.E/2.0)/(1.0 + self.nu)
        
###############################################################################

class Rack:
    '''
    Implements some characteristics of the standard basic rack tooth profile 
    for cylindrical involute gears (external or internal) for general and 
    heavy engineering.
    
    References:
        [1] ISO 53:1998 Cylindrical gears for general and heavy engineering 
        -- Standard basic rack tooth profile
        
        [2] ISO 54:1996 Cylindrical gears for general engineering and for 
        heavy engineering -- Modules
       
    written by:
        Geraldo Rebouças
        - Geraldo.Reboucas@ntnu.no OR
        - gfs.reboucas@gmail.com
        
        Postdoctoral Fellow at:
            Norwegian University of Science and Technology, NTNU
            Department of Marine Technology, IMT
            Marine System Dynamics and Vibration Lab, MD Lab
            https://www.ntnu.edu/imt/lab/md-lab
    '''
    
    def __init__(self, **kwargs):
        # main attributes:
        # [-],    Type of basic rack tooth profile:
        self.type    = kwargs['type']    if('type'    in kwargs) else 'A'
        # [mm],   Module:
        mm           = kwargs['m']       if('m'       in kwargs) else 1.0
        self.m       = self.module(mm)
        # [deg.], Pressure angle:
        self.alpha_P = kwargs['alpha_P'] if('alpha_P' in kwargs) else 20.0
        
        # secondary attributes:
        if(self.type == 'A'):
            k_c_P    = 0.25
            k_rho_fP = 0.38
        elif(self.type == 'B'):
            k_c_P    = 0.25
            k_rho_fP = 0.30
        elif(self.type == 'C'):
            k_c_P    = 0.25
            k_rho_fP = 0.25
        elif(self.type == 'D'):
            k_c_P    = 0.40
            k_rho_fP = 0.39
        else:
            print('Rack type [{}] is NOT defined.'.format(self.type))
            
        self.c_P   = k_c_P*self.m          # [mm], Bottom clearance
        self.h_fP  = (1.0 + k_c_P)*self.m  # [mm], Dedendum
        self.e_P   = self.m/2.0            # [mm], Spacewidth
        self.h_aP  = self.m                # [mm], Addendum
        self.h_FfP = self.h_fP - self.c_P  # [mm], Straight portion of the dedendum
        self.h_P   = self.h_aP + self.h_fP # [mm], Tooth depth
        self.h_wP  = self.h_P - self.c_P   # [mm], Common depth of rack and tooth
        self.p     = pi*self.m             # [mm], Pitch
        self.s_P   = self.e_P              # [mm], Tooth thickness
        self.rho_fP = k_rho_fP*self.m      # [mm], Fillet radius
        
        self.U_FP = 0.0     # [mm],   Size of undercut
        self.alpha_FP = 0.0 # [deg.], Angle of undercut

    def __repr__(self):
        '''
        Return a string containing a printable representation of an object.

        Returns
        -------
        None.

        '''
        val = ('Rack type:                   {}     -\n'.format(self.type) +
               'Module,          m       = {:7.3f} mm\n'.format(self.m) +
               'Pressure angle,  alpha_P = {:7.3f} deg.\n'.format(self.alpha_P) +
               'Addendum,        h_aP    = {:7.3f} mm\n'.format(self.h_aP) +
               'Dedendum,        h_fP    = {:7.3f} mm\n'.format(self.h_fP) +
               'Tooth depth,     h_P     = {:7.3f} mm\n'.format(self.h_P) +
               'Pitch,           p       = {:7.3f} mm\n'.format(self.p) +
               'Tooth thickness, s_P     = {:7.3f} mm\n'.format(self.s_P) +
               'Fillet radius,   rho_fP  = {:7.3f} mm\n'.format(self.rho_fP))
        
        return val
       
    def save(self, filename):
        pass
    
    @staticmethod
    def module(m_x, **kwargs):
        '''
        Returns the values of normal modules for straight and helical gears 
        according to ISO 54:1996 [2]. According to this standard, preference
        should be given to the use of the normal modules as given in series I 
        and the module 6.5 in series II should be avoided.
            
        The module is defined as the quotient between:
            - the pitch, expressed in millimetres, to the number pi, or;
            - the reference diameter, expressed in millimetres, by the number 
            of teeth.
        '''
        option = kwargs['option'] if('option' in kwargs) else 'calc'
        method = kwargs['method'] if('method' in kwargs) else 'nearest'
        
        m_1 = [1.000, 1.250, 1.50, 2.00, 2.50, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0, 32.0, 40.0, 50.0]
        m_2 = [1.125, 1.375, 1.75, 2.25, 2.75, 3.5, 4.5, 5.5, 6.5, 7.0,  9.0, 11.0, 14.0, 18.0, 22.0, 28.0, 36.0, 45.0]
        
        if(option == 'show'):
            print('... to be implemented...')
        elif(option == 'calc'):
            # removing the value 6.5 which should be avoided according to [2]
            # and inserting additional values due to (Nejad et. al., 2015) and
            # (Wang et. al., 2020). See the classes NREL_5MW and DTU_10MW for 
            # detailed references about these works.
            m_2[8] = 21.0 
            m_2.append(30.0)
            x = sorted(m_1 + m_2)
        elif(option == 'calc_1'):
            x = m_1
        elif(option == 'calc_2'):
            x = m_2
        else:
            print('Option [{}] is NOT valid.'.format(option))
        
        idx = interpolate.interp1d(x, list(range(len(x))), kind = method, 
                                   fill_value = (0, len(x) - 1))
        return x[idx(m_x).astype(int)]

    def max_fillet_radius(self):
        ''' 
        returns the maximum fillet radius of the basic rack according to 
        ISO 53:1998 [1], Sec. 5.9.
        '''
        
        if(self.alpha_P != 20.0):
            print('The pressure angle is not 20.0 [deg.]')
        else:
            if((self.c_P <= 0.295*self.m) and (self.h_FfP == self.m)):
                rho_fP_max = self.c_P/(1.0 - sin(radians(self.alpha_P)))
            elif((0.295*self.m < self.c_P) and (self.c_P <= 0.396*self.m)):
                rho_fP_max = (pi*self.m/4.0 - self.h_fP*tan(radians(self.alpha_P)))/tan(radians(90.0 - self.alpha_P)/2.0)
            
        return rho_fP_max

###############################################################################

class Bearing:
    '''
    This class contains some geometric and dynamic properties of rolling 
    bearings.
    
    written by:
        Geraldo Rebouças
        - Geraldo.Reboucas@ntnu.no OR
        - gfs.reboucas@gmail.com
        
        Postdoctoral Fellow at:
            Norwegian University of Science and Technology, NTNU
            Department of Marine Technology, IMT
            Marine System Dynamics and Vibration Lab, MD Lab
            https://www.ntnu.edu/imt/lab/md-lab
    '''
    
    def __init__(self, stiffness = zeros(6), damping = zeros(6), **kwargs):
        
        # [N/m],     Translational stiffness, x axis:
        self.k_x     = stiffness[0]
        # [N/m],     Translational stiffness, y axis:
        self.k_y     = stiffness[1]
        # [N/m],     Translational stiffness, z axis:
        self.k_z     = stiffness[2]
        # [N-m/rad], Torsional stiffness, x axis (rot.):
        self.k_alpha = stiffness[3]
        # [N-m/rad], Torsional stiffness, y axis:
        self.k_beta  = stiffness[4]
        # [N-m/rad], Torsional stiffness, z axis:
        self.k_gamma = stiffness[5]
        
        # [N-s/m],     Translational damping, x axis:
        self.d_x     = damping[0]
        # [N-s/m],     Translational damping, y axis:
        self.d_y     = damping[1]
        # [N-s/m],     Translational damping, z axis:
        self.d_z     = damping[2]
        # [N-m-s/rad], Torsional damping, x axis (rot.):
        self.d_alpha = damping[3]
        # [N-m-s/rad], Torsional damping, y axis:
        self.d_beta  = damping[4]
        # [N-m-s/rad], Torsional damping, z axis:
        self.d_gamma = damping[5]
        
        # [-],       Bearing designation:
        self.name = kwargs['name'] if('name' in kwargs) else '-*-'
        # [-],       Bearing type:
        self.type = kwargs['type'] if('type' in kwargs) else 'none'
        # [mm],      Outer diameter:
        self.OD   = kwargs['OD']   if('OD'   in kwargs) else 0.0
        # [mm],      Inner diameter
        self.ID   = kwargs['ID']   if('ID'   in kwargs) else 0.0
        # [mm],      Thickness
        self.B    = kwargs['B']    if('B'    in kwargs) else 0.0
        
    def __repr__(self):
        
        val = ('Bearing type:                                  {}   -\n'.format(self.type) +
               'Translational Stiffness @ x axis,    k_x     = {:.5e} N/m\n'.format(self.k_x) +
               'Translational Stiffness @ y axis,    k_y     = {:.5e} N/m\n'.format(self.k_y) +
               'Translational Stiffness @ z axis,    k_z     = {:.5e} N/m\n'.format(self.k_z) +
               'Torsional Stiffness @ x axis (rot.), k_alpha = {:.5e} N-m/rad\n'.format(self.k_alpha) +
               'Torsional Stiffness @ y axis,        k_beta  = {:.5e} N-m/rad\n'.format(self.k_beta) +
               'Torsional Stiffness @ z axis,        k_gamma = {:.5e} N-m/rad\n'.format(self.k_gamma) +
               'Translational damping @ x axis,      d_x     = {:.5e} N-s/m\n'.format(self.d_x) +
               'Translational damping @ y axis,      d_y     = {:.5e} N-s/m\n'.format(self.d_y) +
               'Translational damping @ z axis,      d_z     = {:.5e} N-s/m\n'.format(self.d_z) +
               'Torsional damping @ x axis (rot.),   d_alpha = {:.5e} N-m-s/rad\n'.format(self.d_alpha) +
               'Torsional damping @ y axis,          d_beta  = {:.5e} N-m-s/rad\n'.format(self.d_beta) +
               'Torsional damping @ z axis,          d_gamma = {:.5e} N-m-s/rad\n'.format(self.d_gamma) +
               'Outer diameter,                      OD      = {} mm\n'.format(self.OD) +
               'Inner diameter,                      ID      = {} mm\n'.format(self.ID) +
               'Width,                               B       = {} mm\n'.format(self.B))
        
        return val

    def __len__(self):
        return len(self.k_x)

    def __getitem__(self, key):
        return Bearing(array([self.k_x[key],
                              self.k_y[key],
                              self.k_z[key],
                              self.k_alpha[key],
                              self.k_beta[key],
                              self.k_gamma[key]]),
                       array([self.d_x[key],
                              self.d_y[key],
                              self.d_z[key],
                              self.d_alpha[key],
                              self.d_beta[key],
                              self.d_gamma[key]]))
                             
    def series_association(self):
        if(isscalar(self.k_x)):
            print('Only one bearing.')
            return self
        else:
            kx = hmean(self.k_x)    /self.k_x.size()
            ky = hmean(self.k_y)    /self.k_y.size()
            kz = hmean(self.k_z)    /self.k_z.size()
            ka = hmean(self.k_alpha)/self.k_alpha.size()
            kb = hmean(self.k_beta) /self.k_beta.size()
            kg = hmean(self.k_gamma)/self.k_gamma.size()
            
            dx = hmean(self.d_x)    /self.d_x.size()
            dy = hmean(self.d_y)    /self.d_y.size()
            dz = hmean(self.d_z)    /self.d_z.size()
            da = hmean(self.d_alpha)/self.d_alpha.size()
            db = hmean(self.d_beta) /self.d_beta.size()
            dg = hmean(self.d_gamma)/self.d_gamma.size()
            
        k = [kx, ky, kz, ka, kb, kg]
        d = [dx, dy, dz, da, db, dg]
            
        return Bearing(k, d, name = ' / '.join(self.name), type = 'series', \
                       OD = mean(self.OD), ID = mean(self.ID), B = mean(self.B))
        
    def parallel_association(self):
        if(isscalar(self.k_x)):
            print('Only one bearing.')
            return self
        else:
            kx = sum(self.k_x)
            ky = sum(self.k_y)
            kz = sum(self.k_z)
            ka = sum(self.k_alpha)
            kb = sum(self.k_beta)
            kg = sum(self.k_gamma)
            
            dx = sum(self.d_x)
            dy = sum(self.d_y)
            dz = sum(self.d_z)
            da = sum(self.d_alpha)
            db = sum(self.d_beta)
            dg = sum(self.d_gamma)
            
        k = [kx, ky, kz, ka, kb, kg]
        d = [dx, dy, dz, da, db, dg]
            
        return Bearing(k, d, name = ' / '.join(self.name), type = 'parallel', \
                       OD = mean(self.OD), ID = mean(self.ID), B = mean(self.B))
        
    def stiffness_matrix(self):
        if(isscalar(self.k_x)):
            return diag([self.k_x,     self.k_y,    self.k_z, 
                         self.k_alpha, self.k_beta, self.k_gamma])
        else:
            print('Only one bearing.')
        
    def damping_matrix(self):
        if(isscalar(self.d_x)):
            return diag([self.d_x,     self.d_y,    self.d_z, 
                         self.d_alpha, self.d_beta, self.d_gamma])
        else:
            print('Only one bearing.')

###############################################################################

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
        - geraldo.reboucas@ntnu.no OR
        - gfs.reboucas@gmail.com
            
        Postdoctoral Fellow at:
            Norwegian University of Science and Technology, NTNU
            Department of Marine Technology, IMT
            Marine System Dynamics and Vibration Lab, MD Lab
            https://www.ntnu.edu/imt/lab/md-lab
    '''
    
    def __init__(self, dd = 700.0, LL = 2.0e3):
        
        # main attributes: 
        self.d = dd # [mm], diameter
        self.L = LL # [mm], length
        
        # secondary attributes:
        # [m**2], Cross section area:
        self.A    = (pi/4.0)*(self.d*1.0e-3)**2
        # [m**3], Volume:
        self.V    = self.A*self.L*1.0e-3
        # [kg], mass:
        self.mass = Material().rho*self.V
        # [m**4], Area moment of inertia (x axis, rot.):
        self.I_x  = (pi/2.0)*(self.d*1.0e-3/2.0)**4
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
       
        ax = gca()
        rect = Rectangle(C, self.L, self.d, color = color, edgecolor = 'k', \
                         linestyle = '-', facecolor = color)
        
        ax.add_patch(rect)
    
    def apply_lambda(self, gamma):
        dd = self.d*check_key('d', gamma)
        LL = self.L*check_key('L', gamma)
        
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
            k = array([self.stiffness('axial'),
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
            M = eye(2)*2.0
            M[0, 1] = 1.0
            M[1, 0] = 1.0
            
            M = M*(self.mass/6.0)
        elif(option == 'torsional'):
            M = eye(2)*2.0
            M[0, 1] = 1.0
            M[1, 0] = 1.0
            
            M = M*(rho*L*self.I_x/6.0)
        elif(option == 'bending'): # plane x-z
            M = array([[ 156    ,  22 * L   ,   54    , -13 * L   ],
                       [  22 * L,   4 * L**2,   13 * L, - 3 * L**2],
                       [  54    ,  13 * L   ,  156    , -22 * L   ],
                       [- 13 * L, - 3 * L**2, - 22 * L,   4 * L**2]])
            M = M*(rho*L*self.A/420)
        elif(option == 'full'):
            M_a = self.inertia_matrix('axial')
            M_t = self.inertia_matrix('torsional')
            M_b = self.inertia_matrix('bending')
            
            M = block_diag(M_a, M_t, M_b, M_b)
            
            R = zeros((12, 12))
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
            
            R = zeros((12, 6))
            R[2  - 1, 1 - 1] = 1
            R[3  - 1, 2 - 1] = 1
            R[4  - 1, 3 - 1] = 1
            R[8  - 1, 4 - 1] = 1
            R[9  - 1, 5 - 1] = 1
            R[10 - 1, 6 - 1] = 1
            
            M = R.T @ M @ R
        else:
            print('Option [{}] is NOT valid.'.format(option))

        # if(not allclose(M, M.T)):
        #     M = (M + M.T)/2
        
        return M
    
    def stiffness_matrix(self, option):
        steel = Material()
        E   = steel.E
        G   = steel.G
        
        L = self.L*1.0e-3
        
        K = -1.0
        
        if(option == 'axial'):
            K = eye(2)
            K[0, 1] = -1.0
            K[1, 0] = -1.0
            
            K = K*(E*self.A/L)
        elif(option == 'torsional'):
            K = eye(2)
            K[0, 1] = -1.0
            K[1, 0] = -1.0
            
            K = K*(G*self.I_x/L)
        elif(option == 'bending'): # plane x-z
            K = array([[ 12    ,  6 * L   , -12    ,  6 * L   ],
                       [  6 * L,  4 * L**2, - 6 * L,  2 * L**2],
                       [-12    , -6 * L   ,  12    , -6 * L   ],
                       [  6 * L,  2 * L**2, - 6 * L,  4 * L**2]])
            
            K = K*(E*self.I_y/(L**3))
        elif(option == 'full'):
            K_a = self.stiffness_matrix('axial')
            K_t = self.stiffness_matrix('torsional')
            K_b = self.stiffness_matrix('bending')
            
            K = block_diag(K_a, K_t, K_b, K_b)
            
            R = zeros((12, 12))
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
            
            R = zeros((12, 6))
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
        val = (16.0/(pi*dd**3))*sqrt(4.0*(K_f*M_a/S_e)**2 + 3.0*(K_fs*T_a/S_e)**2 + \
                                     4.0*(K_f*M_m/S_y)**2 + 3.0*(K_fs*T_m/S_y)**2)
        n = 1.0/val # n is proportional to d**3 T**-1
        
        # Yielding factor of safety:
        sigma_Max = sqrt(3.0*(16.0*T_m/(pi*dd**3)))
        
        n_y = S_y/sigma_Max
        
        return n, n_y
            
###############################################################################

if(__name__ == '__main__'):
    # rack = Rack(type='A', m=60, alpha_P=20.0)
    # rack.print()
    brg = Bearing(array([[0.0   , 1.50e10, 1.50e10, 0.0, 5.0e6, 5.0e6],
                         [4.06e8, 1.54e10, 1.54e10, 0.0, 0.0  , 0.0  ]]).T,
                  array([[0.0   , 42000.0,	30600.0, 0.0, 34.3 , 47.8],
                         [0.0   , 42000.0,	30600.0, 0.0, 34.3 , 47.8]]).T,
                         name = 'INP', type = ['CARB', 'SRB'], OD = [1750.0, 1220.0], 
                         ID = [1250.0, 750.0],B = [375, 365])
    
    print(brg[0])

    