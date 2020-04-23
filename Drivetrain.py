# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:01:45 2020

@author: geraldod
"""
from json import dumps
from scipy import array#, zeros
from numpy import pi, cumprod, printoptions, iscomplex, empty, hstack
from components import Bearing, Shaft #, check_key
from Gear import GearSet
from dynamic_model import torsional_2DOF, Kahraman_94
from ISO_6336 import ISO_6336

class Drivetrain:
    '''
    '''
    def __init__(self, **kwargs):
        # [-],      Number of stages:
        self.N_st        = kwargs['N_st']       if('N_st'          in kwargs) else 3
        # [-],      gearbox stages:
        self.stage       = kwargs['stage']      if('stage'         in kwargs) else [NREL_5MW.gear_set(idx) for idx in range(3)]
        #[kW],     Rated power:
        self.P_rated     = kwargs['P_rated']    if('P_rated'       in kwargs) else 5.0e3
        #[1/min.], Rated input speed
        self.n_rated     = kwargs['n_rated']    if('n_rated'       in kwargs) else 12.1
        #[-],      Input Shaft:
        self.main_shaft  = kwargs['main_shaft'] if('main_shaft'   in kwargs) else NREL_5MW.shaft(-1)
        # [kg],     Rotor mass:
        self.m_Rotor     = kwargs['m_Rotor']    if('m_Rotor'       in kwargs) else 110.0e3
        # [kg-m**2], Rotor mass moment of inertia:
        self.J_Rotor     = kwargs['J_Rotor']    if('J_Rotor'       in kwargs) else 57231535.0
        # [kg],     Generator mass:
        self.m_Gen       = kwargs['m_Gen']      if('m_Gen'         in kwargs) else 1900.0
        # [kg-m**2], Generator mass moment of inertia
        self.J_Gen       = kwargs['J_Gen']      if('J_Gen'         in kwargs) else 534.116
        
        # specify which dynamic model should be used to perform modal analysis 
        # on the Drivetrain:
        self.dynamic_model = kwargs['dynamic_model'] if('dynamic_model' in kwargs) else torsional_2DOF
        
        # [-],      Cumulative gear ratio at each stage:
        self.u     = self.__gear_ratio()
        # [1/min.], Output speed of each stage:
        self.n_out = self.__output_speed()
        # [N-m],    Output torque of each stage:
        self.T_out = self.__output_torque()
                # Stage 1:
        name = ['m_n1', 'b_1', 'd_1', 'L_1',
                # Stage 2:
                'm_n2', 'b_2', 'd_2', 'L_2',
                # Stage 3:
                'm_n3', 'b_3', 'd_3', 'L_3',
        # M. M. Inertia: rotor, generator
                'J_R' , 'J_G', # Main shaft:
                               'd_s' , 'L_s']
        
        # [-], Scaling factors:
        self.gamma = dict.fromkeys(name, 1.0)

        # Safety factors for gear sets:
        SH = array(0)
        for i in range(self.N_st):
            iso = ISO_6336(self.stage[i])
            SHi = iso.Pitting(P   = self.P_rated,
                              n_1 = self.n_out[i])
            SH = hstack((SH, SHi))

        self.S_H = array(SH[1:])

        # Dynamic model
        DM = self.dynamic_model(self)
        self.f_n = DM.f_n
        self.mode_shape = DM.mode_shape
    
    def __repr__(self):
        
        conf = [self.stage[i].configuration for i in range(self.N_st)]
        ring = [i for i, v in enumerate(conf) if(v == 'planetary')]
        
        with printoptions(precision = 3):
            val = ('Rated power,                                  P       = {} kW\n'.format(self.P_rated) +
                   'Output Speed (Sun/Pinion),                    n_out   = {} 1/min.\n'.format(self.n_out) +
                   'Output Torque (Sun/Pinion),                   T_out   = {} N-m\n'.format(self.T_out) +
                   'Minimum safety factor against pitting,        S_Hmin  = {} -\n'.format(1.25) +
                   'Safety factor against pitting (Sun/Pinion),   S_H1    = {} -\n'.format('TODO') +
                   'Safety factor against pitting (Planet/Wheel), S_H2    = {} -\n'.format('todo') +
                   'Safety factor (Shaft),                        S       = {} -\n'.format('todo') +
                   'Type,                                         -       = {} -\n'.format(conf) +
                   'Gear ratio,                                   u       = {} -\n'.format(     array([self.stage[i].u       for i in range(self.N_st)])) +
                   'Number of planets,                            p       = {} -\n'.format(     array([self.stage[i].N_p     for i in range(self.N_st)])) +
                   'Normal module,                                m_n     = {} mm\n'.format(    array([self.stage[i].m_n     for i in range(self.N_st)])) +
                   'Normal pressure angle,                        alpha_n = {} deg.\n'.format(  array([self.stage[i].alpha_n for i in range(self.N_st)])) +
                   'Helix angle,                                  beta    = {} deg.\n'.format(  array([self.stage[i].beta    for i in range(self.N_st)])) +
                   'Face width,                                   b       = {} mm\n'.format(    array([self.stage[i].b       for i in range(self.N_st)])) +
                   'Center distance,                              a_w     = {} mm\n'.format(    array([self.stage[i].a_w     for i in range(self.N_st)])) +
                   'Number of teeth (Sun/Pinion),                 z_1     = {} -\n'.format(     array([self.stage[i].z[0]    for i in range(self.N_st)])) +
                   'Number of teeth (Planet/Wheel),               z_2     = {} -\n'.format(     array([self.stage[i].z[1]    for i in range(self.N_st)])) +
                   'Number of teeth (Ring),                       z_3     = {} -\n'.format(     array([self.stage[i].z[2]    for i in ring])) +
                   'Profile shift coefficient (Sun/Pinion),       x_1     = {} -\n'.format(     array([self.stage[i].x[0]    for i in range(self.N_st)])) +
                   'Profile shift coefficient (Planet/Wheel),     x_2     = {} -\n'.format(     array([self.stage[i].x[1]    for i in range(self.N_st)])) +
                   'Profile shift coefficient (Ring),             x_3     = {} -\n'.format(     array([self.stage[i].x[2]    for i in ring])) +
                   'Reference diameter (Sun/Pinion),              d_1     = {} mm\n'.format(    array([self.stage[i].d[0]    for i in range(self.N_st)])) +
                   'Reference diameter (Planet/Wheel),            d_2     = {} mm\n'.format(    array([self.stage[i].d[1]    for i in range(self.N_st)])) +
                   'Reference diameter (Ring),                    d_3     = {} mm\n'.format(    array([self.stage[i].d[2]    for i in ring])) +
                   'Mass (Sun/Pinion),                            m_1     = {} kg\n'.format(    array([self.stage[i].mass[0] for i in range(self.N_st)])) +
                   'Mass (Planet/Wheel),                          m_2     = {} kg\n'.format(    array([self.stage[i].mass[1] for i in range(self.N_st)])) +
                   'Mass (Ring),                                  m_3     = {} kg\n'.format(    array([self.stage[i].mass[2] for i in ring])) +
                   'Mass mom. inertia (Sun/Pinion),               J_xx1   = {} kg-m**2\n'.format(array([self.stage[i].J_x[0]  for i in range(self.N_st)])) +
                   'Mass mom. inertia (Planet/Wheel),             J_xx2   = {} kg-m**2\n'.format(array([self.stage[i].J_x[1]  for i in range(self.N_st)])) +
                   'Mass mom. inertia (Ring),                     J_xx3   = {} kg-m**2\n'.format(array([self.stage[i].J_x[2]  for i in ring])))
               # 'Mass mom. inertia (Sun/Pinion),               J_yy1   = {} kg-m**2' +
               # 'Mass mom. inertia (Planet/Wheel),             J_yy2   = {} kg-m**2' +
               # 'Mass mom. inertia (Ring),                     J_yy3   = {} kg-m**2' +
               # 'Mass mom. inertia (Sun/Pinion),               J_zz1   = {} kg-m**2' +
               # 'Mass mom. inertia (Planet/Wheel),             J_zz2   = {} kg-m**2' +
               # 'Mass mom. inertia (Ring),                     J_zz3   = {} kg-m**2' +)
               
        return val
    
    def __gear_ratio(self):
        return cumprod([self.stage[idx].u for idx in range(self.N_st)])
    
    def __output_speed(self):
        return self.n_rated*self.u
    
    def __output_torque(self):
        return (self.P_rated*1.0e3)/(self.n_out*pi/30.0)
    
    def min_func(self,**kwargs):
        gamma_P = kwargs['gamma_P'] if('gamma_P' in kwargs) else 1
        gamma_n = kwargs['gamma_n'] if('gamma_n' in kwargs) else 1
        gamma   = kwargs['gamma']   if('gamma'   in kwargs) else self.gamma
        SHref   = kwargs['SHref']   if('SHref'   in kwargs) else self.S_H
        n       = kwargs['n']       if('n'       in kwargs) else None
        
        fnref   = kwargs['fnref']   if('fnref'   in kwargs) else self.f_n
        fnref  *= 1/fnref[0]

        scale_DT = self
        scale_DT.__init__(gamma_P = gamma_P,
                          gamma_n = gamma_n,
                          gamma   = gamma)

        SH  = scale_DT.S_H
        fn  = scale_DT.f_n
        fn *= 1/fn[0]

        diffSH = SHref - SH
        difffn = 1 - fn/fnref

        return hstack((diffSH, difffn[1:n]))

    def save(self):
        pass
        
    def toJson(self):
        return dumps(self, default = lambda x: x.__dict__)

###############################################################################

class NREL_5MW(Drivetrain):
    '''
    This class contains some of the properties of the NREL 5MW wind turbine 
    gearbox proposed by Nejad et. al. [1]. More information about the 
    NREL 5MW wind turbine can be found at [2].
    
    References:
        [1] Nejad, A. R., Guo, Y., Gao, Z., Moan, T. (2016). Development of a
        5 MW reference gearbox for offshore wind turbines. Wind Energy. 
        https://doi.org/10.1002/we.1884
        
        [2] Jonkman, J., Butterfield, S., Musial, W., & Scott, G. Definition
        of a 5-MW Reference Wind Turbine for Offshore System Development. 
        https://doi.org/10.2172/947422
        
        [3] Anaya-Lara, O., Tande, J.O., Uhlen, K., Merz, K. and Nejad, A.R.
        (2019). Modelling and Analysis of Drivetrains in Offshore Wind
        Turbines. In Offshore Wind Energy Technology (eds O. Anaya-Lara, J.O.
        Tande, K. Uhlen and K. Merz). 
        https://doi.org/10.1002/9781119097808.ch3
    '''
    
    def __init__(self, **kwargs):
        name = ['m_n1', 'b_1', 'd_1', 'L_1',  # Stage 01
                'm_n2', 'b_2', 'd_2', 'L_2',  # Stage 02
                'm_n3', 'b_3', 'd_3', 'L_3',  # Stage 03
                'm_R' , 'J_R' ,               # Rotor: mass and M. M. Inertia
                'm_G' , 'J_G' ,               # Generator: mass and M. M. Inertia
                               'd_s' , 'L_s'] # Main shaft
        
        # [-], Scaling factors:
        self.gamma = kwargs['gamma']     if('gamma'     in kwargs) else dict.fromkeys(name, 1.0)
        # [-], Rated power scaling factor:
        gamma_P    = kwargs['gamma_P']   if('gamma_P'   in kwargs) else 1.0
        # [-], Rated rotor speed scaling factor:
        gamma_n    = kwargs['gamma_n']   if('gamma_n'   in kwargs) else 1.0
        # [-], Dynamic model:
        dyn_mod    = kwargs['dynamic_model'] if('dynamic_model' in kwargs) else Kahraman_94
        
        p_r = 5.0e3*gamma_P
        n_r = 12.1*gamma_n
        
        m_R = 110.0e3
        m_R *= self.gamma['m_R'] if('m_R' in self.gamma) else 1.0
        J_R = 57231535.0
        J_R *= self.gamma['J_R'] if('J_R' in self.gamma) else 1.0
        m_G = 1900.0
        m_G *= self.gamma['m_G'] if('m_G' in self.gamma) else 1.0
        J_G = 534.116
        J_G *= self.gamma['J_G'] if('J_G' in self.gamma) else 1.0

        stage = [None]*3
        
        for idx in range(3):
            gm_idx = dict(filter(lambda item: str(idx + 1) in item[0], self.gamma.items()))
            stage[idx] = NREL_5MW.gear_set(idx).apply_lambda(gm_idx)
        
        inp_shaft = NREL_5MW.shaft(-1)
        
        d_s = inp_shaft.d
        d_s *= self.gamma['d_s'] if('d_s' in self.gamma) else 1.0
        L_s = inp_shaft.L
        L_s *= self.gamma['L_s'] if('L_s' in self.gamma) else 1.0
        
        super().__init__(P_rated = p_r,
                         n_rated = n_r,
                         stage = stage,
                         main_shaft = Shaft(d_s, L_s),
                         m_Rotor = m_R,
                         J_Rotor = J_R,
                         m_Gen = m_G,
                         J_Gen = J_G,
                         N_st = 3,
                         dynamic_model = dyn_mod)

    def save(self, filename):
        pass
            # file = open(filename, 'w')
            # dump({'P_rated': self.P_rated}, file)
            # dump({'n_rated': self.n_rated}, file)
            # dump(self.gamma,   file)
        
    @staticmethod
    def gear_set(idx):
        '''
        returns the gear stages of the NREL 5 MW wind turbine drivetrain 
        according to [4], Table V. The values for the tip alteration 
        coefficients k, were taken from KISSsoft [1] reports.
        
        References
        ----------
        [1] https://www.kisssoft.ch/

        Parameters
        ----------
        idx : int
            DESCRIPTION.

        Raises
        ------
        Exception
            if idx is out of range.

        Returns
        -------
        GearSet
            the idx-th gear stage.

        '''
        
        alphaN = 20.0  # [deg.], Pressure angle (at reference cylinder)
        rack_type = 'A' # [-],    Type of the basic rack from A to D
        
        if(idx == 0):
            conf = 'planetary' # [-],    GearSet configuration
            Np   =   3         # [-],    Number of planet gears
            mn   =  45.0       # [mm],   Normal module
            beta =   0.0       # [deg.], Helix angle (at reference cylinder)
            bb   = 491.0       # [mm],   Face width
            aw   = 863.0       # [mm],   Center distance
            z_s  =  19         # [-],    Number of teeth (sun)
            z_p  =  17         # [-],    Number of teeth (planet)
            z_r  =  56         # [-],    Number of teeth (ring)
            x_s  =   0.617     # [-],    Profile shift coefficient (sun)
            x_p  =   0.802     # [-],    Profile shift coefficient (planet)
            x_r  =  -0.501     # [-],    Profile shift coefficient (ring)
            k_s  = -10.861/mn  # [-],    Tip alteration coefficient (sun)
            k_p  = -10.861/mn  # [-],    Tip alteration coefficient (planet)
            k_r  =   0.0       # [-],    Tip alteration coefficient (ring)
            
            bore_Rs = 80.0/171.0
            bore_Rp = 80.0/153.0
            bore_Rr = 1.2
            
            zz = array([z_s, z_p, z_r])
            xx = array([x_s, x_p, x_r])
            kk = array([k_s, k_p, k_r])
            bore_R = array([bore_Rs, bore_Rp, bore_Rr])
            
        elif(idx == 1):
            conf = 'planetary' # [-],    GearSet configuration
            Np   =   3         # [-],    Number of planet gears
            mn   =  21.0       # [mm],   Normal module
            beta =   0.0       # [deg.], Helix angle (at reference cylinder)
            bb   = 550.0       # [mm],   Face width
            aw   = 584.0       # [mm],   Center distance
            z_s  =  18         # [-],    Number of teeth (sun)
            z_p  =  36         # [-],    Number of teeth (planet)
            z_r  =  93         # [-],    Number of teeth (ring)
            x_s  =   0.389     # [-],    Profile shift coefficient (sun)
            x_p  =   0.504     # [-],    Profile shift coefficient (planet)
            x_r  =   0.117     # [-],    Profile shift coefficient (ring)
            k_s  =  -1.75/mn   # [-],    Tip alteration coefficient (sun)
            k_p  =  -1.75/mn   # [-],    Tip alteration coefficient (planet)
            k_r  =   0.0       # [-],    Tip alteration coefficient (ring)
            
            bore_Rs = 100.0/189.0
            bore_Rp = 95.0/189.0
            bore_Rr = 1.2

            zz = array([z_s, z_p, z_r])
            xx = array([x_s, x_p, x_r])
            kk = array([k_s, k_p, k_r])
            bore_R = array([bore_Rs, bore_Rp, bore_Rr])
            
        elif(idx == 2):
            conf = 'parallel' # [-],    GearSet configuration
            Np   =   1        # [-],    Number of planet gears
            mn   =  14.0      # [mm],   Normal module
            beta =  10.0      # [deg.], Helix angle (at reference cylinder)
            bb   = 360.0      # [mm],   Face width
            aw   = 861.0      # [mm],   Center distance
            z_1  =  24        # [-],    Number of teeth (pinion)
            z_2  =  95        # [-],    Number of teeth (wheel)
            x_1  =   0.480    # [-],    Profile shift coefficient (pinion)
            x_2  =   0.669    # [-],    Profile shift coefficient (wheel)
            k_1  =  -0.938/mn # [-],    Tip alteration coefficient (pinion)
            k_2  =  -0.938/mn # [-],    Tip alteration coefficient (wheel)
            
            bore_R1 = 1809.0/3086.0
            bore_R2 = 3385.0/9143.0

            zz = array([z_1, z_2])
            xx = array([x_1, x_2])
            kk = array([k_1, k_2])
            bore_R = array([bore_R1, bore_R2])
            
        else:
            raise Exception('Option [{}] is NOT valid.'.format(idx))
        
        bearing = NREL_5MW.bearing(idx)
        shaft = NREL_5MW.shaft(idx)
        
        return GearSet(configuration = conf,
                       m_n = mn,
                       alpha_n = alphaN,
                       z = zz,
                       b = bb,
                       x = xx,
                       beta = beta,
                       k = kk,
                       bore_ratio = bore_R,
                       N_p = Np,
                       a_w = aw,
                       rack_type = rack_type,
                       bearing = bearing,
                       shaft = shaft)

    @staticmethod
    def bearing(idx):
        
        damping = array([453.0 , 42000.0,	30600.0, 0.0, 34.3 , 47.8 ])
        
        if(idx == -1): # main shaft
            #                          K_x,    K_y,     K_z,     K_a, K_b,   K_g
            INP_A     = Bearing(array([0.0   , 1.50e10, 1.50e10, 0.0, 5.0e6, 5.0e6]),
                                # damping:
                                array([0.0   , 42000.0,	30600.0, 0.0, 34.3 , 47.8 ]),
                                name = 'INP_A', type = 'CARB', OD = 1750.0, ID = 1250.0, B = 375.0)
            INP_B     = Bearing(array([4.06e8, 1.54e10, 1.54e10, 0.0, 0.0  , 0.0  ]),
                                damping,
                                name = 'INP_B', type = 'SRB', OD = 1220.0, ID = 750.0, B = 365.0)
        
            brg = [INP_A, INP_B]
        elif(idx == 0): # stage 1
                               #       K_x,    K_y,     K_z,     K_a, K_b,   K_g,   OD,     ID,     B
            PL_A      = Bearing(array([9.1e4,  9.4e9,   3.2e9,   0.0, 1.4e6, 4.5e6]),
                                damping,
                                name = 'PL_A', type = 'CRB', OD = 600.0, ID = 400.0, B = 272.0)
            PL_B      = Bearing(array([9.1e4,  9.4e9,   3.2e9,   0.0, 1.4e6, 4.5e6]),
                                damping,
                                name = 'PL_B', type = 'CRB', OD = 600.0, ID = 400.0, B = 272.0)
            PLC_A     = Bearing(array([6.6e4,  1.7e9,   1.1e9,   0.0, 5.6e5, 1.3e5]),
                                damping,
                                name = 'PLC_A', type = 'SRB', OD = 1030.0, ID = 710.0, B = 315.0)
            PLC_B     = Bearing(array([6.6e7,  1.7e9,   1.1e9,   0.0, 5.6e5, 1.3e5]),
                                damping,
                                name = 'PLC_B', type = 'CRB', OD = 1220.0, ID = 1000.0, B = 128.0)
                   # Planet
            brg = [PL_A,  PL_B,
                   # Carrier
                   PLC_A, PLC_B]
        elif(idx == 1): # stage 2
                               #       K_x,    K_y,     K_z,     K_a, K_b,   K_g,   OD,     ID,     B
            IMS_PL_A  = Bearing(array([9.1e4,  6.0e7,   1.2e9,   0.0, 7.5e4, 7.5e4]),
                                damping,
                                name = 'IMS_PL_A' , type = 'CRB',  OD = 520.0,  ID = 380.0, B = 140.0)
            IMS_PL_B  = Bearing(array([9.1e4,  6.0e7,   1.2e9,   0.0, 7.5e4, 7.5e4]),
                                name = 'IMS_PL_B' , type = 'CRB',  OD = 520.0,  ID = 380.0, B = 140.0)
            IMS_PLC_A = Bearing(array([9.1e4,  6.0e7,   1.2e9,   0.0, 7.5e4, 7.5e4]),
                                name = 'IMS_PLC_A', type = 'CARB', OD = 1030.0, ID = 710.0, B = 236.0)
            IMS_PLC_B = Bearing(array([9.1e7,  6.0e7,   1.2e9,   0.0, 7.5e4, 7.5e4]),
                                name = 'IMS_PLC_B', type = 'CRB' , OD = 870.0, ID = 600.0,  B = 155.0)
            
            brg = [IMS_PL_A,  IMS_PL_B,  # Planet
                   IMS_PLC_A, IMS_PLC_B] # Carrier
        elif(idx == 2): # stage 3
                               #       K_x,    K_y,     K_z,     K_a, K_b,   K_g,   OD,     ID,     B
            IMS_A     = Bearing(array([0.0,    6.0e7,   1.2e9,   0.0, 7.5e4, 7.5e4]),
                                damping,
                                name = 'IMS_A'    , type = 'CRB',  OD = 360.0,  ID = 200.0,   B = 98.0)
            IMS_B     = Bearing(array([7.4e7,  5.0e8,   5.0e8,   0.0, 1.6e6, 1.8e6]),
                                damping,
                                name = 'IMS_B'    , type = 'TRB',  OD = 460.0,  ID = 200.0,  B = 100.0)
            IMS_C     = Bearing(array([7.8e7,  7.4e8,   3.3e8,   0.0, 1.1e6, 2.5e6]),
                                damping,
                                name = 'IMS_C'    , type = 'TRB',  OD = 460.0,  ID = 200.0,  B = 100.0)
            HS_A      = Bearing(array([1.3e8,  8.2e8,   8.2e8,   0.0, 1.7e5, 1.0e6]),
                                damping,
                                name = 'HS_A'     , type = 'CRB',  OD = 500.0,  ID = 400.0,  B = 100.0)
            HS_B      = Bearing(array([6.7e7,  8.0e8,   1.3e8,   0.0, 1.7e5, 1.0e6]),
                                damping,
                                name = 'HS_B'     , type = 'TRB',  OD = 550.0,  ID = 410.0,   B = 86.0)
            HS_C      = Bearing(array([8.0e7,  1.0e9,   7.3e7,   0.0, 1.7e5, 1.0e6]),
                                damping,
                                name = 'HS_C'     , type = 'TRB',  OD = 550.0,  ID = 410.0,   B = 86.0)
            
            brg = [HS_A,  HS_B,  HS_C,  # Pinion
                   IMS_A, IMS_B, IMS_C] # Wheel
        else:
            raise Exception('Option [{}] is NOT valid.'.format(idx))
    
        return brg
    
    @staticmethod
    def shaft(idx):
        if(idx == -1): # Main shaft, Low-speed shaft, LSS
            d = 700.0  # 1000.0
            L = 2000.0 # 3000.0
        elif(idx == 0): # stage 1, Intermediate-speed shaft, ISS
            d = 533.0 # 800.0
            L = 500.0 # 750.0
        elif(idx == 1): # stage 2, High-speed Intermediate-speed shaft, HSIS
            d = 333.0 # 500.0
            L = 666.0 # 1000.0
        elif(idx == 2): # stage 3, High-speed shaft, HSS
            d = 333.0
            L = 1000.0
        else:
            raise Exception('Option [{}] is NOT valid.'.format(idx))
    
        return Shaft(d, L)
        
    @staticmethod
    def property_estimation(idx):
        return 0.0
        
###############################################################################

if(__name__ == '__main__'):
    DT = NREL_5MW()
    
    # with open('mydata.json', 'w') as f:
    #     json.dump(DT.__dict__, f)
    # DT.print()
    print(DT.f_n)