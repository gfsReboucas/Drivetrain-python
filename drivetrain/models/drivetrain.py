# -*- coding: utf-8 -*-
"""Core drivetrain model."""

from abc import ABC, abstractmethod
from json import dumps

import numpy as np

from ..components.configs import DrivetrainConfig
from ..dynamic_formulation import torsional_2DOF
from ..iso6336 import ISO_6336


class Drivetrain(ABC):
    '''
    '''
    @property
    @abstractmethod
    def reference_model(self):
        """Name of the concrete drivetrain model."""

    def __init__(self, config: DrivetrainConfig | None = None, **kwargs):
        if config is not None:
            if config.stage is not None:
                kwargs.setdefault("stage", config.stage)
            if config.main_shaft is not None:
                kwargs.setdefault("main_shaft", config.main_shaft)
            kwargs.setdefault("N_st", config.N_st)
            kwargs.setdefault("P_rated", config.P_rated)
            kwargs.setdefault("n_rated", config.n_rated)
            kwargs.setdefault("m_Rotor", config.m_Rotor)
            kwargs.setdefault("J_Rotor", config.J_Rotor)
            kwargs.setdefault("m_Gen", config.m_Gen)
            kwargs.setdefault("J_Gen", config.J_Gen)
            kwargs.setdefault("dynamic_model", config.dynamic_model)
        missing = [name for name in ("stage", "main_shaft") if name not in kwargs]
        if missing:
            raise ValueError("Drivetrain requires explicit {}".format(", ".join(missing)))

        # [-],      Number of stages:
        self.N_st        = kwargs['N_st']       if('N_st'          in kwargs) else len(kwargs['stage'])
        # [-],      gearbox stages:
        self.stage       = kwargs['stage']
        #[kW],     Rated power:
        self.P_rated     = kwargs['P_rated']    if('P_rated'       in kwargs) else 5.0e3
        #[1/min.], Rated input speed
        self.n_rated     = kwargs['n_rated']    if('n_rated'       in kwargs) else 12.1
        #[-],      Input Shaft:
        self.main_shaft  = kwargs['main_shaft']
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
        SH = np.array(0)
        for i in range(self.N_st):
            iso = ISO_6336(self.stage[i])
            SHi = iso.Pitting(P   = self.P_rated,
                              n_1 = self.n_out[i])
            SH = np.hstack((SH, SHi))

        self.S_H = np.array(SH[1:])

        # Dynamic model
        DM = self.dynamic_model(self)
        self.f_n = DM.f_n
        self.mode_shape = DM.mode_shape
    
    def __repr__(self):
        
        conf = [self.stage[i].configuration for i in range(self.N_st)]
        ring = [i for i, v in enumerate(conf) if(v == 'planetary')]
        
        with np.printoptions(precision = 3):
            val = ('Rated power,                                  P       = {} kW\n'.format(self.P_rated) +
                   'Output Speed (Sun/Pinion),                    n_out   = {} 1/min.\n'.format(self.n_out) +
                   'Output Torque (Sun/Pinion),                   T_out   = {} N-m\n'.format(self.T_out) +
                   'Minimum safety factor against pitting,        S_Hmin  = {} -\n'.format(1.25) +
                   'Safety factor against pitting (Sun/Pinion),   S_H1    = {} -\n'.format('TODO') +
                   'Safety factor against pitting (Planet/Wheel), S_H2    = {} -\n'.format('todo') +
                   'Safety factor (Shaft),                        S       = {} -\n'.format('todo') +
                   'Type,                                         -       = {} -\n'.format(conf) +
                   'Gear ratio,                                   u       = {} -\n'.format(     np.array([self.stage[i].u       for i in range(self.N_st)])) +
                   'Number of planets,                            p       = {} -\n'.format(     np.array([self.stage[i].N_p     for i in range(self.N_st)])) +
                   'Normal module,                                m_n     = {} mm\n'.format(    np.array([self.stage[i].m_n     for i in range(self.N_st)])) +
                   'Normal pressure angle,                        alpha_n = {} deg.\n'.format(  np.array([self.stage[i].alpha_n for i in range(self.N_st)])) +
                   'Helix angle,                                  beta    = {} deg.\n'.format(  np.array([self.stage[i].beta    for i in range(self.N_st)])) +
                   'Face width,                                   b       = {} mm\n'.format(    np.array([self.stage[i].b       for i in range(self.N_st)])) +
                   'Center distance,                              a_w     = {} mm\n'.format(    np.array([self.stage[i].a_w     for i in range(self.N_st)])) +
                   'Number of teeth (Sun/Pinion),                 z_1     = {} -\n'.format(     np.array([self.stage[i].z[0]    for i in range(self.N_st)])) +
                   'Number of teeth (Planet/Wheel),               z_2     = {} -\n'.format(     np.array([self.stage[i].z[1]    for i in range(self.N_st)])) +
                   'Number of teeth (Ring),                       z_3     = {} -\n'.format(     np.array([self.stage[i].z[2]    for i in ring])) +
                   'Profile shift coefficient (Sun/Pinion),       x_1     = {} -\n'.format(     np.array([self.stage[i].x[0]    for i in range(self.N_st)])) +
                   'Profile shift coefficient (Planet/Wheel),     x_2     = {} -\n'.format(     np.array([self.stage[i].x[1]    for i in range(self.N_st)])) +
                   'Profile shift coefficient (Ring),             x_3     = {} -\n'.format(     np.array([self.stage[i].x[2]    for i in ring])) +
                   'Reference diameter (Sun/Pinion),              d_1     = {} mm\n'.format(    np.array([self.stage[i].d[0]    for i in range(self.N_st)])) +
                   'Reference diameter (Planet/Wheel),            d_2     = {} mm\n'.format(    np.array([self.stage[i].d[1]    for i in range(self.N_st)])) +
                   'Reference diameter (Ring),                    d_3     = {} mm\n'.format(    np.array([self.stage[i].d[2]    for i in ring])) +
                   'Mass (Sun/Pinion),                            m_1     = {} kg\n'.format(    np.array([self.stage[i].mass[0] for i in range(self.N_st)])) +
                   'Mass (Planet/Wheel),                          m_2     = {} kg\n'.format(    np.array([self.stage[i].mass[1] for i in range(self.N_st)])) +
                   'Mass (Ring),                                  m_3     = {} kg\n'.format(    np.array([self.stage[i].mass[2] for i in ring])) +
                   'Mass mom. inertia (Sun/Pinion),               J_xx1   = {} kg-m**2\n'.format(np.array([self.stage[i].J_x[0]  for i in range(self.N_st)])) +
                   'Mass mom. inertia (Planet/Wheel),             J_xx2   = {} kg-m**2\n'.format(np.array([self.stage[i].J_x[1]  for i in range(self.N_st)])) +
                   'Mass mom. inertia (Ring),                     J_xx3   = {} kg-m**2\n'.format(np.array([self.stage[i].J_x[2]  for i in ring])))
               # 'Mass mom. inertia (Sun/Pinion),               J_yy1   = {} kg-m**2' +
               # 'Mass mom. inertia (Planet/Wheel),             J_yy2   = {} kg-m**2' +
               # 'Mass mom. inertia (Ring),                     J_yy3   = {} kg-m**2' +
               # 'Mass mom. inertia (Sun/Pinion),               J_zz1   = {} kg-m**2' +
               # 'Mass mom. inertia (Planet/Wheel),             J_zz2   = {} kg-m**2' +
               # 'Mass mom. inertia (Ring),                     J_zz3   = {} kg-m**2' +)
               
        return val
    
    def __gear_ratio(self):
        return np.cumprod([self.stage[idx].u for idx in range(self.N_st)])
    
    def __output_speed(self):
        return self.n_rated*self.u
    
    def __output_torque(self):
        return (self.P_rated*1.0e3)/(self.n_out*np.pi/30.0)
    
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

        return np.hstack((diffSH, difffn[1:n]))

    def save(self):
        pass
        
    def toJson(self):
        return dumps(self, default = lambda x: x.__dict__)
