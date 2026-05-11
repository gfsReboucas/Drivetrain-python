# -*- coding: utf-8 -*-
"""Planet carrier geometry."""

from dataclasses import InitVar, dataclass

import numpy as np

from ..components.materials import Material


@dataclass
class Carrier:
    '''
    Implements some dimensions of a planet carrier.
    
    written by:
        Geraldo Rebouças
        - gfs.reboucas@gmail.com
        - https://gfsreboucas.github.io
    '''
    
    aw: InitVar[float]
    bg: InitVar[float]

    def __post_init__(self, aw, bg):
        # main attributes:
        self.a_w = aw # [mm], Center distance
        self.b_g = bg # [mm], Face width
        
        # [mm],    Tip diameter:
        self.d_a  = 2.6*self.a_w
        # [mm],    Root diameter:
        self.d_f  = 1.4*self.a_w
        # [mm],    Witdh:
        self.b    = 1.2*self.b_g
        # [m^3],  Volume:
        self.V    = (np.pi/4.0)*(self.d_a**2 - self.d_f**2)*self.b*1.0e-9
        # [kg],    Mass:
        self.mass = Material().rho*self.V
        # [kg-m**2], Mass moment of inertia, (x axis, rot.):
        self.J_x  = (self.mass/2.0)*(self.d_a**2 + self.d_f**2)*1.0e-6
        # [kg-m**2], Mass moment of inertia, (y axis):
        self.J_y  = (self.mass/2.0)*((3.0/4.0)*(self.d_a**2 + self.d_f**2) + self.b**2)*1.0e-6
        # [kg-m**2], Mass moment of inertia, (z axis):
        self.J_z  = self.J_y
