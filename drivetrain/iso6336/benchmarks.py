# -*- coding: utf-8 -*-
"""ISO 6336 benchmark cases."""

import numpy as np

from ..gears import GearSet
from .calculator import ISO_6336


def benchmark():
    
    gset = GearSet.example_01_ISO6336()
    
    T1 = 9.0e3 # N-m
    n1 = 360.0 # 1/min.
    P1 = T1*n1*(np.pi/30.0) # W
    P1 = P1*1.0e-3 # kW
    
    example_01 = ISO_6336(gset,
                  K_A    = 1.0,
                  L_h    = 50.0e3,
                  S_Hmin = 1.0,
                  S_Fmin = 1.0)
    
    SH = example_01.Pitting(P     = P1,
                n_1   = n1,
                R_a   = 1.0,
                nu_40 = 320.0,
                line  = 2,
                C_a   = 70.0) # um
    
    print(SH)
    

