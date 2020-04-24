# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:03:51 2020

@author: geraldod
"""

from Drivetrain import NREL_5MW
from dynamic_formulation import Kahraman_94, torsional_2DOF
from matplotlib import pyplot

if(__name__ == '__main__'):
    ref = NREL_5MW()
    # stage = ref.stage[0]
    # stage.sub_set('planet-ring')
    MS = ref.mode_shape
    gm = ref.gamma

    gm_P = 1/2

    for key, value in ref.gamma.items():
        # do something with value
        gm[key] = ref.gamma[key]*gm_P**(1/3)
        
    val = ref.min_func(gamma_P = gm_P,
                       gamma   = gm)

    print(val)

    # for i in range(14):
    #     pyplot.plot(range(14), MS[:, i])



    