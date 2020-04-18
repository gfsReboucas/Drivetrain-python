# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:03:51 2020

@author: geraldod
"""

from Drivetrain import NREL_5MW
from dynamic_model import Kahraman_94, torsional_2DOF

if(__name__ == '__main__'):
    # ref = NREL_5MW()
    # stage = ref.stage[0]
    # stage.sub_set('planet-ring')
    dm = torsional_2DOF(NREL_5MW())
    