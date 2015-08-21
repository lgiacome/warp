"""
This file defines useful correspondance dictionaries
which are used in the openPMD writer
"""
import numpy as np

# Correspondance between quantity and corresponding dimensions
# As specified in the openPMD standard, the arrays represent the
# 7 basis dimensions L, M, T, I, theta, N, J
unit_dimension_dict = {
    "rho" : np.array([-3., 0., 1., 1., 0., 0., 0.]),
    "J" : np.array([-3., 1., 0., 1., 0., 0., 0.]),
    "E" : np.array([ 1., 1.,-3.,-1., 0., 0., 0.]),
    "B" : np.array([ 0., 1.,-2.,-1., 0., 0., 0.]),
    "charge" : np.array([0., 0., 1., 1., 0., 0., 0.]),
    "mass" : np.array([1., 0., 0., 0., 0., 0., 0.]),
    "weighting" : np.array([0., 0., 0., 0., 0., 0., 0.]),
    "position" : np.array([1., 0., 0., 0., 0., 0., 0.]),
    "positionOffset" : np.array([1., 0., 0., 0., 0., 0., 0.]),
    "momentum" : np.array([1., 1.,-1., 0., 0., 0., 0.]) }

# Correspondance between the names in OpenPMD and the names in Warp
circ_dict_quantity = { 'rho':'Rho', 'Er':'Exp', 'Et':'Eyp', 'Ez':'Ezp', 
                        'Br':'Bxp', 'Bt':'Byp', 'Bz':'Bzp' }
cart_dict_quantity = { 'rho':'Rho', 'Ex':'Exp', 'Ey':'Eyp', 'Ez':'Ezp', 
                        'Bx':'Bxp', 'By':'Byp', 'Bz':'Bzp' }
circ_dict_Jindex = { 'Jr':0, 'Jt':1, 'Jz':2 }
cart_dict_Jindex = { 'Jx':0, 'Jy':1, 'Jz':2 }
