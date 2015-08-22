"""
This file defines useful correspondance dictionaries
which are used in the openPMD writer
"""
import numpy as np

# List of dictionaries to import when doing 'from data_dict import *'
__all__ = [ 'unit_dimension_dict', 'circ_dict_quantity', 'cart_dict_quantity',
            'circ_dict_Jindex', 'cart_dict_Jindex', 'field_boundary_dict',
            'particle_boundary_dict', 'field_solver_dict',
            'macro_weighted_dict', 'weighting_power_dict' ]

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

# Typical weighting of different particle properties
macro_weighted_dict = {
    "charge": np.uint32(0),
    "mass": np.uint32(0),
    "weighting": np.uint32(1),
    "position": np.uint32(0),
    "positionOffset": np.uint32(0),
    "momentum" : np.uint32(0) }
weighting_power_dict = {
    "charge": 1.,
    "mass": 1.,
    "weighting": 1.,
    "position": 0.,
    "positionOffset": 0.,
    "momentum": 1. }

# Correspondance between the names in OpenPMD and the names in Warp
circ_dict_quantity = { 'rho':'Rho', 'Er':'Exp', 'Et':'Eyp', 'Ez':'Ezp', 
                        'Br':'Bxp', 'Bt':'Byp', 'Bz':'Bzp' }
cart_dict_quantity = { 'rho':'Rho', 'Ex':'Exp', 'Ey':'Eyp', 'Ez':'Ezp', 
                        'Bx':'Bxp', 'By':'Byp', 'Bz':'Bzp' }
circ_dict_Jindex = { 'Jr':0, 'Jt':1, 'Jz':2 }
cart_dict_Jindex = { 'Jx':0, 'Jy':1, 'Jz':2 }

# Correspondance between the boundary conditions in Warp,
# and the corresponding representative integer
field_boundary_dict = {
    0: "reflecting",
    1: "reflecting",
    2: "periodic",
    3: "openbc" }
particle_boundary_dict = {
    0: "absorbing",
    1: "reflecting",
    2: "periodic" }
# Correspondance between the field solver in Warp,
# and the corresponding representative integer
field_solver_dict = {
    0: "Yee",
    1: "CK",
    2: "Lehe" }
