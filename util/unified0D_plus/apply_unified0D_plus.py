import numpy as np
import pickle
import pdb
from util.unified0D_plus.unified0D_plus import *

def get_angle(direction1, direction2):
    angle = np.arccos(np.sum(np.multiply(direction1/np.linalg.norm(direction1), direction2/np.linalg.norm(direction2)), axis=0))
    return angle

def apply_unified0D_plus(junction_dict):
    rho = 1.06 # density of blood
    num_inlets = np.size(junction_dict["inlet_area"])
    num_outlets = np.size(junction_dict["outlet_area"])

    if np.size(junction_dict["outlet_angle"])==0:
        print("no outlets in junction " + junction_id)
        pdb.set_trace()

    U = np.concatenate([np.abs(junction_dict["inlet_velocity"]), np.multiply(-1, np.abs(junction_dict["outlet_velocity"]))]) # velocity vector for Mynard model
    if junction_dict["inlet_velocity"][0] < 0: # if flow is backwards, invert velocities
        U = -1 * U
    U = U.reshape(-1,)
    A = np.concatenate([junction_dict["inlet_area"], junction_dict["outlet_area"]]).reshape(-1,) # area vector for Mynard model
    theta = np.concatenate([np.asarray([0]).reshape(1,1), np.pi*junction_dict["outlet_angle"]/180]).reshape(-1,)

    theta[0] = np.pi # angle vector for Mynard model
    theta[-1] = -theta[-1]

    try:
        C, K = junction_loss_coeff(U, A, theta) # run JLC function
        dP_mynard = K * rho * U[0]**2
    except:
        print(f"Error in applying Unified0D+ model. U: {U}, A: {A}, theta: {theta}")
        pdb.set_trace()

    return dP_mynard
