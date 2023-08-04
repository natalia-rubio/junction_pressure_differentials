import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.regression.lin_reg import *
from util.regression.svr import *
from util.regression.gpr import *

anatomy = "Aorta_u_40-60_over3"
reg, lin_reg_dP_loss_val, lin_reg_dP_loss_train = train_lin_reg_model_steady(anatomy = anatomy, num_geos = 360)
print(f"Linear Regression RMSE (STEADY): {lin_reg_dP_loss_val} ")

reg, svr_dP_loss_val, svr_dP_loss_train = train_svr_model_steady(anatomy = anatomy, num_geos = 360)
print(f"SVR Regression RMSE (STEADY): {svr_dP_loss_val} ")

reg, gpr_dP_loss_val, gpr_dP_loss_train = train_gpr_model_steady(anatomy = anatomy, num_geos = 360)
print(f"GPR Regression RMSE (STEADY): {gpr_dP_loss_val} ")
