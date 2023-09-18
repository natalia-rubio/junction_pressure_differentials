from knn import *
from dec_tree import *
from lin_reg import *
from svr import *
from gpr import *

anatomy = "Aorta_rand"; num_geos_steady = 110; num_geos_unsteady = 102

print ("{:<8} {:<10} {:<10} {:<10} {:<10}".format('Model','Steady (Train)','Steady (Val)','Unsteady (Train)', 'Unsteady (Val)'))

reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_knn_model_steady(anatomy, num_geos_steady, seed = 0)
reg_unsteady, dP_loss_val_unsteady, dP_loss_train_unsteady = train_knn_model_unsteady(anatomy, num_geos_unsteady, seed = 0)
print ("{:<8} {:<10} {:<10} {:<10} {:<10}".format('KNN',dP_loss_train_steady, dP_loss_val_steady, dP_loss_train_unsteady, dP_loss_val_unsteady))


reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_dt_model_steady(anatomy, num_geos_steady, seed = 0)
reg_unsteady, dP_loss_val_unsteady, dP_loss_train_unsteady = train_dt_model_unsteady(anatomy, num_geos_unsteady, seed = 0)
print ("{:<8} {:<10} {:<10} {:<10} {:<10}".format('DT',dP_loss_train_steady, dP_loss_val_steady, dP_loss_train_unsteady, dP_loss_val_unsteady))

reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_lin_reg_model_steady(anatomy, num_geos_steady, seed = 0)
reg_unsteady, dP_loss_val_unsteady, dP_loss_train_unsteady = train_lin_reg_model_unsteady(anatomy, num_geos_unsteady, seed = 0)
print ("{:<8} {:<10} {:<10} {:<10} {:<10}".format('LR',dP_loss_train_steady, dP_loss_val_steady, dP_loss_train_unsteady, dP_loss_val_unsteady))

reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_svr_model_steady(anatomy, num_geos_steady, seed = 0)
reg_unsteady, dP_loss_val_unsteady, dP_loss_train_unsteady = train_svr_model_unsteady(anatomy, num_geos_unsteady, seed = 0)
print ("{:<8} {:<10} {:<10} {:<10} {:<10}".format('SVR',dP_loss_train_steady, dP_loss_val_steady, dP_loss_train_unsteady, dP_loss_val_unsteady))

reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_gpr_model_steady(anatomy, num_geos_steady, seed = 0)
reg_unsteady, dP_loss_val_unsteady, dP_loss_train_unsteady = train_gpr_model_unsteady(anatomy, num_geos_unsteady, seed = 0)
print ("{:<8} {:<10} {:<10} {:<10} {:<10}".format('GPR',dP_loss_train_steady, dP_loss_val_steady, dP_loss_train_unsteady, dP_loss_val_unsteady))
