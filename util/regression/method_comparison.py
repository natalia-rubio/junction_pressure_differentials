from knn import *
from dec_tree import *
from lin_reg import *
from svr import *
from gpr import *

import sys

anatomy = sys.argv[1];

if anatomy == "Aorta_rand":
    num_geos_steady = 110; num_geos_unsteady = 110
elif anatomy == "Pulmo_rand":
        num_geos_steady = 127; num_geos_unsteady = 127
elif anatomy == "mynard_rand":
        num_geos_steady = 187; num_geos_unsteady = 127

print ("{:<8} {:<10} {:<10} {:<10} {:<10}".format('Model','Steady (Train)','Steady (Val)','Unsteady (Train)', 'Unsteady (Val)'))

hyperparams = {'n_neighbors': 7}
reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_knn_model_steady(anatomy, num_geos_steady, seed = 0, hyperparams = hyperparams)
reg_unsteady, dP_loss_val_unsteady, dP_loss_train_unsteady = train_knn_model_unsteady(anatomy, num_geos_unsteady, seed = 0, hyperparams = hyperparams)
print ("{:<8} {:<10} {:<10} {:<10} {:<10}".format('KNN',dP_loss_train_steady, dP_loss_val_steady, dP_loss_train_unsteady, dP_loss_val_unsteady))

hyperparams = {'max_depth': 4, 'min_samples_leaf': 8}
reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_dt_model_steady(anatomy, num_geos_steady, seed = 0, hyperparams = hyperparams)
reg_unsteady, dP_loss_val_unsteady, dP_loss_train_unsteady = train_dt_model_unsteady(anatomy, num_geos_unsteady, seed = 0, hyperparams = hyperparams)
print ("{:<8} {:<10} {:<10} {:<10} {:<10}".format('DT',dP_loss_train_steady, dP_loss_val_steady, dP_loss_train_unsteady, dP_loss_val_unsteady))


reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_lin_reg_model_steady(anatomy, num_geos_steady, seed = 0, hyperparams = hyperparams)
reg_unsteady, dP_loss_val_unsteady, dP_loss_train_unsteady = train_lin_reg_model_unsteady(anatomy, num_geos_unsteady, seed = 0, hyperparams = hyperparams)
print ("{:<8} {:<10} {:<10} {:<10} {:<10}".format('LR',dP_loss_train_steady, dP_loss_val_steady, dP_loss_train_unsteady, dP_loss_val_unsteady))

hyperparams =  {'C': 1.4054800410100454, 'epsilon': 0.02912856892191263}
reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_svr_model_steady(anatomy, num_geos_steady, seed = 0, hyperparams = hyperparams)
reg_unsteady, dP_loss_val_unsteady, dP_loss_train_unsteady = train_svr_model_unsteady(anatomy, num_geos_unsteady, seed = 0, hyperparams = hyperparams)
print ("{:<8} {:<10} {:<10} {:<10} {:<10}".format('SVR',dP_loss_train_steady, dP_loss_val_steady, dP_loss_train_unsteady, dP_loss_val_unsteady))

hyperparams = {'alpha': 0.0019841134279449057, 'length_scale': 1.5714806472185803}
reg_steady, dP_loss_val_steady, dP_loss_train_steady = train_gpr_model_steady(anatomy, num_geos_steady, seed = 0, hyperparams = hyperparams)
reg_unsteady, dP_loss_val_unsteady, dP_loss_train_unsteady = train_gpr_model_unsteady(anatomy, num_geos_unsteady, seed = 0, hyperparams = hyperparams)
print ("{:<8} {:<10} {:<10} {:<10} {:<10}".format('GPR',dP_loss_train_steady, dP_loss_val_steady, dP_loss_train_unsteady, dP_loss_val_unsteady))
