import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from sklearn.svm import SVR
from util.regression.neural_network.training_util import *

def train_svr_model_steady(anatomy, num_geos, seed = 0, hyperparams = {}):

    scaling_dict = load_dict(f"/home/nrubio/Desktop/junction_pressure_differentials/data/scaling_dictionaries/{anatomy}_scaling_dict_steady")
    train_dataset = load_dict(f"/home/nrubio/Desktop/junction_pressure_differentials/data/dgl_datasets/{anatomy}/train_{anatomy}_num_geos_{num_geos}_seed_{seed}_dataset_steady")
    val_dataset = load_dict(f"/home/nrubio/Desktop/junction_pressure_differentials/data/dgl_datasets/{anatomy}/val_{anatomy}_num_geos_{num_geos}_seed_{seed}_dataset_steady")

    train_dataloader = get_graph_data_loader(train_dataset, batch_size = len(train_dataset))
    train_input, train_output, train_flow, train_flow_der, train_dP = get_master_tensors_steady(train_dataloader)

    val_dataloader = get_graph_data_loader(val_dataset, batch_size = len(train_dataset))
    val_input, val_output, val_flow, val_flow_der, val_dP = get_master_tensors_steady(val_dataloader)

    svr0 = SVR(C=hyperparams["C"], epsilon = hyperparams["epsilon"]).fit(np.asarray(train_input), np.asarray(train_output[:,0]))
    svr1 = SVR(C=hyperparams["C"], epsilon = hyperparams["epsilon"]).fit(np.asarray(train_input), np.asarray(train_output[:,1]))
    pickle.dump(svr0, open(f"/home/nrubio/Desktop/junction_pressure_differentials/results/models/{len(train_dataset)+len(val_dataset)}_lin_svr0", 'wb'))
    pickle.dump(svr1, open(f"/home/nrubio/Desktop/junction_pressure_differentials/results/models/{len(train_dataset)+len(val_dataset)}_lin_svr1", 'wb'))

    pred_coefs_train = tf.convert_to_tensor(np.stack([svr0.predict(np.asarray(train_input)),
                                                        svr1.predict(np.asarray(train_input))]).T,
                                                        dtype=tf.float64)


    pred_dP_train = tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_train[:,0], "coef_a"), (-1,1)) * tf.square(train_flow) + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_train[:,1], "coef_b"), (-1,1)) * train_flow
    dP_loss_train = rmse(pred_dP_train/1333, train_dP/1333)

    pred_coefs_val = tf.convert_to_tensor(np.stack([svr0.predict(np.asarray(val_input)),
                                                        svr1.predict(np.asarray(val_input))]).T,
                                                        dtype=tf.float64)

    pred_dP_val = tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_val[:,0], "coef_a"), (-1,1)) * tf.square(val_flow) + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_val[:,1], "coef_b"), (-1,1)) * val_flow
    dP_loss_val = rmse(pred_dP_val/1333, val_dP/1333)

    return svr0, dP_loss_val, dP_loss_train

def train_svr_model_unsteady(anatomy, num_geos, seed = 0, hyperparams = {}):

    scaling_dict = load_dict(f"/home/nrubio/Desktop/junction_pressure_differentials/data/scaling_dictionaries/{anatomy}_scaling_dict")
    train_dataset = load_dict(f"/home/nrubio/Desktop/junction_pressure_differentials/data/dgl_datasets/{anatomy}/train_{anatomy}_num_geos_{num_geos}_seed_{seed}_dataset")
    val_dataset = load_dict(f"/home/nrubio/Desktop/junction_pressure_differentials/data/dgl_datasets/{anatomy}/val_{anatomy}_num_geos_{num_geos}_seed_{seed}_dataset")

    train_dataloader = get_graph_data_loader(train_dataset, batch_size = len(train_dataset))
    train_input, train_output, train_flow, train_flow_der, train_dP = get_master_tensors_unsteady(train_dataloader)

    val_dataloader = get_graph_data_loader(val_dataset, batch_size = len(train_dataset))
    val_input, val_output, val_flow, val_flow_der, val_dP = get_master_tensors_unsteady(val_dataloader)

    svr0 = SVR(C=hyperparams["C"], epsilon = hyperparams["epsilon"]).fit(np.asarray(train_input), np.asarray(train_output[:,0]))
    svr1 = SVR(C=hyperparams["C"], epsilon = hyperparams["epsilon"]).fit(np.asarray(train_input), np.asarray(train_output[:,1]))
    svr2 = SVR(C=hyperparams["C"], epsilon = hyperparams["epsilon"]).fit(np.asarray(train_input), np.asarray(train_output[:,2]))

    pickle.dump(svr0, open(f"/home/nrubio/Desktop/junction_pressure_differentials/results/models/{len(train_dataset)+len(val_dataset)}_lin_svr0", 'wb'))
    pickle.dump(svr1, open(f"/home/nrubio/Desktop/junction_pressure_differentials/results/models/{len(train_dataset)+len(val_dataset)}_lin_svr1", 'wb'))
    pickle.dump(svr2, open(f"/home/nrubio/Desktop/junction_pressure_differentials/results/models/{len(train_dataset)+len(val_dataset)}_lin_svr2", 'wb'))

    pred_coefs_train = tf.convert_to_tensor(np.stack([svr0.predict(np.asarray(train_input)),
                                                        svr1.predict(np.asarray(train_input)),
                                                        svr2.predict(np.asarray(train_input))]).T,
                                                        dtype=tf.float64)


    pred_dP_train = tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_train[:,0], "coef_a"), (-1,1)) * tf.square(train_flow) + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_train[:,1], "coef_b"), (-1,1)) * train_flow + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_train[:,2], "coef_L"), (-1,1)) * (train_flow_der)
    dP_loss_train = rmse(pred_dP_train/1333, train_dP/1333)

    pred_coefs_val = tf.convert_to_tensor(np.stack([svr0.predict(np.asarray(val_input)),
                                                        svr1.predict(np.asarray(val_input)),
                                                        svr2.predict(np.asarray(val_input))]).T,
                                                        dtype=tf.float64)

    pred_dP_val = tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_val[:,0], "coef_a"), (-1,1)) * tf.square(val_flow) + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_val[:,1], "coef_b"), (-1,1)) * val_flow + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_val[:,2], "coef_L"), (-1,1)) * (val_flow_der)
    dP_loss_val = rmse(pred_dP_val/1333, val_dP/1333)
    return svr0, dP_loss_val, dP_loss_train
