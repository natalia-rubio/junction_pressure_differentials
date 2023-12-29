import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.regression.neural_network.training_util import *
from sklearn.tree import DecisionTreeRegressor

def train_dt_model_steady(anatomy, num_geos, seed = 0, hyperparams = {}):

    scaling_dict = load_dict(f"/home/nrubio/Desktop/junction_pressure_differentials/data/scaling_dictionaries/{anatomy}_scaling_dict")
    train_dataset = load_dict(f"/home/nrubio/Desktop/junction_pressure_differentials/data/dgl_datasets/{anatomy}/train_{anatomy}_num_geos_{num_geos}_seed_{seed}_dataset_steady")
    val_dataset = load_dict(f"/home/nrubio/Desktop/junction_pressure_differentials/data/dgl_datasets/{anatomy}/val_{anatomy}_num_geos_{num_geos}_seed_{seed}_dataset_steady")

    train_dataloader = get_graph_data_loader(train_dataset, batch_size = len(train_dataset))
    train_input, train_output, train_flow, train_flow_der, train_dP, train_output_UO = get_master_tensors_steady(train_dataloader)

    val_dataloader = get_graph_data_loader(val_dataset, batch_size = len(train_dataset))
    val_input, val_output, val_flow, val_flow_der, val_dP, val_output_UO = get_master_tensors_steady(val_dataloader)

    dt = DecisionTreeRegressor(random_state=0, max_depth=hyperparams["max_depth"], min_samples_leaf =hyperparams["min_samples_leaf"] ).fit(np.asarray(train_input), np.asarray(train_output))
    pickle.dump(dt, open(f"/home/nrubio/Desktop/junction_pressure_differentials/results/models/{len(train_dataset)+len(val_dataset)}_dt", 'wb'))


    pred_coefs_train = tf.convert_to_tensor(dt.predict(np.asarray(train_input)), dtype =tf.float64)
    pred_dP_train = tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_train[:,0], "coef_a"), (-1,1)) * tf.square(train_flow) + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_train[:,1], "coef_b"), (-1,1)) * train_flow
    dP_loss_train = rmse(pred_dP_train/1333, train_dP/1333)

    pred_coefs_val = tf.convert_to_tensor(dt.predict(np.asarray(val_input)), dtype =tf.float64)
    pred_dP_val = tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_val[:,0], "coef_a"), (-1,1)) * tf.square(val_flow) + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_val[:,1], "coef_b"), (-1,1)) * val_flow
    dP_loss_val = rmse(pred_dP_val/1333, val_dP/1333)

    return dt, dP_loss_val, dP_loss_train


def train_dt_model_unsteady(anatomy, num_geos, seed = 0, hyperparams = {}):

    scaling_dict = load_dict(f"/home/nrubio/Desktop/junction_pressure_differentials/data/scaling_dictionaries/{anatomy}_scaling_dict")
    train_dataset = load_dict(f"/home/nrubio/Desktop/junction_pressure_differentials/data/dgl_datasets/{anatomy}/train_{anatomy}_num_geos_{num_geos}_seed_{seed}_dataset")
    val_dataset = load_dict(f"/home/nrubio/Desktop/junction_pressure_differentials/data/dgl_datasets/{anatomy}/val_{anatomy}_num_geos_{num_geos}_seed_{seed}_dataset")

    train_dataloader = get_graph_data_loader(train_dataset, batch_size = len(train_dataset))
    train_input, train_output, train_flow, train_flow_der, train_dP, train_output_UO = get_master_tensors_unsteady(train_dataloader)

    val_dataloader = get_graph_data_loader(val_dataset, batch_size = len(train_dataset))
    val_input, val_output, val_flow, val_flow_der, val_dP, val_output_UO = get_master_tensors_unsteady(val_dataloader)


    dt = DecisionTreeRegressor(random_state=0, max_depth=hyperparams["max_depth"], min_samples_leaf =hyperparams["min_samples_leaf"] ).fit(np.asarray(train_input), np.asarray(train_output))
    pickle.dump(dt, open(f"/home/nrubio/Desktop/junction_pressure_differentials/results/models/{len(train_dataset)+len(val_dataset)}_dt", 'wb'))

    pred_coefs_train = tf.convert_to_tensor(dt.predict(np.asarray(train_input)), dtype =tf.float64)
    pred_dP_train = tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_train[:,0], "coef_a"), (-1,1)) * tf.square(train_flow) + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_train[:,1], "coef_b"), (-1,1)) * train_flow + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_train[:,2], "coef_L"), (-1,1)) * (train_flow_der)
    dP_loss_train = rmse(pred_dP_train/1333, train_dP/1333)

    pred_coefs_val = tf.convert_to_tensor(dt.predict(np.asarray(val_input)), dtype =tf.float64)
    pred_dP_val = tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_val[:,0], "coef_a"), (-1,1)) * tf.square(val_flow) + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_val[:,1], "coef_b"), (-1,1)) * val_flow + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_val[:,2], "coef_L"), (-1,1)) * (val_flow_der)
    dP_loss_val = rmse(pred_dP_val/1333, val_dP/1333)

    return dt, dP_loss_val, dP_loss_train

def train_dt_model_unsteady_UO(anatomy, num_geos, seed = 0, hyperparams = {}):

    scaling_dict = load_dict(f"/home/nrubio/Desktop/junction_pressure_differentials/data/scaling_dictionaries/{anatomy}_scaling_dict")
    train_dataset = load_dict(f"/home/nrubio/Desktop/junction_pressure_differentials/data/dgl_datasets/{anatomy}/train_{anatomy}_num_geos_{num_geos}_seed_{seed}_dataset")
    val_dataset = load_dict(f"/home/nrubio/Desktop/junction_pressure_differentials/data/dgl_datasets/{anatomy}/val_{anatomy}_num_geos_{num_geos}_seed_{seed}_dataset")

    train_dataloader = get_graph_data_loader(train_dataset, batch_size = len(train_dataset))
    train_input, train_output, train_flow, train_flow_der, train_dP, train_output_UO = get_master_tensors_unsteady(train_dataloader)

    val_dataloader = get_graph_data_loader(val_dataset, batch_size = len(train_dataset))
    val_input, val_output, val_flow, val_flow_der, val_dP, val_output_UO = get_master_tensors_unsteady(val_dataloader)


    dt = DecisionTreeRegressor(random_state=0, max_depth=hyperparams["max_depth"], min_samples_leaf =hyperparams["min_samples_leaf"] ).fit(np.asarray(train_input), np.asarray(train_output_UO))
    pickle.dump(dt, open(f"/home/nrubio/Desktop/junction_pressure_differentials/results/models/{len(train_dataset)+len(val_dataset)}_dt", 'wb'))

    pred_coefs_train = tf.convert_to_tensor(dt.predict(np.asarray(train_input)), dtype =tf.float64)
    pred_dP_train = tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_train[:,0], "coef_a_UO"), (-1,1)) * tf.square(train_flow) + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_train[:,1], "coef_b_UO"), (-1,1)) * train_flow + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_train[:,2], "coef_L_UO"), (-1,1)) * (train_flow_der)
    dP_loss_train = rmse(pred_dP_train/1333, train_dP/1333)

    pred_coefs_val = tf.convert_to_tensor(dt.predict(np.asarray(val_input)), dtype =tf.float64)
    pred_dP_val = tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_val[:,0], "coef_a_UO"), (-1,1)) * tf.square(val_flow) + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_val[:,1], "coef_b_UO"), (-1,1)) * val_flow + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_coefs_val[:,2], "coef_L_UO"), (-1,1)) * (val_flow_der)
    dP_loss_val = rmse(pred_dP_val/1333, val_dP/1333)

    return dt, dP_loss_val, dP_loss_train
