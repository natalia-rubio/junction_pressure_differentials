import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.regression.neural_network.training_util import *

#scaling_dict = load_dict("/home/nrubio/Desktop/aorta_scaling_dict")
def print_error(train_results, val_results, epoch):
    msg = '{:.0f}\t'.format(epoch)
    msg = msg + 'Train RMSE = {:.2e}.  '.format(train_results['dP_loss']/train_results['count'])
    msg = msg + 'Validation RMSE = {:.2e}.'.format(val_results['dP_loss']/val_results['count'])
    msg = msg + 'Train RMSE (COEF) = {:.2e}.  '.format(train_results['coef_loss']/train_results['count'])
    msg = msg + 'Validation RMSE (COEF) = {:.2e}.'.format(val_results['coef_loss']/val_results['count'])
    print(msg, flush=True)
    return



def loop_over(dataloader, gnn_model, output_name, loss, optimizer = None, unsteady = False):

    coef_loss = 0
    dP_loss = 0
    count = 0

    input_tensors = dataloader[0]
    output_tensors = dataloader[1]
    flow_tensors = dataloader[2]
    flow_der_tensors = dataloader[3]
    dP_tensors = dataloader[4]

    for batch_ind in range(len(dataloader[0])):
        #import pdb; pdb.set_trace()
        pred_outlet_output = gnn_model.forward(input_tensors[batch_ind])
        true_outlet_output = output_tensors[batch_ind]
        # print(f"True Coefficients {true_outlet_output[:2, :]}")
        # print(f"Predicted Coefficients {pred_outlet_output[:2, :]}")
        coef_loss_value = loss(pred_outlet_output, true_outlet_output)
        coef_loss = coef_loss + coef_loss_value.numpy()

        dP_loss_value = tf.math.sqrt(gnn_model.get_dP_loss(input_tensors[batch_ind],
                            flow_tensors[batch_ind],
                            flow_der_tensors[batch_ind],
                            dP_tensors[batch_ind],
                            loss))

        dP_loss = dP_loss + dP_loss_value.numpy()

        if optimizer != None:
            loss_value_gnn = gnn_model.update_nn_weights(input_tensors[batch_ind],
                                                     output_tensors[batch_ind],
                                                     flow_tensors[batch_ind],
                                                     flow_der_tensors[batch_ind],
                                                     dP_tensors[batch_ind],
                                                     optimizer, loss, output_name)

        count = count + 1

    return {'coef_loss': coef_loss, 'dP_loss': dP_loss, 'count': count}


def evaluate_model(gnn_model,
                    train_master_tensors,
                    batch_size,
                    loss,
                    optimizer = None,
                    validation_master_tensors = None,
                    output_name = None,
                    train = True,
                    unsteady = False):

    if validation_master_tensors != None:
        if unsteady:
            validation_batched_tensors = get_batched_tensors_unsteady(validation_master_tensors, validation_master_tensors[0].shape[0], 0)
        else:
            validation_batched_tensors = get_batched_tensors_steady(validation_master_tensors, validation_master_tensors[0].shape[0], 0)
        validation_results = loop_over(dataloader = validation_batched_tensors,
                                        gnn_model = gnn_model,
                                        output_name = output_name,
                                        loss = loss)

    else:
        validation_results = None

    if train_master_tensors != None:
        if unsteady:
            train_batched_tensors = get_batched_tensors_unsteady(train_master_tensors, batch_size, noise_level = 0)
        else:
            train_batched_tensors = get_batched_tensors_steady(train_master_tensors, batch_size, noise_level = 0)

        train_results = loop_over(dataloader = train_batched_tensors,
                                        gnn_model = gnn_model,
                                        output_name = output_name,
                                        loss = loss,
                                        optimizer = optimizer,
                                        unsteady = unsteady)
    else:
        train_results = None
    return train_results, validation_results


def train_gnn_model(anatomy, gnn_model, train_dataset, validation_dataset, train_params, network_params,
    trial=1, percent_train = 60, model_name = None, index = 0,):

    unsteady = network_params["unsteady"]

    print('Training dataset contains {:.0f} graphs'.format(len(train_dataset)))


    train_dataloader = get_graph_data_loader(train_dataset, batch_size=len(train_dataset))
    if unsteady:
        train_master_tensors = get_master_tensors_unsteady(train_dataloader)
    else:
        train_master_tensors = get_master_tensors_steady(train_dataloader)

    train_input_tensor_data_loader = train_master_tensors[0]
    train_output_tensor_data_loader = train_master_tensors[1]
    train_flow_tensor_data_loader = train_master_tensors[2]
    train_flow_der_tensor_data_loader = train_master_tensors[3]
    train_dP_tensor_data_loader = train_master_tensors[4]

    validation_dataloader = get_graph_data_loader(validation_dataset, batch_size=len(validation_dataset))
    if unsteady:
        validation_master_tensors = get_master_tensors_unsteady(validation_dataloader)
    else:
        validation_master_tensors = get_master_tensors_steady(validation_dataloader)

    validation_input_tensor_data_loader = validation_master_tensors[0]
    validation_output_tensor_data_loader = validation_master_tensors[1]
    validation_flow_tensor_data_loader = validation_master_tensors[2]
    validation_flow_der_tensor_data_loader = validation_master_tensors[3]
    validation_dP_tensor_data_loader = validation_master_tensors[4]

    nepochs = train_params['nepochs']
    learning_rate = get_learning_rate(train_params)
    optimizer = get_optimizer(train_params, learning_rate)
    mse_coef_train_list = []; mse_dP_train_list = []; mse_coef_val_list = []; mse_dP_val_list = []
    for epoch in range(nepochs):
        train_results, val_results = evaluate_model(gnn_model = gnn_model,
                                                    train_master_tensors = train_master_tensors,
                                                    loss = mse,
                                                    batch_size = train_params["batch_size"],
                                                    optimizer = optimizer,
                                                    output_name = network_params["output_name"],
                                                    validation_master_tensors = validation_master_tensors,
                                                    unsteady = unsteady)
        mse_coef_train_list.append(train_results['coef_loss']/train_results['count'])
        mse_coef_val_list.append(val_results['coef_loss']/val_results['count'])
        mse_dP_train_list.append(train_results['dP_loss']/train_results['count'])
        mse_dP_val_list.append(val_results['dP_loss']/val_results['count'])
        print_error(train_results, val_results, epoch)

    # compute final loss
    train_results, val_results = evaluate_model(gnn_model = gnn_model,
                                                train_master_tensors = train_master_tensors,
                                                loss = mse,
                                                batch_size = train_params["batch_size"],
                                                optimizer = optimizer,
                                                output_name = network_params["output_name"],
                                                validation_master_tensors = validation_master_tensors,
                                                unsteady = unsteady)


    cp_loss = tf.math.sqrt(mse(validation_dP_tensor_data_loader*0,
        validation_dP_tensor_data_loader/1333).numpy())
    quad_loss = tf.math.sqrt(gnn_model.get_quad_loss(validation_output_tensor_data_loader,
        validation_flow_tensor_data_loader,
        validation_flow_der_tensor_data_loader,
        validation_dP_tensor_data_loader,
        mse))
    quad_loss_train = tf.math.sqrt(gnn_model.get_quad_loss(train_output_tensor_data_loader,
        train_flow_tensor_data_loader,
        train_flow_der_tensor_data_loader,
        train_dP_tensor_data_loader,
        mse))
    print(quad_loss_train)
    if model_name == None:
        model_name = str(network_params["hl_mlp"])[0:4] + "_hl_" + str(network_params["latent_size_mlp"])[0:4] + "_lsmlp_" + (str(train_params["learning_rate"])[0:6]).replace(".", "_") + "_lr_"+ "_bs_" + str(train_params["batch_size"]) + "_nepochs_" + str(train_params["nepochs"]) + anatomy

    plt.clf()
    #plt.style.use('dark_background')
    plt.scatter(np.linspace(1,nepochs, nepochs, endpoint=True), np.asarray(mse_dP_train_list), color = "royalblue", s=30, alpha = 0.6, marker='o', label="NN (Train)")
    plt.scatter(np.linspace(1,nepochs, nepochs, endpoint=True), np.asarray(mse_dP_val_list),  color = "orangered", s=30, alpha = 0.6, marker='d', label="NN (Val)")
    plt.plot(np.linspace(1, nepochs, nepochs, endpoint=True), np.asarray(mse_dP_val_list)*0+cp_loss, "--", color = "peru", label="Constant Pressure (Val)")
    plt.plot(np.linspace(1,nepochs, nepochs, endpoint=True), np.asarray(mse_dP_val_list)*0+quad_loss, "--",  color = "seagreen", label="Optimal Coef Fit (Val)")
    plt.xlabel("epoch"); plt.ylabel("RMSE (mmHg)"); #plt.title(f"MSE Over Epochs"); plt.legend();
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, mode="expand", borderaxespad=0.)

    plt.savefig("results/training_plots/" + model_name + "mse_over_epochs.png", dpi=1200, bbox_inches='tight', transparent=False)

    loss_history = {"train": mse_dP_train_list,
                    "val": mse_dP_val_list}

    train_mse = train_results['dP_loss']/train_results['count']
    val_mse = val_results['dP_loss']/val_results['count']

    get_best_worst = True
    if get_best_worst:
        best = 100; best_ind = 0
        worst = 0; worst_ind = 0
        #import pdb; pdb.set_trace()
        for i, graph in enumerate(validation_dataset):
            validation_dataloader = validation_dataset[i]
            if unsteady:
                validation_master_tensors = get_master_tensors_unsteady([validation_dataloader])
            else:
                validation_master_tensors = get_master_tensors_steady([validation_dataloader])

            train_results, val_results = evaluate_model(gnn_model = gnn_model,
                                                        train_master_tensors = train_master_tensors,
                                                        loss = mse,
                                                        batch_size = train_params["batch_size"],
                                                        optimizer = optimizer,
                                                        output_name = network_params["output_name"],
                                                        validation_master_tensors = validation_master_tensors,
                                                        unsteady = unsteady)
            if val_results['dP_loss'] < best:
                best_ind = i
                best = val_results['dP_loss']
            if val_results['dP_loss'] > worst:
                worst_ind = i
                worst = val_results['dP_loss']
        print(f"Best index: {best_ind} ({best})")
        print(f"Worst index: {worst_ind} ({worst})")
    return gnn_model, val_mse, train_mse
