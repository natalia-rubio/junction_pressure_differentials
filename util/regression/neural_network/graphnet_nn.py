import pdb
import dgl
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
import dgl.function as fn
import numpy as np
import pdb
tf.keras.regularizers.L2(l2=0.1)

def inv_scale_tf(scaling_dict, field, field_name):
    mean = tf.constant(scaling_dict[field_name][0], dtype = "float32")
    std = tf.constant(scaling_dict[field_name][1], dtype = "float32")
    scaled_field = tf.add(tf.multiply(field, std), mean)
    return scaled_field


def MLP(in_feats, latent_space, out_feats, n_h_layers):
    initializer = tf.keras.initializers.Zeros()

    encoder_in = tf.keras.layers.Dense(latent_space, activation="relu", use_bias = True)
    encoder_out = tf.keras.layers.Dense(out_feats, activation=None, use_bias = True)
    n_h_layers = n_h_layers

    model = tf.keras.Sequential()
    model.add(encoder_in)
    #model.add(tf.keras.layers.LeakyReLU())
    for i in range(n_h_layers):
        model.add(tf.keras.layers.Dense(latent_space, activation="relu", use_bias = True
                                            ))
        #model.add(tf.keras.layers.LeakyReLU())
    model.add(encoder_out)

    return model

class GraphNet(tf.Module):
    def __init__(self, params):
        super(GraphNet, self).__init__()

        self.nn_model = MLP(params["num_inlet_ft"] + 2*params["num_outlet_ft"],
                                        params['latent_size_mlp'],
                                        params['out_size'],
                                        params['hl_mlp'])

        self.params = params

    def get_model_list(self):
        model_list = [self.nn_model]
        return model_list

    def forward(self, g):

        output = self.nn_model(g)
        stacked_output = tf.reshape(output, [-1,self.params['out_size']])

        return stacked_output

    def update_nn_weights(self, batched_graph, output_tensor, flow_tensor, flow_der_tensor, dP_tensor, scaling_dict, optimizer, loss, output_name):

        with tf.GradientTape() as tape:
            tape.reset()
            pred_outlet_output = tf.cast(self.forward(batched_graph), dtype=tf.float32)
            true_outlet_output = output_tensor
            #loss_value = loss(pred_outlet_output, true_outlet_output)
            loss_value = self.get_dP_loss(batched_graph, flow_tensor, flow_der_tensor, dP_tensor, loss, scaling_dict)

        model_list =  self.get_model_list()
        for model in model_list:

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return loss_value

    def get_dP_loss(self, input_tensor, flow_tensor, flow_der_tensor, dP_tensor, loss, scaling_dict):

        pred_outlet_coefs = tf.cast(self.forward(input_tensor), dtype=tf.float32)
        pred_dP = tf.reshape(inv_scale_tf(scaling_dict, pred_outlet_coefs[:,0], "coef_a"), (-1,1)) * tf.square(flow_tensor) + \
                    tf.reshape(inv_scale_tf(scaling_dict, pred_outlet_coefs[:,1], "coef_b"), (-1,1)) * flow_tensor + \
                    tf.reshape(inv_scale_tf(scaling_dict, pred_outlet_coefs[:,2], "coef_L"), (-1,1)) * (flow_der_tensor)

        #dP_loss = loss(pred_dP[::2]/1333, dP_tensor[::2]/1333)
        dP_loss = loss(pred_dP/1333, dP_tensor/1333)
        return dP_loss

    def get_quad_loss(self, output_tensor, flow_tensor, flow_der_tensor, dP_tensor, loss, scaling_dict):
        #import pdb; pdb.set_trace()
        true_a = tf.reshape(inv_scale_tf(scaling_dict, output_tensor[:,0], "coef_a"),(-1,1))
        true_b = tf.reshape(inv_scale_tf(scaling_dict, output_tensor[:,1], "coef_b"),(-1,1))
        true_L = tf.reshape(inv_scale_tf(scaling_dict, output_tensor[:,2], "coef_L"),(-1,1))

        dP_qfit = (true_a * flow_tensor**2) + (true_b * flow_tensor) + (true_L * flow_der_tensor)
        #quad_loss = loss(dP_qfit[::2]/1333, dP_tensor[::2]/1333)
        quad_loss = loss(dP_qfit/1333, dP_tensor/1333)
        return quad_loss
