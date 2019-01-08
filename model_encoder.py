


import tensorflow as tf
import numpy as np

from tf_utils_190108.tf_builder import *
from tf_utils_190108.tf_controller import *
import sys, os, time, datetime


def encoder(inputs):
    x = inputs
    s = TF_Series(
        inputs=x,
        name='0_CNN',
        layer_kwargs={
            'activator': tf.nn.relu,
            'out_dim': 32,
            'strides': 1,
            'filter_shape': 3,
        },
        layers=[
            TF_CNN(False, out_dim=32, batch_norm=True),
            TF_CNN(False, out_dim=24, batch_norm=True, pooling=4),
            TF_CNN(False, out_dim=20, batch_norm=True, pooling=[2,1]),
            TF_CNN(False, out_dim=18, pooling=[1,2]),
            TF_CNN(False, out_dim=16, batch_norm=True, pooling=None),
            TF_CNN(False, out_dim=14, pooling=2),
            TF_CNN(False, out_dim=12, pooling=2),
        ],
    )
    x = s.output
    cnn_output = x

    x = tf.unstack(
        value=x,
        axis=1,
        name='rnn_feed',
    )
    x = [
        TF_RNN(
            feed=v,
            name='1_RNN_' + str(i),
            bidirectional=True,
            sequence_length=None,
            cell_count=[24, 12],
            cell_type='lstm',
        )
        for i, v in enumerate(x)
    ]
    rnn_layers = x
    x = [v.output for v in x]
    rnn_outputs = x
    x = tf.stack(
        values=x,
        axis=1,
        name='encoder_output',
    )
    # x
    # s = TF_Series(
    #     inputs=x,
    #     name='1_RNN',
    #     layer_kwargs={
            
    #     },
    #     layers=[
    #         TF_RNN()
    #     ],
    # )
    
    encoder_output = x
    return encoder_output


# op_training = tf.assign(state_training, True)
# op_testing = tf.assign(state_training, False)
# op_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
#     op_train = optimizer.minimize(loss)
        

# c:512, k:(3,3), s:(1,1), p:(0,0), bn -
# c:512, k:(3,3), s:(1,1), p:(1,1), bn po:(2,1), s:(2,1), p:(0,0)
# c:256, k:(3,3), s:(1,1), p:(1,1) po:(1,2), s:(1,2), p(0,0)
# c:256, k:(3,3), s:(1,1), p:(1,1), bn -
# c:128, k:(3,3), s:(1,1), p:(1,1) po:(2,2), s:(2,2), p:(0,0)
# c:64, k:(3,3), s:(1,1), p:(1,1) po:(2,2), s:(2,2), p(0,0)



