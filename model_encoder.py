


import tensorflow as tf
import numpy as np

from tf_utils_190108.tf_builder import *
from tf_utils_190108.tf_controller import *
import sys, os, time, datetime

from model.components.positional import add_timing_signal_nd

class Encoder():
    def __init__(
        self,
        config,
        config_cnn=[],
        config_cnn_key=['out_dim', 'batch_norm', 'pooling'],
    ):
        # self.config_cnn_set = [
        #     (12, True, None),
        #     (16, True, 4),
        #     (20, True, [2,1]),
        #     (24, False, [1,2]),
        #     (28, True, None),
        #     (32, False, 2),
        #     (36, False, 2),
        # ]
        self.config_cnn_set = config_cnn
        self.config_cnn_key = config_cnn_key
        self.config_cnn = [
            { k: v[i] for i, k in enumerate(self.config_cnn_key) }
            for v in self.config_cnn_set
        ]
        self.cnn_layer_count = len(self.config_cnn_set)
        self.cnn_out_dim_final = 1
        
        self.layers = []
        self.output = None
        self.config = config
        return None
    
    def __call__(self, inputs):
        
        x = inputs
        
        # CNN SERIES
        s = TF_Series(
            inputs=x,
            name='0_CNN',
            layer_kwargs={
                'filter_shape': 3,
                'strides': 1,
                'activator': tf.nn.relu,
                # 'out_dim': 32,
            },
            layers=[
                TF_CNN(False, **cfg) for cfg in self.config_cnn
            ],
        )
        self.cnn_series = s
        self.layers += s.layers
        x = s.output
        cnn_output = x
        
        x = tf.reshape(
            x,
            shape=[
                -1,
                tf.shape(x)[1] * tf.shape(x)[2],
                self.config_cnn[-1]['out_dim']
            ],
            name='encoder_output',
        )
        
        if self.config.positional_embeddings:
            # from tensor2tensor lib - positional embeddings
            x = add_timing_signal_nd(x)
        
        self.output = x
        return self.output
    
    def get_output(self):
        return self.output


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



