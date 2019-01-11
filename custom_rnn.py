
import tensorflow as tf
import numpy as np
import collections
import os, sys, time, datetime

class AttentionLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(
        self,
        img,
        dropout,
        out_dim=10,
        ac_config={},
        tiles=1,
        dtype=tf.float32
    ):
        self._scope = 'attention/'
        
        # input img
        self.img = img
        
        # hyperparameters and shapes
        # self.ac_config = ac_config
        self.ac_config = {
            'dim_e': 200,
            'dim_o': 220,
            'num_units': 240,
            'dim_embeddings': 80,
            'n_channels': 512, # should be the same as the img channels
        }
        self._tiles          = tiles
        self._dropout        = dropout
        self._n_channels     = 1
        self._n_regions      = 1
        self._dim_e          = self.ac_config['dim_e']
        self._dim_o          = self.ac_config['dim_o']
        self._num_units      = self.ac_config['num_units']
        self._dim_embeddings = self.ac_config['dim_embeddings']
        self._num_proj       = out_dim
        self._dtype          = dtype
        
        # process input img, get size
        self.image_process(self.img)
        
        # variables and tensors
        with tf.variable_scope(self._scope):
            self._cell = tf.nn.rnn_cell.LSTMCell(
                self._num_units,
                name='rnn_cell',
                reuse=tf.AUTO_REUSE,
            )
        
        # for RNNCell
        self._state_size = AttentionState(self._cell._state_size, self._dim_o)
    
    
    @property
    def state_size(self):
        return self._state_size
    
    @property
    def output_size(self):
        # return self._num_proj
        return self._num_proj + self._n_regions
    
    @property
    def output_dtype(self):
        return self._dtype
    
    def initial_state(self):
        """Returns initial state for the lstm"""
        initial_cell_state = self.attn_mech_initial_cell_state(self._cell)
        initial_o          = self.attn_mech_initial_state("o", self._dim_o)

        return AttentionState(initial_cell_state, initial_o)
    


    def attn_mech_initial_cell_state(self, cell):
        """Returns initial state of a cell computed from the image

        Assumes cell.state_type is an instance of named_tuple.
        Ex: LSTMStateTuple

        Args:
            cell: (instance of RNNCell) must define _state_size

        """
        _states_0 = []
        for hidden_name in cell._state_size._fields:
            hidden_dim = getattr(cell._state_size, hidden_name)
            h = self.attn_mech_initial_state(hidden_name, hidden_dim)
            _states_0.append(h)

        initial_state_cell = type(cell.state_size)(*_states_0)

        return initial_state_cell


    def attn_mech_initial_state(self, name, dim):
        """Returns initial state of dimension specified by dim"""
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            img_mean = tf.reduce_mean(self._img, axis=1)
            W = tf.get_variable("W_{}_0".format(name), shape=[self._n_channels, dim])
            b = tf.get_variable("b_{}_0".format(name), shape=[dim])
            h = tf.tanh(tf.matmul(img_mean, W) + b)

            return h
    
    def image_process(self, img):
        if len(img.shape) == 3:
            self._img = img
        elif len(img.shape) == 4:
            N    = tf.shape(img)[0]
            H, W = tf.shape(img)[1], tf.shape(img)[2] # image
            C    = img.shape[3].value                 # channels
            self._img = tf.reshape(img, shape=[N, H*W, C])
        else:
            print("Image shape not supported")
            raise NotImplementedError
        
        # dimensions
        self._n_regions  = tf.shape(self._img)[1]
        self._n_channels = self._img.shape[2].value
        return True

    
    def compute_h(self, embeddings_prev, cs_prev, o_prev):
        """
        Args:
            embeddings_prev: previous word embeddings [ dim_embeddings ]
            o_prev: previous o state [ dim_o ]
            cs_prev: previous tuple of core rnn states ([ num_u ], [ num_u ])
        Returns:
            h_new: new h state [ num_u ]
            cs_new: new tuple of core rnn states ([ num_u ], [ num_u ])
        """
        
        # concat core inputs [embeddings, o]
        core_inputs = tf.concat(
            [embeddings_prev, o_prev],
            axis=-1,
            name='core_inputs',
        )
        
        # compute new states through the core rnn cell
        h_new, cs_new = self._cell.__call__(
            core_inputs,
            cs_prev
        )
        
        # dropout for h states
        h_new_dropout = tf.nn.dropout(
            h_new,
            self._dropout,
            name='h_dropout',
        )
        
        return h_new_dropout, cs_new
    
    def compute_attention(self, h_new):
        """
        Args:
            h_new: new h state [ num_u ]
        Returns:
            c: current c vector [ n_channels ]
            a: current a vector [ n_regions ]
        """
        
        # attention vector over the image
        self._att_img = tf.layers.dense(
            inputs=self._img,
            units=self._dim_e,
            use_bias=False,
            name='att_img'
        )
        
        # tile _img and _att_img if needed for Beam Search
        if self._tiles > 1:
            att_img = tf.expand_dims(self._att_img, axis=1)
            att_img = tf.tile(att_img, multiples=[1, self._tiles, 1, 1])
            att_img = tf.reshape(att_img, shape=[-1, self._n_regions,
                    self._dim_e])
            img = tf.expand_dims(self._img, axis=1)
            img = tf.tile(img, multiples=[1, self._tiles, 1, 1])
            img = tf.reshape(img, shape=[-1, self._n_regions,
                    self._n_channels])
        else:
            att_img = self._att_img
            img     = self._img

        # computes attention over the hidden vector
        att_h = tf.layers.dense(
            inputs=h_new,
            units=self._dim_e,
            use_bias=False,
        )

        # sums the two contributions
        att_h_expand = tf.expand_dims(att_h, axis=1)
        att_merge = tf.tanh(att_img + att_h_expand)
        
        # dense layer to get attention vector
        att_dense = tf.layers.dense(
            inputs=att_merge,
            units=1,
            use_bias=False,
        )
        att_distribution = tf.nn.softmax(
            att_dense,
            axis=1,
            name='att_distribution',
        )
        att_vector = tf.reduce_sum(att_distribution, axis=2, name='att_vector')
        
        c_img = tf.multiply(att_distribution, img, name='c_img')
        c = tf.reduce_sum(c_img, axis=1, name='c')

        return c, att_vector
    
    def compute_o(self, h_new, c):
        """
        Args:
            h_new: new h state [ num_u ]
            c: current c vector [ n_channels ]
        Returns:
            o_new: new o state [ dim_o ]
        """
        # compute new o state
        o_inputs = tf.concat(
            [h_new, c],
            axis=-1,
            name='o_inputs',
        )
        o_new = tf.layers.dense(
            inputs=o_inputs,
            units=self._dim_o,
            use_bias=False,
            activation=tf.tanh,
            name='o_new_dense'
        )
        o_new = tf.nn.dropout(o_new, self._dropout, name='o_new')
        return o_new
    
    def compute_logits(self, o_new):
        """
        Args:
            o_new: new o state [ dim_o ]
        Returns:
            p: current p logits vector [ num_proj ]
                (probabilistic distribution)
        """
        p = tf.layers.dense(
            inputs=o_new,
            units=self._num_proj,
            use_bias=False,
            name='logits_vector',
        )
        return p
    
    
    def __call__(self, inputs, state):
        """
        Args:
            inputs: previous word embeddings [ dim_embeddings ]
            state: previous state ( core cell state, o ) [ num_u + dim_o ]
        Returns:
            logits: prediction logits as probabilistic distribution [ n_tok ]
            state_new: new state ( core cell state, o ) [ num_u + dim_o ]
        """
        
        self._cs_prev, self._o_prev = state

        # scope = tf.get_variable_scope()
        with tf.variable_scope('attention_step', reuse=tf.AUTO_REUSE):
            
            # compute new h and cs states (self._h_new, self._cs_new)
            self._h_new, self._cs_new = self.compute_h(
                inputs,
                self._cs_prev,
                self._o_prev,
            )
            
            # compute attention vectors c, a
            self._c, self._att_vector = self.compute_attention(self._h_new)
            
            # compute new o state
            self._o_new = self.compute_o(self._h_new, self._c)
            
            # compute new logits vector
            self._logits = self.compute_logits(self._o_new)
            
            # set step outputs
            self._step_outputs = tf.concat(
                [self._logits, self._att_vector],
                axis=-1,
                name='step_outputs',
            )

            # new Attention cell state
            self._new_state = AttentionState(self._cs_new, self._o_new)
            
            return (self._step_outputs, self._new_state)



vocab = {
    'n_tok': 11,
}
config = {
    'dropout': 0.1,
    'dim_embeddings': 80,
    'batch_size': 15,
    'cnn_out_dim': 128
}

inputs = tf.placeholder(tf.float32, shape=[None, None, config['cnn_out_dim']])
tok_embeddings = tf.placeholder(tf.float32, [None, None, config['dim_embeddings']])
fd = {
    inputs: np.random.sample([
        config['batch_size'],
        200,
        config['cnn_out_dim'],
    ]),
    tok_embeddings: np.random.sample([
        config['batch_size'],
        17,
        config['dim_embeddings'],
    ]),
}
attn_cell = AttentionLSTMCell(
    img=inputs,
    dropout=config['dropout'],
    out_dim=vocab['n_tok'],
)

pred, states = tf.nn.dynamic_rnn(
    attn_cell,
    tok_embeddings,
    initial_state=attn_cell.initial_state(),
)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
a, b = sess.run([pred, states], fd)
sess.close()


AttentionState = collections.namedtuple("AttentionState", ("cell_state", "o"))


# class AttentionMechanism(object):
#     """Class to compute attention over an image"""

#     def __init__(self, img, dim_e, tiles=1):
#         if len(img.shape) == 3:
#             self._img = img
#         elif len(img.shape) == 4:
#             N    = tf.shape(img)[0]
#             H, W = tf.shape(img)[1], tf.shape(img)[2] # image
#             C    = img.shape[3].value                 # channels
#             self._img = tf.reshape(img, shape=[N, H*W, C])
#         else:
#             print("Image shape not supported")
#             raise NotImplementedError

#         # dimensions
#         self._n_regions  = tf.shape(self._img)[1]
#         self._n_channels = self._img.shape[2].value
#         self._dim_e      = dim_e
#         self._tiles      = tiles
#         self._scope_name = "att_mechanism"



#     def context(self, h):
#         """Computes attention

#         Args:
#             h: (batch_size, num_units) hidden state

#         Returns:
#             c: (batch_size, channels) context vector

#         """
        
#         with tf.variable_scope(self._scope_name):
#             # attention vector over the image
#             self._att_img = tf.layers.dense(
#                 inputs=self._img,
#                 units=self._dim_e,
#                 use_bias=False,
#                 name="att_img")
            
#             if self._tiles > 1:
#                 att_img = tf.expand_dims(self._att_img, axis=1)
#                 att_img = tf.tile(att_img, multiples=[1, self._tiles, 1, 1])
#                 att_img = tf.reshape(att_img, shape=[-1, self._n_regions,
#                         self._dim_e])
#                 img = tf.expand_dims(self._img, axis=1)
#                 img = tf.tile(img, multiples=[1, self._tiles, 1, 1])
#                 img = tf.reshape(img, shape=[-1, self._n_regions,
#                         self._n_channels])
#             else:
#                 att_img = self._att_img
#                 img     = self._img

#             # computes attention over the hidden vector
#             att_h = tf.layers.dense(inputs=h, units=self._dim_e, use_bias=False)

#             # sums the two contributions
#             att_h = tf.expand_dims(att_h, axis=1)
#             att = tf.tanh(att_img + att_h)

#             # computes scalar product with beta vector
#             # works faster with a matmul than with a * and a tf.reduce_sum
#             att_beta = tf.get_variable("att_beta", shape=[self._dim_e, 1],
#                     dtype=tf.float32)
#             att_flat = tf.reshape(att, shape=[-1, self._dim_e])
#             e = tf.matmul(att_flat, att_beta)
#             e = tf.reshape(e, shape=[-1, self._n_regions])

#             # compute weights
#             a = tf.nn.softmax(e)
#             a = tf.expand_dims(a, axis=-1)
#             c = tf.reduce_sum(a * img, axis=1)

#             return c







