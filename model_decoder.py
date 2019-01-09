import tensorflow as tf
import numpy as np

from model_decoder_lib import *


# recreate attention mechanism
# img = tf.placeholder(tf.float32, shape=[None, 200, 12, 20])
class Decoder():
    def __init__(self):
        return None
    
    def __call__(self, inputs, config, E, start_token, vocab, formula, formula_length):
        self.inputs = inputs
        self.config = config
        self.E = E
        self.start_token = start_token
        self.vocab = vocab
        self.formula = formula
        self.formula_length = formula_length
        self.attn_meca = AttentionMechanism(
            img=self.inputs,
            dim_e=self.config.attn_cell_config["dim_e"],
            tiles=1,
        )
        self.recu_cell = tf.nn.rnn_cell.LSTMCell(
            self.config.attn_cell_config["num_units"],
            reuse=tf.AUTO_REUSE,
        )
        self.attn_cell = AttentionCell(
            self.recu_cell,
            self.attn_meca,
            self.config.dropout,
            self.config.attn_cell_config,
            self.config.n_tok,
        )

        # if self._config.decoding == "greedy":
        self.decoder_cell = GreedyDecoderCell(
            self.E,
            self.attn_cell,
            self.config.batch_size,
            self.start_token,
            self.vocab.id_end,
        )

        # elif self._config.decoding == "beam_search":
        # decoder_cell = BeamSearchDecoderCell(E, attn_cell, batch_size,
        #         start_token, self._id_end, self._config.beam_size,
        #         self._config.div_gamma, self._config.div_prob)
        
        self.embeddings = get_embeddings(
            self.formula,
            self.E,
            self.config.attn_cell_config['dim_embeddings'],
            self.start_token,
            self.config.batch_size,
        )
        
        self.train_outputs, self.train_states = tf.nn.dynamic_rnn(
            self.attn_cell,
            self.embeddings,
            initial_state=self.attn_cell.initial_state()
        )
        
        self.test_outputs, self.test_states = dynamic_decode(
            self.decoder_cell,
            self.config.max_length_formula+1
        )
        
        return self.train_outputs, self.test_outputs




