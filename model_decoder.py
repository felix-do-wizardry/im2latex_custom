import tensorflow as tf
import numpy as np

from model_decoder_lib import *

# from model.components.greedy_decoder_cell import GreedyDecoderCell
# from model.components.beam_search_decoder_cell import BeamSearchDecoderCell

# recreate attention mechanism
# img = tf.placeholder(tf.float32, shape=[None, 200, 12, 20])
class Decoder():
    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab
        # self._n_tok = n_tok
        # self._id_end = id_end
        self.tiles = 1 if self.config.decoding == "greedy" else self.config.beam_size
        return None
    
    def __call__(self, inputs, formula, formula_length, E, start_token):
        self.inputs = inputs
        self.E = E
        self.start_token = start_token
        self.formula = formula
        self.tiled_batch_size = tf.shape(self.formula)[0]
        self.formula_length = formula_length
        self.recu_cell = tf.nn.rnn_cell.LSTMCell(
            self.config.attn_cell_config["num_units"],
            reuse=tf.AUTO_REUSE,
        )
        with tf.variable_scope("attn_cell", reuse=False):
            self.attn_meca = AttentionMechanism(
                img=self.inputs,
                dim_e=self.config.attn_cell_config["dim_e"],
                tiles=1,
            )
            self.attn_cell = AttentionCell(
                self.recu_cell,
                self.attn_meca,
                self.config.dropout,
                self.config.attn_cell_config,
                self.vocab.n_tok,
            )
        
        with tf.variable_scope("attn_cell", reuse=True):
            self.attn_meca = AttentionMechanism(
                img=self.inputs,
                dim_e=self.config.attn_cell_config["dim_e"],
                tiles=self.tiles,
            )
            self.attn_cell = AttentionCell(
                self.recu_cell,
                self.attn_meca,
                self.config.dropout,
                self.config.attn_cell_config,
                self.vocab.n_tok,
            )
            if self.config.decoding == "greedy":
                self.decoder_cell = GreedyDecoderCell(
                    self.E,
                    self.attn_cell,
                    self.tiled_batch_size,
                    self.start_token,
                    self.vocab.id_end,
                )
            elif self.config.decoding == "beam_search":
                self.decoder_cell = BeamSearchDecoderCell(
                    self.E,
                    self.attn_cell,
                    self.tiled_batch_size,
                    self.start_token,
                    self.vocab.id_end,
                    self.config.beam_size,
                    self.config.div_gamma,
                    self.config.div_prob,
                )
        
        self.embeddings = get_embeddings(
            self.formula,
            self.E,
            self.config.attn_cell_config['dim_embeddings'],
            self.start_token,
            self.tiled_batch_size
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




