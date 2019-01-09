import tensorflow as tf
import numpy as np

from model_decoder_lib import *


# recreate attention mechanism
# img = tf.placeholder(tf.float32, shape=[None, 200, 12, 20])
def decoder(inputs, config, E, start_token, vocab, formula, formula_length):
    attn_meca = AttentionMechanism(
        img=inputs,
        dim_e=config.attn_cell_config["dim_e"],
        tiles=1,
    )
    recu_cell = tf.nn.rnn_cell.LSTMCell(
        config.attn_cell_config["num_units"],
        reuse=tf.AUTO_REUSE,
    )
    attn_cell = AttentionCell(
        recu_cell,
        attn_meca,
        config.dropout,
        config.attn_cell_config,
        config.n_tok,
    )

    # if self._config.decoding == "greedy":
    decoder_cell = GreedyDecoderCell(
        E,
        attn_cell,
        config.batch_size,
        start_token,
        vocab.id_end,
    )

    # elif self._config.decoding == "beam_search":
    # decoder_cell = BeamSearchDecoderCell(E, attn_cell, batch_size,
    #         start_token, self._id_end, self._config.beam_size,
    #         self._config.div_gamma, self._config.div_prob)
    
    embeddings = get_embeddings(
        formula,
        E,
        config.attn_cell_config['dim_embeddings'],
        start_token,
        config.batch_size,
    )
    
    train_outputs, _ = tf.nn.dynamic_rnn(
        attn_cell,
        embeddings,
        initial_state=attn_cell.initial_state()
    )
    
    test_outputs, _ = dynamic_decode(
        decoder_cell,
        config.max_length_formula+1
    )
    
    return train_outputs, test_outputs




