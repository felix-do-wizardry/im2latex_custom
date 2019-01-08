
import tensorflow as tf
import numpy as np
import collections

from model_encoder import *
from model_decoder import *


_config = {
    'max_length_formula': 32,
    'n_tok': 20,
    'batch_size': 20,
    'dropout': 0.01,
}
_attn_cell_config = {
    'cell_type': 'lstm',
    'num_units': 32,
    'dim_e'    : 40,
    'dim_o'    : 48,
    'dim_embeddings': 60,
}
_vocab = {
    'id_end': 0,
}


start_token = tf.get_variable(
    "start_token",
    dtype=tf.float32,
    shape=[_attn_cell_config['dim_embeddings']],
    initializer=embedding_initializer(),
)
E = tf.get_variable(
    "E",
    initializer=embedding_initializer(),
    shape=[_config['n_tok'], _attn_cell_config['dim_embeddings']],
    dtype=tf.float32
)

# inputs
inputs = tf.placeholder(tf.float32, shape=[None, 128, 1280, 1], name='inputs')
# labels
formula = tf.placeholder(tf.int32, shape=[None, _config['n_tok']], name='labels')
formula_length = tf.placeholder(tf.int32, shape=(None, ), name='labels_length')

learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
dropout = tf.placeholder(tf.float32, shape=(), name='dropout')
training = tf.placeholder(tf.bool, shape=(), name="training")


# encoder_output
encoder_output = encoder(inputs)
pred_train, pred_test = decoder(
    encoder_output,
    _attn_cell_config,
    _config,
    E,
    start_token,
    _vocab,
    formula,
    formula_length,
)

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=pred_train,
    labels=formula,
    name='losses'
)

mask = tf.sequence_mask(formula_length, name='labels_mask')
losses_masked = tf.boolean_mask(losses, mask, name='losses_masked')

# loss for training
loss = tf.reduce_mean(losses_masked, name='loss')

# # to compute perplexity for test
ce_words = tf.reduce_sum(losses, name='loss_each_word') # sum of CE for each word
n_words = tf.reduce_sum(formula_length, name='n_words') # number of words

optimizer_dict = {
    'adam': tf.train.AdamOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'adadelta': tf.train.AdadeltaOptimizer,
    'sgd': tf.train.GradientDescentOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
}

learning_rate = 0.01

optimizer_name = 'adam'
optimizer_func = tf.train.AdamOptimizer
if optimizer_name.lower() in optimizer_dict:
    optimizer_func = optimizer_dict[optimizer_name]

optimizer = optimizer_func(learning_rate)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    op_train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# for tensorboard
timecode = time.strftime("%y%m%d_%H%M%S", time.gmtime())
fileWriter = tf.summary.FileWriter('tensorboard/' + timecode, tf.get_default_graph())
fileWriter.flush()
# tf.summary.scalar("loss", loss)


import click
from model.utils.data_generator import DataGenerator

from model.utils.general import Config
from model.utils.text import Vocab
from model.utils.image import greyscale


@click.command()
@click.option('--data', default="configs/data_small.json",
        help='Path to data json config')
@click.option('--vocab', default="configs/vocab_small.json",
        help='Path to vocab json config')
@click.option('--training', default="configs/training_small.json",
        help='Path to training json config')
@click.option('--model', default="configs/model.json",
        help='Path to model json config')
@click.option('--output', default="results/small/",
        help='Dir for results and model weights')
def main(data, vocab, training, model, output):
    print('got here')
    # Load configs
    dir_output = output
    config = Config([data, vocab, training, model])
    config.save(dir_output)
    vocab = Vocab(config)
    
    train_set = DataGenerator(
        path_formulas=config.path_formulas_train,
        dir_images=config.dir_images_train,
        img_prepro=greyscale,
        max_iter=config.max_iter,
        bucket=config.bucket_train,
        path_matching=config.path_matching_train,
        max_len=config.max_length_formula,
        form_prepro=vocab.form_prepro
    )
    val_set = DataGenerator(
        path_formulas=config.path_formulas_val,
        dir_images=config.dir_images_val,
        img_prepro=greyscale,
        max_iter=config.max_iter,
        bucket=config.bucket_val,
        path_matching=config.path_matching_val,
        max_len=config.max_length_formula,
        form_prepro=vocab.form_prepro
    )
    return train_set, val_set


a = main(
    "configs/data_small.json",
    "configs/vocab_small.json",
    "configs/training_small.json",
    "configs/model.json",
    "results/small/",
)
if __name__ == "__main__":
    main()


from model.utils.general import Config, Progbar, minibatches
from model.utils.image import pad_batch_images
from model.utils.text import pad_batch_formulas
# from model.evaluation.text import score_files, write_answers, truncate_end

dir_output = "results/small/"
config = Config([
    "configs/data_small.json",
    "configs/vocab_small.json",
    "configs/training_small.json",
    "configs/model.json",
])
config.save(dir_output)
vocab = Vocab(config)

train_set = DataGenerator(
    path_formulas=config.path_formulas_train,
    dir_images=config.dir_images_train,
    img_prepro=greyscale,
    max_iter=config.max_iter,
    bucket=config.bucket_train,
    path_matching=config.path_matching_train,
    max_len=config.max_length_formula,
    form_prepro=vocab.form_prepro
)
val_set = DataGenerator(
    path_formulas=config.path_formulas_val,
    dir_images=config.dir_images_val,
    img_prepro=greyscale,
    max_iter=config.max_iter,
    bucket=config.bucket_val,
    path_matching=config.path_matching_val,
    max_len=config.max_length_formula,
    form_prepro=vocab.form_prepro
)

feed_dicts = []
batch_size = config.batch_size
for i, (_img, _formula) in enumerate(minibatches(train_set, batch_size)):
    fd = {
        inputs: _img,
        dropout: 0.2,
        training: True,
        learning_rate: 0.02,
    }
    if _formula is not None:
        _formula, _formula_length = pad_batch_formulas(
            _formula,
            vocab.id_pad,
            vocab.id_end
        )
        fd[formula] = _formula
        fd[formula_length] = _formula_length
    feed_dicts.append(fd)



fd = feed_dicts[0]
sess.run([op_train, loss], feed_dict=fd)


def _run_epoch(self, config, train_set, val_set, epoch, lr_schedule):
    """Performs an epoch of training

    Args:
        config: Config instance
        train_set: Dataset instance
        val_set: Dataset instance
        epoch: (int) id of the epoch, starting at 0
        lr_schedule: LRSchedule instance that takes care of learning proc

    Returns:
        score: (float) model will select weights that achieve the highest
            score

    """
    # logging
    batch_size = config.batch_size
    nbatches = (len(train_set) + batch_size - 1) // batch_size
    # prog = Progbar(nbatches)

    # iterate over dataset
    for i, (img, formula) in enumerate(minibatches(train_set, batch_size)):
        # get feed dict
        fd = self._get_feed_dict(img, training=True, formula=formula,
                lr=lr_schedule.lr, dropout=config.dropout)

        # update step
        _, loss_eval = self.sess.run([self.train_op, self.loss],
                feed_dict=fd)
        # prog.update(i + 1, [("loss", loss_eval), ("perplexity",
        #         np.exp(loss_eval)), ("lr", lr_schedule.lr)])

        # update learning rate
        lr_schedule.update(batch_no=epoch*nbatches + i)

    # logging
    self.logger.info("- Training: {}".format(prog.info))

    # evaluation
    config_eval = Config({"dir_answers": self._dir_output + "formulas_val/",
            "batch_size": config.batch_size})
    scores = self.evaluate(config_eval, val_set)
    score = scores[config.metric_val]
    lr_schedule.update(score=score)

    return score


# exit()
# clear;python

# from tf_utils_190108.tf_builder import *
# from tf_utils_190108.tf_controller import *
# import sys, os, time, datetime

# a = tf.tuple(
#     [
#         tf.placeholder(tf.float32, [None, 80, 32])
#         for _ in range(4)
#     ],
#     name='input_tuple',
# )
# c = tf.nn.rnn_cell.LSTMCell(24)
# d = tf.nn.dynamic_rnn(
#     cell=c,
#     inputs=a,
#     sequence_length=None,
#     initial_state=None,
#     dtype=tf.float32,
# )
# b = TF_RNN(
#     feed=a,
#     bidirectional=True,
#     cell_type='lstm',
#     cell_count=[24,12],
#     sequence_length=None,
# )


