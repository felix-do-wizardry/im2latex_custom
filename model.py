
import tensorflow as tf
import numpy as np
import collections

from model_encoder import *
from model_decoder import *


config = {
    'max_length_formula': 32,
    'n_tok': 20,
    'batch_size': 20,
    'dropout': 0.01,
}
attn_cell_config = {
    'cell_type': 'lstm',
    'num_units': 32,
    'dim_e'    : 40,
    'dim_o'    : 48,
    'dim_embeddings': 60,
}
vocab = {
    'id_end': 0,
}


start_token = tf.get_variable(
    "start_token",
    dtype=tf.float32,
    shape=[attn_cell_config['dim_embeddings']],
    initializer=embedding_initializer(),
)
E = tf.get_variable(
    "E",
    initializer=embedding_initializer(),
    shape=[config['n_tok'], attn_cell_config['dim_embeddings']],
    dtype=tf.float32
)

# inputs
inputs = tf.placeholder(tf.float32, shape=[None, 128, 1280, 1], name='inputs')
# labels
formula = tf.placeholder(tf.int32, shape=[None, config['n_tok']], name='labels')
formula_length = tf.placeholder(tf.int32, shape=(None, ), name='labels_length')


# encoder_output
encoder_output = encoder(inputs)
pred_train, pred_test = decoder(
    encoder_output,
    attn_cell_config,
    config,
    E,
    start_token,
    vocab,
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

# for tensorboard
timecode = time.strftime("%y%m%d_%H%M%S", time.gmtime())
fileWriter = tf.summary.FileWriter('tensorboard/' + timecode, tf.get_default_graph())
fileWriter.flush()
# tf.summary.scalar("loss", loss)

print("DONE!!!!!! TADA")


# def _add_loss_op(self):
#     """Defines self.loss"""
#     losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
#         logits=self.pred_train, labels=self.formula)

#     mask = tf.sequence_mask(self.formula_length)
#     losses = tf.boolean_mask(losses, mask)

#     # loss for training
#     self.loss = tf.reduce_mean(losses)

#     # # to compute perplexity for test
#     self.ce_words = tf.reduce_sum(losses) # sum of CE for each word
#     self.n_words = tf.reduce_sum(self.formula_length) # number of words

#     # for tensorboard
#     tf.summary.scalar("loss", self.loss)



# def _run_epoch(self, config, train_set, val_set, epoch, lr_schedule):
#     """Performs an epoch of training

#     Args:
#         config: Config instance
#         train_set: Dataset instance
#         val_set: Dataset instance
#         epoch: (int) id of the epoch, starting at 0
#         lr_schedule: LRSchedule instance that takes care of learning proc

#     Returns:
#         score: (float) model will select weights that achieve the highest
#             score

#     """
#     # logging
#     batch_size = config.batch_size
#     nbatches = (len(train_set) + batch_size - 1) // batch_size
#     prog = Progbar(nbatches)

#     # iterate over dataset
#     for i, (img, formula) in enumerate(minibatches(train_set, batch_size)):
#         # get feed dict
#         fd = self._get_feed_dict(img, training=True, formula=formula,
#                 lr=lr_schedule.lr, dropout=config.dropout)

#         # update step
#         _, loss_eval = self.sess.run([self.train_op, self.loss],
#                 feed_dict=fd)
#         prog.update(i + 1, [("loss", loss_eval), ("perplexity",
#                 np.exp(loss_eval)), ("lr", lr_schedule.lr)])

#         # update learning rate
#         lr_schedule.update(batch_no=epoch*nbatches + i)

#     # logging
#     self.logger.info("- Training: {}".format(prog.info))

#     # evaluation
#     config_eval = Config({"dir_answers": self._dir_output + "formulas_val/",
#             "batch_size": config.batch_size})
#     scores = self.evaluate(config_eval, val_set)
#     score = scores[config.metric_val]
#     lr_schedule.update(score=score)

#     return score



