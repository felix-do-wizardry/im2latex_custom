
import tensorflow as tf
import numpy as np
import collections

from model_encoder import *
from model_decoder import *

import click
from model.utils.data_generator import DataGenerator
from model.utils.text import Vocab, pad_batch_formulas
from model.utils.image import greyscale, pad_batch_images
from model.utils.general import Config, Progbar, minibatches
# from model.evaluation.text import score_files, write_answers, truncate_end


# _attn_cell_config = {
#     'cell_type': 'lstm',
#     'num_units': 12,
#     'dim_e'    : 14,
#     'dim_o'    : 16,
#     'dim_embeddings': 32,
# }

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


start_token = tf.get_variable(
    "start_token",
    dtype=tf.float32,
    shape=[config.attn_cell_config['dim_embeddings']],
    initializer=embedding_initializer(),
)
E = tf.get_variable(
    "E",
    initializer=embedding_initializer(),
    shape=[vocab.n_tok, config.attn_cell_config['dim_embeddings']],
    dtype=tf.float32
)

# inputs
inputs = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='inputs')
# labels
formula = tf.placeholder(tf.int32, shape=[None, None], name='labels')
formula_length = tf.placeholder(tf.int32, shape=(None, ), name='labels_length')

learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
dropout = tf.placeholder(tf.float32, shape=(), name='dropout')
training = tf.placeholder(tf.bool, shape=(), name="training")

# config cnn (out_dim, batch_norm, pooling)
config_cnn_set = [
    (12, True, None),
    (16, True, 4),
    (20, True, [2,1]),
    (24, False, [1,2]),
    (28, True, None),
    (32, False, 2),
    (36, False, 2),
]
# get encoder and decoder
encoder = Encoder(config, config_cnn_set)
decoder = Decoder(config, vocab)

# get output from encoder and decoder
encoder_output = encoder(inputs)
pred_train, pred_test = decoder(
    encoder_output,
    formula,
    formula_length,
    E,
    start_token,
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

optimizer_name = 'adam'
optimizer_func = tf.train.AdamOptimizer
if optimizer_name.lower() in optimizer_dict:
    optimizer_func = optimizer_dict[optimizer_name]

optimizer = optimizer_func(learning_rate)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    op_train = optimizer.minimize(loss)


# for tensorboard
timecode = time.strftime("%y%m%d_%H%M%S", time.gmtime())
fileWriter = tf.summary.FileWriter('tensorboard/' + timecode, tf.get_default_graph())
fileWriter.flush()

print('graph saved to tensorboard')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

feed_dicts = []
batch_size = config.batch_size
for i, (_img, _formula) in enumerate(minibatches(train_set, batch_size)):
    fd = {
        inputs: _img,
        dropout: 0.2,
        training: True,
        learning_rate: 0.0001,
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

# all_img = []
# all_formula = []
# for i, (_img, _formula) in enumerate(minibatches(train_set, batch_size)):
#     all_img.append(_img)
#     if _formula is not None:
#         _formula, _formula_length = pad_batch_formulas(
#             _formula,
#             vocab.id_pad,
#             vocab.id_end
#         )
#     all_formula.append(_formula)


# np.save('np_formula', np.array(all_formula))
# np.save('np_img', np.array(all_img))

print('start training')

# fd = feed_dicts[0]
sess.run(tf.global_variables_initializer())
run_loss = sess.run(loss, fd)
# sess.run(op_train, fd)
# print(sess.run(loss, fd))
print('initial loss: {}', run_loss)
# print('DONE TESTING! EXITING')
# sess.run([op_train, loss], feed_dict=fd)

model_timecode = time.strftime("%y%m%d_%H%M%S", time.gmtime())
saver = tf.train.Saver()

def start_training(epoch=20, model_save_iter=10):
    for i in range(epoch):
        _ = [sess.run(op_train, fd) for fd in feed_dicts]
        run_loss = np.mean([sess.run(loss, fd) for fd in feed_dicts])
        print('epoch: {}, loss: {}'.format(i, run_loss))
        if (i + 1) % model_save_iter == 0:
            save_path = saver.save(
                sess,
                '/saved_models/{}.ckpt'.format(model_timecode)
            )
            print('model checkpoint saved at {}'.format(save_path))
    return True


start_training()

print('Done. Exiting!')

# print(sess.run(pred_test, fd))

