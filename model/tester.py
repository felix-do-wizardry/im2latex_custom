import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


# from utils.general import Config, Progbar, minibatches
# from utils.image import pad_batch_images
# from utils.text import pad_batch_formulas
# from evaluation.text import score_files, write_answers, truncate_end


from encoder import Encoder
from decoder import Decoder
from base import BaseModel

from utils.general import Config



@click.option('--data', default="configs/data_small.json",
        help='Path to data json config')
@click.option('--vocab', default="configs/vocab_small.json",
        help='Path to vocab json config')
@click.option('--training', default="configs/training_small.json",
        help='Path to training json config')
@click.option('--model', default="configs/model.json",
        help='Path to model json config')
# @click.option('--output', default="results/small/",
#         help='Dir for results and model weights')


_config_args = [
    'configs/data_small.json',
    'configs/vocab_small.json',
    'configs/training_small.json',
    'configs/model.json',
    # 'results/small/',
]

_config = Config(_config_args)
encoder = Encoder(_config)
decoder = Decoder(_config, _vocab.n_tok, _vocab.id_end)
