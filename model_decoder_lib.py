
import tensorflow as tf
import collections

from tensorflow.nn.rnn_cell import RNNCell, LSTMCell

from tensorflow.contrib.framework import nest
from tensorflow.contrib.ops import rnn


class AttentionMechanism(object):
    """Class to compute attention over an image"""

    def __init__(self, img, dim_e, tiles=1):
        """Stores the image under the right shape.

        We loose the H, W dimensions and merge them into a single
        dimension that corresponds to "regions" of the image.

        Args:
            img: (tf.Tensor) image
            dim_e: (int) dimension of the intermediary vector used to
                compute attention
            tiles: (int) default 1, input to context h may have size
                    (tile * batch_size, ...)

        """
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
        self._dim_e      = dim_e
        self._tiles      = tiles
        self._scope_name = "att_mechanism"

        # attention vector over the image
        self._att_img = tf.layers.dense(
            inputs=self._img,
            units=self._dim_e,
            use_bias=False,
            name="att_img")


    def context(self, h):
        """Computes attention

        Args:
            h: (batch_size, num_units) hidden state

        Returns:
            c: (batch_size, channels) context vector

        """
        with tf.variable_scope(self._scope_name):
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
            att_h = tf.layers.dense(inputs=h, units=self._dim_e, use_bias=False)

            # sums the two contributions
            att_h = tf.expand_dims(att_h, axis=1)
            att = tf.tanh(att_img + att_h)

            # computes scalar product with beta vector
            # works faster with a matmul than with a * and a tf.reduce_sum
            att_beta = tf.get_variable("att_beta", shape=[self._dim_e, 1],
                    dtype=tf.float32)
            att_flat = tf.reshape(att, shape=[-1, self._dim_e])
            e = tf.matmul(att_flat, att_beta)
            e = tf.reshape(e, shape=[-1, self._n_regions])

            # compute weights
            a = tf.nn.softmax(e)
            a = tf.expand_dims(a, axis=-1)
            c = tf.reduce_sum(a * img, axis=1)

            return c


    def initial_cell_state(self, cell):
        """Returns initial state of a cell computed from the image

        Assumes cell.state_type is an instance of named_tuple.
        Ex: LSTMStateTuple

        Args:
            cell: (instance of RNNCell) must define _state_size

        """
        _states_0 = []
        for hidden_name in cell._state_size._fields:
            hidden_dim = getattr(cell._state_size, hidden_name)
            h = self.initial_state(hidden_name, hidden_dim)
            _states_0.append(h)

        initial_state_cell = type(cell.state_size)(*_states_0)

        return initial_state_cell


    def initial_state(self, name, dim):
        """Returns initial state of dimension specified by dim"""
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            img_mean = tf.reduce_mean(self._img, axis=1)
            W = tf.get_variable("W_{}_0".format(name), shape=[self._n_channels, dim])
            b = tf.get_variable("b_{}_0".format(name), shape=[dim])
            h = tf.tanh(tf.matmul(img_mean, W) + b)

            return h



AttentionState = collections.namedtuple("AttentionState", ("cell_state", "o"))

class AttentionCell(RNNCell):
    def __init__(self, cell, attention_mechanism, dropout, attn_cell_config,
        num_proj, dtype=tf.float32):
        """
        Args:
            cell: (RNNCell)
            attention_mechanism: (AttentionMechanism)
            dropout: (tf.float)
            attn_cell_config: (dict) hyper params

        """
        # variables and tensors
        self._cell                = cell
        self._attention_mechanism = attention_mechanism
        self._dropout             = dropout

        # hyperparameters and shapes
        self._n_channels     = self._attention_mechanism._n_channels
        self._dim_e          = attn_cell_config["dim_e"]
        self._dim_o          = attn_cell_config["dim_o"]
        self._num_units      = attn_cell_config["num_units"]
        self._dim_embeddings = attn_cell_config["dim_embeddings"]
        self._num_proj       = num_proj
        self._dtype          = dtype

        # for RNNCell
        self._state_size = AttentionState(self._cell._state_size, self._dim_o)


    @property
    def state_size(self):
        return self._state_size


    @property
    def output_size(self):
        return self._num_proj


    @property
    def output_dtype(self):
        return self._dtype


    def initial_state(self):
        """Returns initial state for the lstm"""
        initial_cell_state = self._attention_mechanism.initial_cell_state(self._cell)
        initial_o          = self._attention_mechanism.initial_state("o", self._dim_o)

        return AttentionState(initial_cell_state, initial_o)


    def step(self, embedding, attn_cell_state):
        """
        Args:
            embedding: shape = (batch_size, dim_embeddings) embeddings
                from previous time step
            attn_cell_state: (AttentionState) state from previous time step

        """
        prev_cell_state, o = attn_cell_state

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # compute new h
            x                     = tf.concat([embedding, o], axis=-1)
            new_h, new_cell_state = self._cell.__call__(x, prev_cell_state)
            new_h = tf.nn.dropout(new_h, self._dropout)

            # compute attention
            c = self._attention_mechanism.context(new_h)

            # compute o
            o_W_c = tf.get_variable("o_W_c", dtype=tf.float32,
                    shape=(self._n_channels, self._dim_o))
            o_W_h = tf.get_variable("o_W_h", dtype=tf.float32,
                    shape=(self._num_units, self._dim_o))

            new_o = tf.tanh(tf.matmul(new_h, o_W_h) + tf.matmul(c, o_W_c))
            new_o = tf.nn.dropout(new_o, self._dropout)

            y_W_o = tf.get_variable("y_W_o", dtype=tf.float32,
                    shape=(self._dim_o, self._num_proj))
            logits = tf.matmul(new_o, y_W_o)

            # new Attn cell state
            new_state = AttentionState(new_cell_state, new_o)

            return logits, new_state


    def __call__(self, inputs, state):
        """
        Args:
            inputs: the embedding of the previous word for training only
            state: (AttentionState) (h, o) where h is the hidden state and
                o is the vector used to make the prediction of
                the previous word

        """
        new_output, new_state = self.step(inputs, state)

        return (new_output, new_state)



class DecoderOutput(collections.namedtuple("DecoderOutput", ("logits", "ids"))):
    pass


class GreedyDecoderCell(object):

    def __init__(self, embeddings, attention_cell, batch_size, start_token,
        end_token):

        self._embeddings = embeddings
        self._attention_cell = attention_cell
        self._dim_embeddings = embeddings.shape[-1].value
        self._batch_size = batch_size
        self._start_token = start_token
        self._end_token = end_token


    @property
    def output_dtype(self):
        """for the custom dynamic_decode for the TensorArray of results"""
        return DecoderOutput(logits=self._attention_cell.output_dtype,
                ids=tf.int32)


    @property
    def final_output_dtype(self):
        """For the finalize method"""
        return self.output_dtype


    def initial_state(self):
        """Return initial state for the lstm"""
        return self._attention_cell.initial_state()


    def initial_inputs(self):
        """Returns initial inputs for the decoder (start token)"""
        return tf.tile(tf.expand_dims(self._start_token, 0),
            multiples=[self._batch_size, 1])


    def initialize(self):
        initial_state = self.initial_state()
        initial_inputs = self.initial_inputs()
        initial_finished = tf.zeros(shape=[self._batch_size], dtype=tf.bool)
        return initial_state, initial_inputs, initial_finished


    def step(self, time, state, embedding, finished):
        # next step of attention cell
        logits, new_state = self._attention_cell.step(embedding, state)

        # get ids of words predicted and get embedding
        new_ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        new_embedding = tf.nn.embedding_lookup(self._embeddings, new_ids)

        # create new state of decoder
        new_output = DecoderOutput(logits, new_ids)

        new_finished = tf.logical_or(finished, tf.equal(new_ids,
                self._end_token))

        return (new_output, new_state, new_embedding, new_finished)


    def finalize(self, final_outputs, final_state):
        return final_outputs



def transpose_batch_time(t):
    if t.shape.ndims == 2:
        return tf.transpose(t, [1, 0])
    elif t.shape.ndims == 3:
        return tf.transpose(t, [1, 0, 2])
    elif t.shape.ndims == 4:
        return tf.transpose(t, [1, 0, 2, 3])
    else:
        raise NotImplementedError


def dynamic_decode(decoder_cell, maximum_iterations):
    """Similar to dynamic_rnn but to decode

    Args:
        decoder_cell: (instance of DecoderCell) with step method
        maximum_iterations: (int)

    """
    try:
        maximum_iterations = tf.convert_to_tensor(maximum_iterations,
                dtype=tf.int32)
    except ValueError:
        pass

    # create TA for outputs by mimicing the structure of decodercell output
    def create_ta(d):
        return tf.TensorArray(dtype=d, size=0, dynamic_size=True)

    initial_time = tf.constant(0, dtype=tf.int32)
    initial_outputs_ta = nest.map_structure(create_ta,
            decoder_cell.output_dtype)
    initial_state, initial_inputs, initial_finished = decoder_cell.initialize()

    def condition(time, unused_outputs_ta, unused_state, unused_inputs,
        finished):
        return tf.logical_not(tf.reduce_all(finished))

    def body(time, outputs_ta, state, inputs, finished):
        new_output, new_state, new_inputs, new_finished = decoder_cell.step(
            time, state, inputs, finished)

        outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                      outputs_ta, new_output)

        new_finished = tf.logical_or(
            tf.greater_equal(time, maximum_iterations),
            new_finished)

        return (time + 1, outputs_ta, new_state, new_inputs, new_finished)

    with tf.variable_scope("rnn_dynamic_decode", reuse=tf.AUTO_REUSE):
        res = tf.while_loop(
            condition,
            body,
            loop_vars=[initial_time, initial_outputs_ta, initial_state,
                       initial_inputs, initial_finished],
            back_prop=False)

    # get final outputs and states
    final_outputs_ta, final_state = res[1], res[2]

    # unfold and stack the structure from the nested tas
    final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)

    # finalize the computation from the decoder cell
    final_outputs = decoder_cell.finalize(final_outputs, final_state)

    # transpose the final output
    final_outputs = nest.map_structure(transpose_batch_time, final_outputs)

    return final_outputs, final_state



def embedding_initializer():
    """Returns initializer for embeddings"""
    def _initializer(shape, dtype, partition_info=None):
        E = tf.random_uniform(shape, minval=-1.0, maxval=1.0, dtype=dtype)
        E = tf.nn.l2_normalize(E, -1)
        return E

    return _initializer



def get_embeddings(formula, E, dim, start_token, batch_size):
    """Returns the embedding of the n-1 first elements in the formula concat
    with the start token

    Args:
        formula: (tf.placeholder) tf.uint32
        E: tf.Variable (matrix)
        dim: (int) dimension of embeddings
        start_token: tf.Variable
        batch_size: tf variable extracted from placeholder

    Returns:
        embeddings_train: tensor

    """
    formula_ = tf.nn.embedding_lookup(E, formula)
    start_token_ = tf.reshape(start_token, [1, 1, dim])
    start_tokens = tf.tile(start_token_, multiples=[batch_size, 1, 1])
    embeddings = tf.concat([start_tokens, formula_[:, :-1, :]], axis=1)

    return embeddings


class BeamSearchDecoderCellState(collections.namedtuple(
        "BeamSearchDecoderCellState", ("cell_state", "log_probs"))):
    """State of the Beam Search decoding

    cell_state: shape = structure of [batch_size, beam_size, ?]
        cell state for all the hypotheses
    embedding: shape = [batch_size, beam_size, embedding_size]
        embeddings of the previous time step for each hypothesis
    log_probs: shape = [batch_size, beam_size]
        log_probs of the hypotheses
    finished: shape = [batch_size, beam_size]
        boolean to know if one beam hypothesis has reached token id_end

    """
    pass


class BeamSearchDecoderOutput(collections.namedtuple(
        "BeamSearchDecoderOutput", ("logits", "ids", "parents"))):
    """Stores the logic for the beam search decoding

    logits: shape = [batch_size, beam_size, vocab_size]
        scores before softmax of the beam search hypotheses
    ids: shape = [batch_size, beam_size]
        ids of the best words at this time step
    parents: shape = [batch_size, beam_size]
        ids of the beam index from previous time step

    """
    pass


class BeamSearchDecoderCell(object):

    def __init__(self, embeddings, cell, batch_size, start_token, end_token,
            beam_size=5, div_gamma=1, div_prob=0):
        """Initializes parameters for Beam Search

        Args:
            embeddings: (tf.Variable) shape = (vocab_size, embedding_size)
            cell: instance of Cell that defines a step function, etc.
            batch_size: tf.int extracted with tf.Shape or int
            start_token: id of start token
            end_token: int, id of the end token
            beam_size: int, size of the beam
            div_gamma: float, amount of penalty to add to beam hypo for
                diversity. Coefficient of penaly will be log(div_gamma).
                Use value between 0 and 1. (1 means no penalty)
            div_prob: only apply div penalty with probability div_prob.
                div_prob = 0. means never apply penalty

        """

        self._embeddings = embeddings
        self._cell = cell
        self._dim_embeddings = embeddings.shape[-1].value
        self._batch_size = batch_size
        self._start_token = start_token
        self._beam_size  = beam_size
        self._end_token = end_token
        self._vocab_size = embeddings.shape[0].value
        self._div_gamma = float(div_gamma)
        self._div_prob = float(div_prob)


    @property
    def output_dtype(self):
        """Needed for custom dynamic_decode for the TensorArray of results"""
        return BeamSearchDecoderOutput(logits=self._cell.output_dtype,
                ids=tf.int32, parents=tf.int32)


    @property
    def final_output_dtype(self):
        """For the finalize method"""
        return DecoderOutput(logits=self._cell.output_dtype, ids=tf.int32)


    @property
    def state_size(self):
        return BeamSearchDecoderOutput(
                logits=tf.TensorShape([self._beam_size, self._vocab_size]),
                ids=tf.TensorShape([self._beam_size]),
                parents=tf.TensorShape([self._beam_size]))


    @property
    def final_output_size(self):
        return DecoderOutput(
            logits=tf.TensorShape([self._beam_size, self._vocab_size]),
            ids=tf.TensorShape([self._beam_size])
        )


    def initial_state(self):
        """Returns initial state for the decoder"""
        # cell initial state
        cell_state = self._cell.initial_state()
        cell_state = nest.map_structure(
            lambda t: tile_beam(t, self._beam_size),
            cell_state
        )

        # prepare other initial states
        log_probs =  tf.zeros(
            [self._batch_size, self._beam_size],
            dtype=self._cell.output_dtype
        )

        return BeamSearchDecoderCellState(cell_state, log_probs)


    def initial_inputs(self):
        return tf.tile(
            tf.reshape(
                self._start_token,
                [1, 1, self._dim_embeddings]
            ),
            multiples=[self._batch_size, self._beam_size, 1]
        )


    def initialize(self):
        initial_state = self.initial_state()
        initial_inputs = self.initial_inputs()
        initial_finished = tf.zeros(
            shape=[self._batch_size, self._beam_size],
            dtype=tf.bool
        )
        return initial_state, initial_inputs, initial_finished


    def step(self, time, state, embedding, finished):
        """
        Args:
            time: tensorf or int
            embedding: shape [batch_size, beam_size, d]
            state: structure of shape [bach_size, beam_size, ...]
            finished: structure of shape [batch_size, beam_size, ...]

        """
        # merge batch and beam dimension before callling step of cell
        cell_state = nest.map_structure(merge_batch_beam, state.cell_state)
        embedding = merge_batch_beam(embedding)

        # compute new logits
        logits, new_cell_state = self._cell.step(embedding, cell_state)

        # split batch and beam dimension before beam search logic
        new_logits = split_batch_beam(logits, self._beam_size)
        new_cell_state = nest.map_structure(
                lambda t: split_batch_beam(t, self._beam_size), new_cell_state)

        # compute log probs of the step
        # shape = [batch_size, beam_size, vocab_size]
        step_log_probs = tf.nn.log_softmax(new_logits)
        # shape = [batch_size, beam_size, vocab_size]
        step_log_probs = mask_probs(step_log_probs, self._end_token, finished)
        # shape = [batch_size, beam_size, vocab_size]
        log_probs = tf.expand_dims(state.log_probs, axis=-1) + step_log_probs
        log_probs = add_div_penalty(log_probs, self._div_gamma, self._div_prob,
                self._batch_size, self._beam_size, self._vocab_size)

        # compute the best beams
        # shape =  (batch_size, beam_size * vocab_size)
        log_probs_flat = tf.reshape(log_probs,
                [self._batch_size, self._beam_size * self._vocab_size])
        # if time = 0, consider only one beam, otherwise beams are equal
        log_probs_flat = tf.cond(time > 0, lambda: log_probs_flat,
                lambda: log_probs[:, 0])
        new_probs, indices = tf.nn.top_k(log_probs_flat, self._beam_size)

        # of shape [batch_size, beam_size]
        new_ids = indices % self._vocab_size
        new_parents = indices // self._vocab_size

        # get ids of words predicted and get embedding
        new_embedding = tf.nn.embedding_lookup(self._embeddings, new_ids)

        # compute end of beam
        finished = gather_helper(finished, new_parents,
                self._batch_size, self._beam_size)
        new_finished = tf.logical_or(finished,
                tf.equal(new_ids, self._end_token))

        new_cell_state = nest.map_structure(
                lambda t: gather_helper(t, new_parents, self._batch_size,
                self._beam_size), new_cell_state)


        # create new state of decoder
        new_state  = BeamSearchDecoderCellState(cell_state=new_cell_state,
                log_probs=new_probs)

        new_output = BeamSearchDecoderOutput(logits=new_logits, ids=new_ids,
                parents=new_parents)

        return (new_output, new_state, new_embedding, new_finished)


    def finalize(self, final_outputs, final_state):
        """
        Args:
            final_outputs: structure of tensors of shape
                    [time dimension, batch_size, beam_size, d]
            final_state: instance of BeamSearchDecoderOutput

        Returns:
            [time, batch, beam, ...] structure of Tensor

        """
        # reverse the time dimension
        maximum_iterations = tf.shape(final_outputs.ids)[0]
        final_outputs = nest.map_structure(lambda t: tf.reverse(t, axis=[0]),
                final_outputs)

        # initial states
        def create_ta(d):
            return tf.TensorArray(dtype=d, size=maximum_iterations)

        initial_time = tf.constant(0, dtype=tf.int32)
        initial_outputs_ta = nest.map_structure(create_ta,
                self.final_output_dtype)
        initial_parents = tf.tile(
                tf.expand_dims(tf.range(self._beam_size), axis=0),
                multiples=[self._batch_size, 1])

        def condition(time, outputs_ta, parents):
            return tf.less(time, maximum_iterations)

        # beam search decoding cell
        def body(time, outputs_ta, parents):
            # get ids, logits and parents predicted at time step by decoder
            input_t = nest.map_structure(lambda t: t[time], final_outputs)

            # extract the entries corresponding to parents
            new_state = nest.map_structure(
                    lambda t: gather_helper(t, parents, self._batch_size,
                    self._beam_size), input_t)

            # create new output
            new_output = DecoderOutput(logits=new_state.logits,
                    ids=new_state.ids)

            # write beam ids
            outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                    outputs_ta, new_output)

            return (time + 1), outputs_ta, parents

        res = tf.while_loop(
                condition,
                body,
                loop_vars=[initial_time, initial_outputs_ta, initial_parents],
                back_prop=False)

        # unfold and stack the structure from the nested tas
        final_outputs = nest.map_structure(lambda ta: ta.stack(), res[1])

        # reverse time step
        final_outputs = nest.map_structure(lambda t: tf.reverse(t, axis=[0]),
                final_outputs)

        return DecoderOutput(logits=final_outputs.logits, ids=final_outputs.ids)


def sample_bernoulli(p, s):
    """Samples a boolean tensor with shape = s according to bernouilli"""
    return tf.greater(p, tf.random_uniform(s))


def add_div_penalty(log_probs, div_gamma, div_prob, batch_size, beam_size,
        vocab_size):
    """Adds penalty to beam hypothesis following this paper by Li et al. 2016
    "A Simple, Fast Diverse Decoding Algorithm for Neural Generation"

    Args:
        log_probs: (tensor of floats)
            shape = (batch_size, beam_size, vocab_size)
        div_gamma: (float) diversity parameter
        div_prob: (float) adds penalty with proba div_prob

    """
    if div_gamma is None or div_prob is None: return log_probs
    if div_gamma == 1. or div_prob == 0.: return log_probs

    # 1. get indices that would sort the array
    top_probs, top_inds = tf.nn.top_k(log_probs, k=vocab_size, sorted=True)
    # 2. inverse permutation to get rank of each entry
    top_inds = tf.reshape(top_inds, [-1, vocab_size])
    index_rank = tf.map_fn(tf.invert_permutation, top_inds, back_prop=False)
    index_rank = tf.reshape(index_rank, shape=[batch_size, beam_size,
            vocab_size])
    # 3. compute penalty
    penalties = tf.log(div_gamma) * tf.cast(index_rank, log_probs.dtype)
    # 4. only apply penalty with some probability
    apply_penalty = tf.cast(
            sample_bernoulli(div_prob, [batch_size, beam_size, vocab_size]),
            penalties.dtype)
    penalties *= apply_penalty

    return log_probs + penalties


def merge_batch_beam(t):
    """
    Args:
        t: tensor of shape [batch_size, beam_size, ...]
            whose dimensions after beam_size must be statically known

    Returns:
        t: tensorf of shape [batch_size * beam_size, ...]

    """
    batch_size = tf.shape(t)[0]
    beam_size = t.shape[1].value

    if t.shape.ndims == 2:
        return tf.reshape(t, [batch_size*beam_size, 1])
    elif t.shape.ndims == 3:
        return tf.reshape(t, [batch_size*beam_size, t.shape[-1].value])
    elif t.shape.ndims == 4:
        return tf.reshape(t, [batch_size*beam_size, t.shape[-2].value,
                t.shape[-1].value])
    else:
        raise NotImplementedError


def split_batch_beam(t, beam_size):
    """
    Args:
        t: tensorf of shape [batch_size*beam_size, ...]

    Returns:
        t: tensor of shape [batch_size, beam_size, ...]

    """
    if t.shape.ndims == 1:
        return tf.reshape(t, [-1, beam_size])
    elif t.shape.ndims == 2:
        return tf.reshape(t, [-1, beam_size, t.shape[-1].value])
    elif t.shape.ndims == 3:
        return tf.reshape(t, [-1, beam_size, t.shape[-2].value,
                t.shape[-1].value])
    else:
        raise NotImplementedError


def tile_beam(t, beam_size):
    """
    Args:
        t: tensor of shape [batch_size, ...]

    Returns:
        t: tensorf of shape [batch_size, beam_size, ...]

    """
    # shape = [batch_size, 1 , x]
    t = tf.expand_dims(t, axis=1)
    if t.shape.ndims == 2:
        multiples = [1, beam_size]
    elif t.shape.ndims == 3:
        multiples = [1, beam_size, 1]
    elif t.shape.ndims == 4:
        multiples = [1, beam_size, 1, 1]

    return tf.tile(t, multiples)


def mask_probs(probs, end_token, finished):
    """
    Args:
        probs: tensor of shape [batch_size, beam_size, vocab_size]
        end_token: (int)
        finished: tensor of shape [batch_size, beam_size], dtype = tf.bool
    """
    # one hot of shape [vocab_size]
    vocab_size = probs.shape[-1].value
    one_hot = tf.one_hot(end_token, vocab_size, on_value=0.,
            off_value=probs.dtype.min, dtype=probs.dtype)
    # expand dims of shape [batch_size, beam_size, 1]
    finished = tf.expand_dims(tf.cast(finished, probs.dtype), axis=-1)

    return (1. - finished) * probs + finished * one_hot


def gather_helper(t, indices, batch_size, beam_size):
    """
    Args:
        t: tensor of shape = [batch_size, beam_size, d]
        indices: tensor of shape = [batch_size, beam_size]

    Returns:
        new_t: tensor w shape as t but new_t[:, i] = t[:, new_parents[:, i]]

    """
    range_  = tf.expand_dims(tf.range(batch_size) * beam_size, axis=1)
    indices = tf.reshape(indices + range_, [-1])
    output  = tf.gather(
        tf.reshape(t, [batch_size*beam_size, -1]),
        indices)

    if t.shape.ndims == 2:
        return tf.reshape(output, [batch_size, beam_size])

    elif t.shape.ndims == 3:
        d = t.shape[-1].value
        return tf.reshape(output, [batch_size, beam_size, d])



