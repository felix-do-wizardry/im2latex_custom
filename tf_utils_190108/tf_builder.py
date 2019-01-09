import tensorflow as tf
import numpy as np
import time, timeit
from sys import stdout

TF_ALL_BUILT_LAYERS = {}
STATE_TRAINING = tf.get_variable(
    'state_training',
    initializer=True,
    trainable=False,
)
# [n.name for n in tf.get_default_graph().as_graph_def().node]

def is_tf_tensor(v=None):
    return isinstance(v, (tf.Tensor))

def is_tf_variable(v=None):
    return isinstance(v, (tf.Variable))

def is_tf_tensor_or_variable(v=None, rank_range=None):
    return isinstance(v, (tf.Tensor, tf.Variable))

def is_tf_operation(v=None):
    return isinstance(v, (tf.Operation))

def is_string(v=None):
    return isinstance(v, (str))

def is_int(v=None):
    try:
        mod = v % 1
        if mod == 0:
            return True
    except Exception:
        pass
    return False

def is_array(v=None, len_min=None, len_max=None):
    if not isinstance(v, (list, tuple, np.ndarray)):
        return False
    try:
        if len(v) < len_min:
            return False
    except Exception:
        pass
    try:
        if len(v) > len_max:
            return False
    except Exception:
        pass
    return True
    
def is_in_range(v=None, v_min=None, v_max=None):
    try:
        if v > 0:
            pass
    except Exception:
        return False
    try:
        if v < v_min:
            return False
    except Exception:
        pass
    try:
        if v > v_max:
            return False
    except Exception:
        pass
    return True


class TF_Base():
    
    """
    [BASE]
    feed
    name
    dropout
    auto_build
    l2_loss_scale
    activator
    activator_kwargs
    out_dim
    batch_norm
    """
    
    kwargs_def_base = {
        'feed': [None],
        'name': ['base', is_string],
        'dropout': [0.0],
        'auto_build': [[True, False], None],
        'l2_loss_scale': [0],
        'activator': [tf.nn.relu],
        'activator_kwargs': [{}],
        'out_dim': [1],
        'batch_norm': [[False, True], None],
    }
    kwargs_def_custom = {}
    
    # locked / internal
    def __init__(self, *args, **kwargs):
        self.kwargs_def = {
            **self.kwargs_def_base,
            **self.kwargs_def_custom,
        }
        self.kwargs = {**kwargs}
        if len(args) >= 1:
            # dont init
            kwargs.update({'auto_build': False})
            return None
        self.kwargs = {
            **{k: None for k in self.kwargs_def},
            **self.kwargs,
        }
        for k in self.kwargs_def:
            self.arg_process(
                name       = k,
                value      = self.kwargs[k],
                kwargs_def = self.kwargs_def[k],
            )
        self.name_checked = False
        _ = self.layer_check()
        if not self.name_checked:
            return None
        self.scope = self.name + '/'
        self.count = 0
        self.feed_accepted = False
        self.feed_received = []
        self.feed_processed = []
        self.feed_rank = 1
        self.feed_shape = [1]
        self.calc = []
        self.calc_current = {}
        self.prop_accepted = False
        self.variable = {}
        self.variable_created = False
        self.outputs = []
        self.outputs_final = []
        self.dropout_skip = False
        self.l2_loss = 0
        self.output = None
        self.built_once = False
        if self.auto_build and self.feed != None:
            self.build(self.feed)
        return None
    
    def reinit(self, *args, **kwargs):
        _kwargs = {
            **kwargs,
            **self.kwargs,
        }
        # _kwargs.update((k,v) for k,v in self.kwargs.items() if v is not None)
        self.__init__(*args, **_kwargs)
        return True
    
    def layer_check(self):
        for i in range(100):
            name = self.name + ('_' + str(i) if i > 0 else '')
            if name not in TF_ALL_BUILT_LAYERS or TF_ALL_BUILT_LAYERS[name] == self:
                self.name_checked = True
                self.name = name
                TF_ALL_BUILT_LAYERS[name] = self
                break
        return self.name_checked
    
    # locked / internal
    def arg_process(self, name='_', value=None, kwargs_def=[]):
        if not isinstance(name, (str)) or len(name) < 1:
            return False
        if len(kwargs_def) < 2:
            if value is None:
                value = kwargs_def[0]
        elif not callable(kwargs_def[1]):
            if (is_array(kwargs_def[0], len_min=1)
                    and value not in kwargs_def[0]):
                value = kwargs_def[0][0]
        else:
            if not kwargs_def[1](value):
                value = kwargs_def[0]
        self.__dict__[name] = value
        return True
    
    # locked / internal
    def prop_process_base(self):
        if not isinstance(self.dropout, (int, float)) or self.dropout <= 0:
            self.dropout = 0
            self.dropout_skip = True
            self.dropout_op_remove = []
            self.dropout_op_reset = []
        else:
            with self.get_scope('dropout'):
                self.dropout = np.clip(self.dropout, 0., 1.).astype(float)
                self.dropout_var = tf.Variable(self.dropout, trainable=False, name='var')
                self.dropout_op_remove = tf.assign(self.dropout_var, 0.0, name='op_remove')
                self.dropout_op_reset = tf.assign(self.dropout_var, self.dropout, name='op_reset')
        # if not callable(self.activator):
        #     self.activator = tf.identity
        if not is_int(self.out_dim) or self.out_dim < 1:
            self.out_dim = 1
        return True
    
    # customizable / internal
    def prop_process(self):
        return True
    
    # customizable
    def feed_check(self, feed):
        return True
    
    # customizable
    def feed_process(self, feed):
        feed_received = feed
        return feed
        # feed_processed = tf.identity(feed_received, name='feed_processed')
        # return feed_processed
    
    # customizable / internal
    def output_calc(self, feed=[]):
        self.calc_current = {}
        self.calc_current['output'] = tf.identity(
            feed,
            name='calc_pre',
        )
        return True
    
    def output_batch_norm(self):
        if not self.batch_norm:
            return False
        self.calc_current['batch_norm'] = tf.layers.batch_normalization(
            inputs=self.calc_current['output'],
            training=STATE_TRAINING,
            name=self.name + '/batch_norm',
            # reuse=True,
            reuse=tf.AUTO_REUSE,
        )
        self.calc_current['output'] = self.calc_current['batch_norm']
        return True
    
    # locked / internal
    def output_dropout(self):
        if self.dropout_skip:
            return False
        self.calc_current['dropout'] = tf.layers.dropout(
            self.calc_current['output'],
            rate=self.dropout,
            name='dropout',
        )
        self.calc_current['output'] = self.calc_current['dropout']
        return True
    
    # locked / internal
    def output_activation(self):
        if not callable(self.activator):
            return False
        self.calc_current['activation'] = self.activator(
            self.calc_current['output'],
            **{
                **self.activator_kwargs,
                **{'name': 'activation'},
            },
        )
        self.calc_current['output'] = self.calc_current['activation']
        return True
    
    # customizable / internal
    def output_finalize_base(self):
        self.calc_current['output'] = tf.identity(
            self.calc_current['output'],
            name='output'
        )
        self.output = self.calc_current['output']
        return self.output
    
    def output_finalize(self):
        self.output_final = tf.identity(self.output, name='output_final')
        return self.output_final
    
    # customizable / internal
    def variable_create(self):
        # create variables here
        # self.variable['']
        # log the keys for variables with l2_loss
        self.variable_l2_keys = []
        return True
    
    # locked / internal
    def variable_l2_loss(self):
        if len(self.variable_l2_keys) < 1:
            return False
        if is_tf_tensor_or_variable(self.l2_loss_scale):
            self.l2_loss_scale = tf.identity(
                self.l2_loss_scale,
                name='scale'
            )
        else:
            try:
                _ = float(self.l2_loss_scale)
            except Exception:
                self.l2_loss_scale = 0.0
            self.l2_loss_scale = tf.Variable(
                self.l2_loss_scale,
                dtype=tf.float32,
                trainable=False,
                name='scale',
            )
        self.l2_losses = [
            tf.nn.l2_loss(self.variable[k], name=k)
            for k in self.variable
            if k in self.variable_l2_keys
        ]
        self.l2_loss_concat = tf.concat(self.l2_losses, axis=0, name='concat')
        self.l2_loss = tf.reduce_sum(
            tf.multiply(
                self.l2_loss_concat,
                self.l2_loss_scale,
                name='concat_scaled'
            ),
            name='sum',
        )
        return True
    
    # locked / external_main
    def build(self, feed=None):
        # if not is_tf_tensor_or_variable(feed):
        #     return False
        if not self.feed_check(feed):
            return False
        self.feed = feed
        if is_tf_tensor_or_variable(self.feed):
            self.feed_rank = self.feed._rank()
            self.feed_shape = self.feed._shape_as_list()
        self.feed_accepted = True
        with self.get_scope(self.count):
            feed_received = tf.identity(feed, name='feed_received')
            feed_processed = self.feed_process(feed_received)
            if is_tf_tensor_or_variable(feed_processed):
                self.feed_rank = feed_processed._rank()
                self.feed_shape = feed_processed._shape_as_list()
        self.prop_accepted = self.prop_process_base() and self.prop_process()
        if not self.prop_accepted:
            return False
        if not self.variable_created and self.feed_accepted:
            with self.get_scope('var'):
                self.variable_created = self.variable_create()
                # add copies of variables to __dict__
                self.__dict__.update(self.variable)
        if not self.variable_created:
            return False
        with self.get_scope('l2_loss'):
            self.variable_l2_loss()
        with self.get_scope(self.count):
            _ = self.output_calc(feed_processed)
            _ = self.output_batch_norm()
            _ = self.output_activation()
            _ = self.output_dropout()
            # self.output = self.calc_current['output']
            # self.output_final = self.output_finalize()
            self.output_finalize_base()
            self.output_finalize()
            self.feed_received.append(feed_received)
            self.feed_processed.append(feed_processed)
            self.outputs.append(self.output)
            self.outputs_final.append(self.output_final)
            self.calc.append(self.calc_current)
        self.count += 1
        self.built_once = True
        return True
    
    # locked
    def get_calc_scope(self):
        return self.get_scope(self.count)
    
    # locked
    def get_scope(self, *args):
        return tf.name_scope('{0}{1}/'.format(self.scope, '/'.join(str(v) for v in args)))
    


class TF_CNN(TF_Base):
    
    """
    [2D CNN]
    filter_shape
    strides
    padding
    use_cudnn_on_gpu
    data_format
    dilations
    pooling
    """
    
    kwargs_def_custom = {
        'name': ['cnn', is_string],
        'filter_shape': [None],
        'strides': [[1,1,1,1]],
        'padding': [['SAME', 'VALID'], None],
        'use_cudnn_on_gpu': [[True, False], None],
        'data_format': [['NHWC', 'NCHW'], None],
        'dilations': [[1, 1, 1, 1], lambda v: is_array(v, len_min=4, len_max=4)],
        'pooling': [None],
    }
    
    # locked / internal
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # customizable
    def feed_process(self, feed):
        # fix rank to 4
        feed_rank = feed._rank()
        fixed_rank = 4
        axis = np.arange(*sorted([feed_rank, fixed_rank]))
        if feed_rank == fixed_rank:
            feed_processed = tf.identity(feed, name='feed_processed')
        elif feed_rank < fixed_rank:
            feed_processed = tf.expand_dims(feed, axis=axis, name='feed_processed')
        elif feed_rank > fixed_rank:
            feed_processed = tf.reduce_mean(feed, axis=axis, name='feed_processed')
        return feed_processed
    
    # customizable / internal
    def prop_process(self):
        # strides
        strides_HW = [1]
        if is_int(self.strides):
            strides_HW = [self.strides]
        elif is_array(self.strides):
            if len(self.strides) > 1:
                strides_HW = [v for v in self.strides[:3][-2:]]
            elif len(self.strides) == 1:
                strides_HW = [self.strides[0]]
        self.strides = (([1] + strides_HW * 2)[:3] + [1] * 4)[:4]
        # filter_shape
        filter_shape_HW = []
        if is_int(self.filter_shape) and self.filter_shape > 0:
            filter_shape_HW = [self.filter_shape]
        elif is_array(self.filter_shape, len_min=1):
            filter_shape_HW = [v for v in self.filter_shape if is_int(v)]
        filter_shape_HW = (filter_shape_HW * 2 + self.feed_shape[1:3])[:2]
        self.filter_shape = filter_shape_HW + self.feed_shape[-1:] + [self.out_dim]
        # pooling
        pooling_checked = False
        if isinstance(self.pooling, (int, np.int32, np.int64)):
            # self.pooling = None
            pooling_checked = True
            # self.pooling = [[1] + [int(self.pooling)] * 2 + [1]] * 2
            self.pooling = [int(self.pooling)] * 2
        if is_array(self.pooling, len_min=1):
            pooling_checked = True
            self.pooling = [int(v) for v in self.pooling if v % 1 == 0 and v >= 1]
            self.pooling = (self.pooling * 2)[:2]
        if not pooling_checked:
            self.pooling = None
        else:
            self.pooling = [[1] + self.pooling + [1]] * 2
        return True
    
    # customizable / internal
    def variable_create(self):
        self.variable['filter'] = tf.Variable(
            tf.truncated_normal(
                shape=self.filter_shape,
                # mean=0.0,
                stddev=1.0/np.product(self.filter_shape[:-1]),
            ),
            name='filter',
        )
        self.variable['bias'] = tf.Variable(
            tf.zeros(shape=[self.out_dim]),
            name='bias',
        )
        # keys of Variables to get l2_loss
        self.variable_l2_keys = ['filter']
        return True
    
    # customizable / internal
    def output_calc(self, feed=None):
        # output = tf.identity(feed, name='output')
        self.calc_current = {}
        x = feed
        x = tf.nn.conv2d(
            input=x,
            filter=self.variable['filter'],
            strides=self.strides,
            padding=self.padding,
            use_cudnn_on_gpu=self.use_cudnn_on_gpu,
            data_format=self.data_format,
            dilations=self.dilations,
            name='conv2d',
        )
        self.calc_current['conv2d'] = x
        x = tf.nn.bias_add(
            value=x,
            bias=self.variable['bias'],
            name='bias_add'
        )
        self.calc_current['bias_add'] = x
        x = self.activator(
            x,
            **{
                **self.activator_kwargs,
                **{'name': 'activation'},
            },
        )
        self.calc_current['activation'] = x
        if self.pooling:
            x = tf.nn.max_pool(
                value=x,
                ksize=self.pooling[0],
                strides=self.pooling[1],
                padding='VALID',
                name='max_pool'
            )
            self.calc_current['max_pool'] = x
        self.calc_current['output'] = x
        return True
    


class TF_Dense(TF_Base):
    
    """
    [DENSE]
    out_dim
    axes
    """
    
    kwargs_def_custom = {
        'name': ['dense', is_string],
        'out_dim': [1, is_int],
        'axes': [1]
    }
    
    # locked / internal
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # customizable
    def feed_process(self, feed):
        feed_processed = tf.identity(feed, name='feed_processed')
        return feed_processed
    
    # customizable / internal
    def prop_process(self):
        if self.axes == None:
            self.axes = [v for v in range(1, self.feed_rank)]
        if is_array(self.axes, len_min=1):
            self.axes = list(np.unique([
                v for v in self.axes
                if is_int(v) and v in range(self.feed_rank)
            ]))
            self.weight_shape = [self.feed_shape[v] for v in self.axes]
            self.axes = [self.axes, [i for i in range(len(self.axes))]]
        else:
            if not self.axes in range(1, self.feed_rank):
                self.axes = self.feed_rank - 1
            self.weight_shape = self.feed_shape[-self.axes:]
        self.weight_shape += [self.out_dim]
        return True
    
    # customizable / internal
    def variable_create(self):
        self.variable['weight'] = tf.Variable(
            tf.truncated_normal(
                shape=self.weight_shape,
                mean=0.0,
                stddev=1.0/np.product(self.weight_shape[:-1]),
            ),
            name='weight',
        )
        self.variable['bias'] = tf.Variable(
            tf.zeros(
                shape=[self.out_dim] if self.out_dim == self.weight_shape[-1] else []
            ),
            name='bias',
        )
        # keys of Variables to get l2_loss
        self.variable_l2_keys = ['weight']
        return True
    
    # customizable / internal
    def output_calc(self, feed=None):
        # output = tf.identity(feed, name='output')
        self.calc_current = {}
        self.calc_current['tensordot'] = tf.tensordot(
            a=feed,
            b=self.variable['weight'],
            axes=self.axes,
            name='tensordot',
        )
        self.calc_current['bias_add'] = tf.add(
            self.calc_current['tensordot'],
            self.variable['bias'],
            name='bias_add'
        )
        self.calc_current['output'] = self.calc_current['bias_add']
        return True
    


class TF_RNN(TF_Base):
    
    """
    [RNN]
    bidirectional
    sequence_length
    cell_type
    cell_count
    """
    
    kwargs_def_custom = {
        'name': ['rnn', is_string],
        'bidirectional': [[False, True], None],
        'sequence_length': [None],
        'cell_count': [1],
        'cell_type': [['lstm', 'basic', 'simple', 'gru'], None],
    }
    
    # locked / internal
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # customizable
    # def feed_check(self, feed):
    #     return True
    
    # customizable
    # def feed_process(self, feed):
    #     feed_processed = tf.identity(feed, name='feed_processed')
    #     return feed_processed
    
    # customizable / internal
    def prop_process(self):
        self.direction_count = [1, 2][self.bidirectional == True]
        if not is_array(self.cell_count):
            self.cell_count = [self.cell_count]
        self.cell_count = [
            v for v in self.cell_count
            if is_int(v) and v >= 1
        ]
        if len(self.cell_count) < 1:
            self.cell_count = [1]
        self.cell_count = (self.cell_count * 2)[:]
        return True
    
    # customizable / internal
    def variable_create(self):
        cell_classes = {
            'lstm': tf.nn.rnn_cell.LSTMCell,
            'gru': tf.nn.rnn_cell.GRUCell,
            'basic': tf.nn.rnn_cell.BasicRNNCell,
            'simple': tf.nn.rnn_cell.BasicRNNCell,
        }
        if self.cell_type not in cell_classes:
            self.cell_type = [k for k in cell_classes][0]
        cell_class = cell_classes[self.cell_type]
        self.cells = [
            cell_class(
                num_units=self.cell_count[i],
                reuse=tf.AUTO_REUSE,
                name='cells/{}'.format(i),
                # initializer=None,
                # dtype=None,
                # activation=None,
                # use_peepholes=False,
                # cell_clip=None,
                # forget_bias=1.0,
                # state_is_tuple=True,
            ) for i in range(self.direction_count)
        ]
        # keys of Variables to get l2_loss
        self.variable_l2_keys = []
        return True
    
    # customizable / internal
    def output_calc(self, feed=None):
        self.calc_current = {}
        self.calc_current['output'] = tf.identity(feed, name='output')
        kwargs_rnn = {
            'inputs': feed,
            'dtype': tf.float32,
            'scope': self.scope,
            'sequence_length': self.sequence_length,
            'time_major': False,
            # 'parallel_iterations': None,
            # 'swap_memory': False,
        }
        if self.bidirectional == True:
            self.calc_current['rnn'] = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.cells[0],
                cell_bw=self.cells[1],
                # initial_state_fw=None,
                # initial_state_bw=None,
                **kwargs_rnn,
            )
            self.calc_current['outputs'] = self.calc_current['rnn'][0]
            self.calc_current['states'] = self.calc_current['rnn'][1]
        else:
            self.calc_current['rnn'] = tf.nn.dynamic_rnn(
                cell=self.cells[0],
                # initial_state=None,
                **kwargs_rnn,
            )
            self.calc_current['outputs'] = (self.calc_current['rnn'][0],)
            self.calc_current['states'] = (self.calc_current['rnn'][1],)
        
        self.calc_current['output'] = tf.concat(
            self.calc_current['outputs'],
            axis=2,
            name="output"
        )
        self.calc_current['state_final'] = tf.concat(
            [v.h for v in self.calc_current['states']],
            axis=1,
            name="state_final"
        )
        # self.calc_current['output'] = tf.identity(self.calc_current['outputs'], name="output")
        # self.calc_current['state_final'] = tf.identity(self.calc_current['states'].h, name="state_final")
        return True
    


class TF_Process(TF_Base):
    
    """
    [process]
    """
    
    kwargs_def_custom = {
        'name': ['process', is_string],
        'func': [tf.identity, callable],
        'func_kwargs': [{}, lambda x: isinstance(x, (dict))],
    }
    
    # locked / internal
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # customizable / internal
    def prop_process(self):
        self.name = 'process_' + self.name
        self.scope = self.name + '/'
        return True
    
    # customizable / internal
    def output_calc(self, feed=None):
        self.calc_current = {}
        self.calc_current['func'] = self.func(
            feed,
            name='output',
            **self.func_kwargs,
        )
        self.calc_current['output'] = self.calc_current['func']
        return True
    

TF_Func = TF_Process


class TF_Template(TF_Base):
    
    """
    [TEMPLATE]
    """
    
    kwargs_def_custom = {
        'name': ['template', is_string],
    }
    
    # locked / internal
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # customizable
    def feed_check(self, feed):
        return True
    
    # customizable
    def feed_process(self, feed):
        feed_processed = tf.identity(feed, name='feed_processed')
        return feed_processed
    
    # customizable / internal
    def prop_process(self):
        return True
    
    # customizable / internal
    def variable_create(self):
        # keys of Variables to get l2_loss
        self.variable_l2_keys = []
        return True
    
    # customizable / internal
    def output_calc(self, feed=None):
        self.calc_current = {}
        self.calc_current['output'] = tf.identity(feed, name='output')
        return True
    





