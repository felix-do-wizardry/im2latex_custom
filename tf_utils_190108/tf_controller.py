import tensorflow as tf
import numpy as np
import time, timeit
from sys import stdout

# from tf_builder import *

TF_ALL_SERIES = {}

def np_splitter(a=np.array([]), s=1):
    if not isinstance(a, (np.ndarray)):
        return []
    if len(a) <= 0:
        return []
    a_split = np.split(
        a,
        np.arange(1, np.ceil(len(a)/s).astype(int)) * s
    )
    return a_split

def advanced_print(txt='', a=[], index=0, endline=False, txt_sub=''):
    _str_select = ['[', ']']
    str_select = [_str_select] + [[' ' * len(v) for v in _str_select]]
    str_separator = ' '
    a_str = [str(v) for v in a]
    a_str_len = [len(v) for v in a_str]
    a_str = [
        v.rjust(max(a_str_len), ' ').join(
            str_select[index!=i]
        )
        for i, v in enumerate(a_str)
    ]
    stdout.flush()
    stdout.write(
        "\r{0}{2}{1}".format(
            txt,
            txt_sub,
            str_separator.join(a_str),
        ) + ' ' * 4
    )
    if endline:
        print()
    return True



class TF_Controller():
    def __init__(
        self,
        inputs=None,
        labels=None,
        pred=None,
        loss=None,
        data=None,
        class_count=2,
        learning_rate=0.1,
        batch_size=100,
        val_split_ratio=0.0,
        sess=None,
        optimizer=tf.train.AdagradOptimizer,
        l2_loss_scale=0.0,
        layers=[],
        op_init_train=[],
        op_init_check=[],
        *args,
        **kwargs,
    ):
        self.scope = 'controller/'
        self.class_count = class_count
        self.tensors_got_set = False
        self.data_got_set = [False] * 3
        self.batch_got_set = False
        self.timer = []
        self.layers = layers
        self.l2_loss_scale = l2_loss_scale
        self.op_init_train = op_init_train + [v.dropout_op_reset for v in self.layers]
        self.op_init_check = op_init_check + [v.dropout_op_remove for v in self.layers]
        self.l2_loss_array = [v.l2_loss for v in self.layers]
        self.set_tensors([inputs, labels, pred, loss])
        # if not self.tensors_got_set:
        #     print('setting tensors failed!')
        #     return None
        self.set_summary()
        self.set_ops()
        self.set_optimizer(optimizer, learning_rate)
        self.set_session(sess)
        self.reset()
        self.set_data(data, val_split_ratio)
        self.set_data_batch(batch_size)
    
    def run_optimize_batch(
        self,
        batch_size=[],
        repeats=1,
        fetches=None,
        quick=True,
        auto_set=False,
    ):
        if not self.check_tensor_operation(fetches, True):
            fetches = self.op_train
        # batching training data
        results = []
        batches = []
        pre_run = False
        data_train = self.data[0]
        data_len = min([len(data_train[0]), len(data_train[1])])
        data_formated = np.array([
            [
                data_train[0][i],
                data_train[1][i],
            ] for i in range(data_len)
        ])
        # if not isinstance(batch_size, (list, tuple, np.ndarray)):
        #     batch_size = (2**np.arange(-4.0,1.1,.5)*1000).astype(int)
        batch_size = np.unique(np.round(batch_size)).astype(int)
        for s in batch_size:
            data_split = np_splitter(data_formated, int(s))
            data_batch_count_total = len(data_split)
            data_split = data_split[:-1]
            np.random.shuffle(data_split)
            data_split = data_split[:5]
            data_batch_count_run = len(data_split)
            batch = [
                {
                    self.inputs: np.array([v[0] for v in d]),
                    self.labels: np.array([v[1] for v in d]),
                }
                for d in data_split
            ]
            batches.append(batch)
            if not pre_run or True:
                pre_run = True
                run_output = [self.sess.run(fetches, b) for b in batch[:2]]
                # print('pre_run Done')
            # print('running with batch_size=[{0}]'.format(s))
            time_elapsed = []
            time_elapsed_single = []
            txt = 'batch_size ' + '[{0}]'.format(s).rjust(7)
            for i in range(repeats):
                advanced_print(txt + ' {0}/{1}'.format(
                    i+1, repeats
                ))
                time_start = time.time()
                time_elapsed_single.append([])
                for j, b in enumerate(batch):
                    advanced_print(
                        txt
                        + ' {0}/{1}'.format(
                            i+1, repeats
                        ) + ' [{0}%]'.format(
                            np.round((j / len(batch) + i) / repeats * 100).astype(int)
                        ).rjust(7)
                    )
                    time_start_single = time.time()
                    _ = self.sess.run(fetches, b)
                    time_finish_single = time.time()
                    time_elapsed_single[-1].append(
                        time_finish_single - time_start_single
                    )
                # run_output = [self.sess.run(fetches, b) for b in batch]
                time_finish = time.time()
                time_elapsed.append(
                    (time_finish - time_start)
                    * data_batch_count_total / data_batch_count_run
                )
            advanced_print(txt + ' <{0}> {1}'.format(
                np.round(np.mean(time_elapsed), 3),
                ' '.join([str(v).rjust(5) for v in np.round(time_elapsed, 2)])
            ))
            print()
            results.append((s, np.mean(time_elapsed), time_elapsed_single))
        best_batch_size = sorted(results, key=lambda v: v[1])[0][0]
        return results
    
    def train(self, count=1):
        self.run(self.op_init_train, quick=True)
        step_start = self.step
        timer_index = len(self.timer)
        for i in range(count):
            time_start = time.time()
            self.run(self.op_train, batch_index=0)
            time_finish = time.time()
            self.timer.append(time_finish - time_start)
            self.step += 1
            advanced_print('training [{0}/{1}] {2}'.format(
                self.step,
                step_start + count,
                ' '.join(str(np.round(v, 2)) for v in self.timer[timer_index:])
            ))
        print()
        return True
    
    def check(self, batch_index=[0,1]):
        if self.check_array(batch_index, 1):
            return [self.check(i) for i in batch_index]
        if batch_index not in np.arange(len(self.batch)):
            return False
        self.run(self.op_init_check, quick=True)
        self.run(self.summ_op_reset, quick=True)
        self.run(self.summ_op_add, batch_index=batch_index)
        check_output = self.run(self.summ_value_mean_dict, quick=True)
        return check_output
    
    def check_summary(self, cf_matrix_index=[0,1]):
        if not isinstance(cf_matrix_index, (list, np.ndarray)):
            cf_matrix_index = [cf_matrix_index]
        check_output = self.check(cf_matrix_index)
        cf_matrix = np.array([v['cf_matrix'] for v in check_output])
        # cf_matrix_per = 100*cf_matrix/np.tile(np.sum(cf_matrix, axis=2),[10,1]).T
        cf_matrix_per = 100 * cf_matrix / np.tile(np.expand_dims(np.sum(cf_matrix, axis=2), 3), self.class_count)
        cfmt_num_len = 4
        cfmt_hl = ['[]', '<>'][0]
        cf_matrix_txt = [
            '\n'.join([
                ''.join([
                    [
                        ('#' * (cfmt_num_len - 2)).join('()'),
                        '{:>.1f}'.format(v),
                        '',
                    ][np.argmax([
                        np.round(v, 1) == 100,
                        v >= 0.5 or x == y,
                        True
                    ])].rjust(cfmt_num_len)[-cfmt_num_len:]
                    .join(cfmt_hl if x == y else ''.rjust(len(cfmt_hl)))
                    for x, v in enumerate(r)
                ]).join('||')
                for y, r in enumerate(m)
            ])
            for m in cf_matrix_per
        ]
        summ_txt = '\n'.join([
            '{0} {1}'.format(
                k.rjust(6),
                ' '.join(['{:>10.6f}'.format(v * s)
                    for v in np.clip([v[k] for v in check_output], -99, 999)
                ]),
            )
            for k, s in [('acc', 100), ('loss', 1)]
        ])
        if not self.check_array(cf_matrix_index):
            cf_matrix_index = [cf_matrix_index]
        cf_matrix_txt_sep = ('-' * ((cfmt_num_len + 2) * self.class_count + 2)).join(['\n']*2)
        txt = ''.join([
            v + cf_matrix_txt_sep
            for i, v in enumerate(cf_matrix_txt)
            if i in cf_matrix_index
        ])
        txt += summ_txt
        print(txt)
        # print(cf_matrix_txt_x)
        # print(summ_txt)
        return None
    
    def run(self, fetches=None, batch_index=0, allow_batch_index_array=True, quick=False):
        # if not self.check_tensor_operation(fetches, allow_array=True):
        #     return False
        if quick:
            return self.sess.run(fetches)
        if self.check_array(batch_index, 1) and allow_batch_index_array:
            return [
                self.run(
                    fetches,
                    batch_index=i,
                    allow_batch_index_array=False,
                ) for i in batch_index
            ]
        batch = [{}]
        if batch_index in range(len(self.batch)):
            batch = self.batch[batch_index]
        run_output = [self.sess.run(fetches, b) for b in batch]
        return run_output
    
    def reset(self):
        self.step = 0
        self.run(self.op_init_global, quick=True)
        return True
    
    def check_tensor_operation(self, fetches=None, allow_array=False):
        if self.check_array(fetches, 1) and allow_array:
            return sum(
                not isinstance(v, (tf.Tensor, tf.Operation))
                for v in fetches
            ) <= 0
        else:
            return isinstance(fetches, (tf.Tensor, tf.Operation))
    
    def check_array(self, a=[], len_min=1, len_max=None):
        if not isinstance(a, (list, tuple, np.ndarray)):
            return False
        try:
            if len_min >= 0 and len(a) < len_min:
                return False
        except Exception:
            pass
        try:
            if len_max >= 0 and len(a) > len_max:
                return False
        except Exception:
            pass
        return True
    
    def set_session(self, sess=None):
        if not isinstance(sess, (tf.Session)):
            sess = tf.Session()
        self.sess = sess
        self.op_init_global = tf.global_variables_initializer()
        # self.sess.run(self.op_init_global)
        return self.sess
    
    def set_summary(self):
        self.summ_scope = self.scope + 'summary/'
        with tf.variable_scope(self.summ_scope):
            key_inputs_count = 'inputs_count'
            self.summ_tensors = {
                key_inputs_count: self.inputs_count,
                'acc':        self.match_count,
                'loss':       self.loss_sum,
                'cf_matrix':  self.cf_matrix,
            }
            self.summ = {}
            self.summ[key_inputs_count] = TF_Summary(
                self.inputs_count,
                inputs_count=1
            )
            self.summ.update({
                k: TF_Summary(
                    self.summ_tensors[k],
                    inputs_count=self.summ[key_inputs_count].value,
                )
                for k in self.summ_tensors if k != key_inputs_count
            })
            self.summ_obj = [self.summ[k] for k in self.summ]
            with tf.name_scope('group/'):
                self.summ_op_reset = tf.group(
                    [s.op_reset for s in self.summ_obj],
                    name='op_reset',
                )
                self.summ_op_add = tf.group(
                    [s.op_add for s in self.summ_obj],
                    name='op_add',
                )
                self.summ_value = tf.tuple(
                    [s.value for s in self.summ_obj],
                    name='value',
                )
                self.summ_value_mean = tf.tuple(
                    [s.value_mean for s in self.summ_obj],
                    name='value_mean',
                )
                self.summ_value_main = tf.tuple(
                    [
                        self.summ[k].value_mean
                        # if k not in [key_inputs_count]
                        # else self.summ[k].value
                        for k in self.summ
                    ],
                    name='value_main',
                )
                self.summ_value_mean_dict = {
                    k: self.summ[k].value_mean
                    for k in self.summ
                }
        return True
    
    def set_ops(self):
        self.op_all = []
        self.op_reset = tf.group([])
        return True
    
    def set_data(self, data, val_split_ratio=0.0):
        # takes in 2 sets of data for <train+val> and <test>
        empty_data = [np.array([])] * 2
        self.val_split_ratio = np.clip(val_split_ratio, 0.0, 1.0)
        self.data = []
        if self.check_array(data, 1):
            self.data += [v for v in data]
        self.data.insert(1, empty_data)
        self.data += [empty_data] * 3
        self.data = self.data[:3]
        self.set_data_split_val_from_train(self.val_split_ratio)
        self.data_len = [len(d[0]) for d in self.data]
        return self.data
    
    def set_data_split_val_from_train(self, val_split_ratio=0.0):
        try:
            if val_split_ratio > 1:
                return False
        except Exception:
            return False
        split_len = np.floor(
            val_split_ratio * len(self.data[0][0])
        ).astype(int)
        if split_len <= 0:
            return True
        self.data[1] = [self.data[0][i][-split_len:] for i in range(2)]
        self.data[0] = [v[:-split_len] for v in self.data[0]]
        return self.data
    
    def set_data_batch(self, batch_size=100):
        self.batch_size = batch_size
        self.batch = [
            [
                {
                    [self.inputs, self.labels][g]
                    : d[g][self.batch_size * i : self.batch_size * (i+1)]
                    for g in range(2)
                }
                for i in np.arange(len(d[0]) / self.batch_size).astype(int)
            ]
            for d in self.data
        ]
        self.batch_size = [len(b) for b in self.batch]
        self.batch_len = [[len(d[self.inputs]) for d in b] for b in self.batch]
        self.batch_got_set = True
        return self.batch
    
    def set_tensors(self, tensors):
        if not self.check_array(tensors, 4):
            return False
        if min([isinstance(v, (tf.Tensor, tf.Variable)) for v in tensors]) <= 0:
            return False
        self.tensors = tensors[:4]
        with tf.name_scope(self.scope):
            self.inputs, self.labels, self.pred, self.loss = self.tensors
            self.inputs_count = tf.identity(tf.shape(self.inputs, name='input_shape')[0], name='input_count')
            self.loss_sum = tf.reduce_sum(self.loss, name='loss_sum')
            self.loss_mean = tf.reduce_mean(self.loss, name='loss_mean')
            self.loss_train = tf.add(
                self.loss_mean,
                tf.reduce_sum(self.l2_loss_array) * self.l2_loss_scale,
                name='loss_train',
            )
            self.diff_raw = tf.subtract(
                tf.cast(self.pred, tf.float32),
                tf.cast(self.labels, tf.float32),
                name='difference_raw'
            )
            self.diff = tf.reduce_sum(
                tf.abs(self.diff_raw),
                axis=np.arange(1, self.diff_raw._rank()),
                name='difference',
            )
            self.match = tf.less_equal(self.diff, 0.0, name='match')
            self.match_int32 = tf.cast(self.match, tf.int32, name='match_int32')
            self.match_float32 = tf.cast(self.match, tf.float32, name='match_float32')
            self.match_count = tf.reduce_sum(self.match_int32, name='match_count')
            self.acc = tf.reduce_mean(self.match_float32, name='accuracy')
            self.cf_matrix = tf.confusion_matrix(
                labels=self.labels,
                predictions=self.pred,
                num_classes=self.class_count,
                name='confusion_matrix'
            )
        self.tensors_got_set = True
        return self.tensors
    
    def set_optimizer(self, optimizer=None, learning_rate=None):
        try:
            _ = learning_rate > 0
        except Exception:
            learning_rate = 0.1
        self.learning_rate = learning_rate
        if not isinstance(optimizer, (tf.train.Optimizer)):
            optimizer = tf.train.AdagradOptimizer
        self.optimizer = optimizer(self.learning_rate)
        self.op_train = self.optimizer.minimize(self.loss_train)
        return self.optimizer, self.learning_rate
    
    def _(self):
        pass



class TF_Graph():
    """
    Currently only supporting classification
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = {
            'inputs': None,
            'layers': [],
            'labels': None,
            'loss_func': None,
            'loss_func_keys': ['inputs', 'logits'],
            'loss_func_kwargs': {},
            **kwargs,
        }
        # if len(self.args) >= 1:
        #     self.kwargs.update({'inputs': self.args[0]})
        # if len(self.args) >= 2:
        #     self.kwargs.update({'layers': self.args[1]})
        self.layers = self.kwargs['layers']
        self.inputs = None
        self.inputs_all = []
        self.graph_constructed = False
        self.controller_constructed = False
        self.loss_contructed = False
        self.graph_fed = False
        self.inputs_checked = False
        self.set_inputs(self.kwargs['inputs'])
        if not self.inputs_checked:
            return None
        self.graph_construct()
        self.loss_contruct(self.kwargs['labels'])
        # if self.graph_constructed:
        #     self.controller_construct()
        return None
    
    def set_inputs(self, inputs=None):
        if inputs == None:
            return False
        self.inputs = inputs
        self.inputs_all.append(self.inputs)
        self.inputs_checked = True
        return True
    
    def graph_construct(self):
        x = self.inputs
        for layer in self.layers:
            layer.build(x)
            x = layer.output
            # x = layer.output_final
        self.output = x
        self.pred = tf.argmax(
            tf.reduce_sum(
                self.output,
                axis=np.arange(2, self.output._rank()),
                name='output_collapsed',
            ),
            axis=-1,
            output_type=tf.int32,
            name='prediction'
        )
        self.graph_constructed = True
        return x
    
    def loss_contruct(self, labels=None):
        if labels == None:
            return False
        self.labels = labels
        if not callable(self.kwargs['loss_func']):
            return False
        self.loss_func = self.kwargs['loss_func']
        self.loss_calc = self.loss_func(
            **{
                self.kwargs['loss_func_keys'][0]: self.labels,
                self.kwargs['loss_func_keys'][1]: self.output,
                'name': 'loss_calc'
            },
            **self.kwargs['loss_func_kwargs'],
        )
        self.loss = tf.reduce_mean(self.loss_calc, name='loss')
        self.loss_contructed = True
        return True
    
    # def controller_construct(self):
    #     super().__init__(*self.args, **self.kwargs)
    #     self.controller_constructed = True
    #     return True



class TF_Series():
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = {
            'inputs': None,
            'layers': [],
            'labels': None,
            'loss_func': None,
            'loss_func_keys': ['inputs', 'logits'],
            'loss_func_kwargs': {},
            'layer_kwargs': {},
            'name': 'series',
            **kwargs,
        }
        # if len(self.args) >= 1:
        #     self.kwargs.update({'inputs': self.args[0]})
        # if len(self.args) >= 2:
        #     self.kwargs.update({'layers': self.args[1]})
        self.name = self.kwargs['name']
        for i in range(100):
            name = self.kwargs['name'] + (str(i) if i > 0 else '')
            if name not in TF_ALL_SERIES or TF_ALL_SERIES[name] == self:
                self.name = name
                TF_ALL_SERIES[name] = self
                break
        if name not in TF_ALL_SERIES:
            return None
        self.layers = self.kwargs['layers']
        self.inputs = None
        self.inputs_all = []
        self.graph_constructed = False
        self.controller_constructed = False
        self.loss_contructed = False
        self.graph_fed = False
        self.inputs_checked = False
        self.set_inputs(self.kwargs['inputs'])
        if not self.inputs_checked:
            return None
        self.graph_construct()
        return None
    
    def set_inputs(self, inputs=None):
        if inputs == None:
            return False
        self.inputs = inputs
        self.inputs_all.append(self.inputs)
        self.inputs_checked = True
        return True
    
    def graph_construct(self):
        self.layer_outputs = []
        x = self.inputs
        for i, layer in enumerate(self.layers):
            _kwargs = {
                **self.kwargs['layer_kwargs'],
                **{
                    'name': '{}/{}'.format(self.name, i),
                    'auto_build': False,
                },
            }
            layer.reinit(**_kwargs)
            layer.build(x)
            x = layer.output
            self.layer_outputs.append(layer.output)
            # x = layer.output_final
        self.output = x
        # self.pred = tf.argmax(
        #     tf.reduce_sum(
        #         self.output,
        #         axis=np.arange(2, self.output._rank()),
        #         name='output_collapsed',
        #     ),
        #     axis=-1,
        #     output_type=tf.int32,
        #     name='prediction'
        # )
        self.graph_constructed = True
        return x
        

# class TF_Master():
#     def __init__(self, *args, **kwargs):
        
#         pass
    
#     # def 

# for totaling tensors ONLY
class TF_Summary():
    def __init__(
        self,
        tensor,
        inputs_count=None,
        init_func=None,
        name=None,
        scope=None,
        *args,
        **kwargs,
    ):
        self.name = [tensor.name.split(':')[0], name][isinstance(name, str)]
        if not isinstance(tensor, (tf.Tensor, tf.Variable)):
            return None
        self.scope = (scope + '/') if isinstance(scope, str) else ''
        self.scope += '{0}/'.format(self.name)
        with tf.variable_scope(self.scope):
            self.tensor = tf.identity(tensor, name='tensor')
            self.shape = self.tensor._shape_as_list()
            self.init_func = tf.zeros
            if callable(init_func):
                self.init_func = init_func
            self.value = tf.Variable(
                self.init_func(
                    self.shape,
                    self.tensor.dtype,
                    # name='value_init'
                ),
                trainable=False,
                name='value',
            )
            self.inputs_count = self.value
            if isinstance(inputs_count, (tf.Tensor, tf.Variable)):
                self.inputs_count = inputs_count
            self.value_mean = tf.divide(
                tf.cast(self.value, tf.float32, name='value_float32'),
                tf.cast(self.inputs_count, tf.float32, name='inputs_count_float32'),
                name='value_mean',
            )
            self.op_add = tf.assign_add(self.value, self.tensor, name='op_add')
            self.op_reset = tf.group(self.value.initializer, name='op_reset')
            # self.op_inputs_count = tf.assign(self.value, self.value / inputs_count, name='op_scale')



