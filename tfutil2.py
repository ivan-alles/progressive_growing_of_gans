# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

""" A port of the tfutil.py to make the generator work with TensorFlow 2. """

import numpy as np
from collections import OrderedDict
import tensorflow as tf
import networks2

# Use it to skip unpickling unnecessary objects.
UNPICKLE_COUNTER = 0

#----------------------------------------------------------------------------
# Convenience.

def run(*args, **kwargs): # Run the specified ops in the default session.
    return tf.get_default_session().run(*args, **kwargs)

def is_tf_expression(x):
    return isinstance(x, tf.Tensor) or isinstance(x, tf.Variable) or isinstance(x, tf.Operation)

def shape_to_list(shape):
    return [dim.value if hasattr(dim, 'value') else dim for dim in shape]

def flatten(x):
    with tf.name_scope('Flatten'):
        return tf.reshape(x, [-1])

def log2(x):
    with tf.name_scope('Log2'):
        return tf.log(x) * np.float32(1.0 / np.log(2.0))

def exp2(x):
    with tf.name_scope('Exp2'):
        return tf.exp(x * np.float32(np.log(2.0)))

def lerp(a, b, t):
    with tf.name_scope('Lerp'):
        return a + (b - a) * t

def lerp_clip(a, b, t):
    with tf.name_scope('LerpClip'):
        return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)

def absolute_name_scope(scope): # Forcefully enter the specified name scope, ignoring any surrounding scopes.
    return tf.name_scope(scope + '/')

#----------------------------------------------------------------------------
# Initialize TensorFlow graph and session using good default settings.

def init_tf(config_dict=dict()):
    if tf.get_default_session() is None:
        tf.set_random_seed(np.random.randint(1 << 31))
        create_session(config_dict, force_as_default=True)

#----------------------------------------------------------------------------
# Create tf.Session based on config dict of the form
# {'gpu_options.allow_growth': True}

def create_session(config_dict=dict(), force_as_default=False):
    config = tf.ConfigProto()
    for key, value in config_dict.items():
        fields = key.split('.')
        obj = config
        for field in fields[:-1]:
            obj = getattr(obj, field)
        setattr(obj, fields[-1], value)
    session = tf.Session(config=config)
    if force_as_default:
        session._default_session = session.as_default()
        session._default_session.enforce_nesting = False
        session._default_session.__enter__()
    return session

#----------------------------------------------------------------------------
# Initialize all tf.Variables that have not already been initialized.
# Equivalent to the following, but more efficient and does not bloat the tf graph:
#   tf.variables_initializer(tf.report_unitialized_variables()).run()

def init_uninited_vars(vars=None):
    if vars is None: vars = tf.global_variables()
    test_vars = []; test_ops = []
    with tf.control_dependencies(None): # ignore surrounding control_dependencies
        for var in vars:
            assert is_tf_expression(var)
            try:
                tf.get_default_graph().get_tensor_by_name(var.name.replace(':0', '/IsVariableInitialized:0'))
            except KeyError:
                # Op does not exist => variable may be uninitialized.
                test_vars.append(var)
                with absolute_name_scope(var.name.split(':')[0]):
                    test_ops.append(tf.is_variable_initialized(var))
    init_vars = [var for var, inited in zip(test_vars, run(test_ops)) if not inited]
    run([var.initializer for var in init_vars])

#----------------------------------------------------------------------------
# Set the values of given tf.Variables.
# Equivalent to the following, but more efficient and does not bloat the tf graph:
#   tfutil.run([tf.assign(var, value) for var, value in var_to_value_dict.items()]

def set_vars(var_to_value_dict):
    ops = []
    feed_dict = {}
    for var, value in var_to_value_dict.items():
        assert is_tf_expression(var)
        try:
            setter = tf.get_default_graph().get_tensor_by_name(var.name.replace(':0', '/setter:0')) # look for existing op
        except KeyError:
            with absolute_name_scope(var.name.split(':')[0]):
                with tf.control_dependencies(None): # ignore surrounding control_dependencies
                    setter = tf.assign(var, tf.placeholder(var.dtype, var.shape, 'new_value'), name='setter') # create new setter
        ops.append(setter)
        feed_dict[setter.op.inputs[1]] = value
    run(ops, feed_dict)


#----------------------------------------------------------------------------
# Generic network abstraction.
#
# Acts as a convenience wrapper for a parameterized network construction
# function, providing several utility methods and convenient access to
# the inputs/outputs/weights.
#
# Network objects can be safely pickled and unpickled for long-term
# archival purposes. The pickling works reliably as long as the underlying
# network construction function is defined in a standalone Python module
# that has no side effects or application-specific imports.

network_import_handlers = []    # Custom import handlers for dealing with legacy data in pickle import.
_network_import_modules = []    # Temporary modules create during pickle import.

class Network:
    def _init_graph(self):
        self.input_names = ['latents_in', 'labels_in']
        self.num_inputs = len(self.input_names)
        assert self.num_inputs >= 1

        self.scope = tf.get_default_graph().unique_name(self.name.replace('/', '_'), mark_as_used=False)

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            assert tf.get_variable_scope().name == self.scope
            self.latent_inputs = tf.keras.Input(name=self.input_names[0], shape=[None])
            self.label_inputs = tf.keras.Input(name=self.input_names[1], shape=[None])
            self.output = networks2.G_paper(self.latent_inputs, self.label_inputs, **self.static_kwargs)

        self.vars = OrderedDict([(self.get_var_localname(var), var) for var in tf.global_variables(self.scope + '/')])
        self.trainables = OrderedDict(
            [(self.get_var_localname(var), var) for var in tf.trainable_variables(self.scope + '/')])


    # Run initializers for all variables defined by this network.
    def reset_vars(self):
        run([var.initializer for var in self.vars.values()])

    # Run initializers for all trainable variables defined by this network.
    def reset_trainables(self):
        run([var.initializer for var in self.trainables.values()])

    # Get the local name of a given variable, excluding any surrounding name scopes.
    def get_var_localname(self, var_or_globalname):
        assert is_tf_expression(var_or_globalname) or isinstance(var_or_globalname, str)
        globalname = var_or_globalname if isinstance(var_or_globalname, str) else var_or_globalname.name
        assert globalname.startswith(self.scope + '/')
        localname = globalname[len(self.scope) + 1:]
        localname = localname.split(':')[0]
        return localname

    # Find variable by local or global name.
    def find_var(self, var_or_localname):
        assert is_tf_expression(var_or_localname) or isinstance(var_or_localname, str)
        return self.vars[var_or_localname] if isinstance(var_or_localname, str) else var_or_localname

    # Get the value of a given variable as NumPy array.
    # Note: This method is very inefficient -- prefer to use tfutil.run(list_of_vars) whenever possible.
    def get_var(self, var_or_localname):
        return self.find_var(var_or_localname).eval()
        
    # Set the value of a given variable based on the given NumPy array.
    # Note: This method is very inefficient -- prefer to use tfutil.set_vars() whenever possible.
    def set_var(self, var_or_localname, new_value):
        return set_vars({self.find_var(var_or_localname): new_value})

    # Pickle export.
    def __getstate__(self):
        return {
            'version':          2,
            'name':             self.name,
            'static_kwargs':    self.static_kwargs,
            'build_module_src': self._build_module_src,
            'build_func_name':  self._build_func_name,
            'variables':        list(zip(self.vars.keys(), run(list(self.vars.values()))))}

    # Pickle import.
    def __setstate__(self, state):
        global UNPICKLE_COUNTER
        UNPICKLE_COUNTER += 1
        if UNPICKLE_COUNTER != 3:
            # Skip unused objects.
            return

        # Execute custom import handlers.
        for handler in network_import_handlers:
            state = handler(state)

        # Set basic fields.
        assert state['version'] == 2
        self.name = state['name']
        self.static_kwargs = state['static_kwargs']

        # Init graph.
        self._init_graph()
        self.reset_vars()
        set_vars({self.find_var(name): value for name, value in state['variables']})


    # Create a clone of this network with its own copy of the variables.
    def clone(self, name=None):
        net = object.__new__(Network)
        net._init_fields()
        net.name = name if name is not None else self.name
        net.static_kwargs = dict(self.static_kwargs)
        net._init_graph()
        net.copy_vars_from(self)
        return net

    # Copy the values of all variables from the given network.
    def copy_vars_from(self, src_net):
        assert isinstance(src_net, Network)
        name_to_value = run({name: src_net.find_var(name) for name in self.vars.keys()})
        set_vars({self.find_var(name): value for name, value in name_to_value.items()})

    # Copy the values of all trainable variables from the given network.
    def copy_trainables_from(self, src_net):
        assert isinstance(src_net, Network)
        name_to_value = run({name: src_net.find_var(name) for name in self.trainables.keys()})
        set_vars({self.find_var(name): value for name, value in name_to_value.items()})

    # Create new network with the given parameters, and copy all variables from this network.
    def convert(self, name=None, func=None, **static_kwargs):
        net = Network(name, func, **static_kwargs)
        net.copy_vars_from(self)
        return net

    # Construct a TensorFlow op that updates the variables of this network
    # to be slightly closer to those of the given network.
    def setup_as_moving_average_of(self, src_net, beta=0.99, beta_nontrainable=0.0):
        assert isinstance(src_net, Network)
        with absolute_name_scope(self.scope):
            with tf.name_scope('MovingAvg'):
                ops = []
                for name, var in self.vars.items():
                    if name in src_net.vars:
                        cur_beta = beta if name in self.trainables else beta_nontrainable
                        new_value = lerp(src_net.vars[name], var, cur_beta)
                        ops.append(var.assign(new_value))
                return tf.group(*ops)

    def run_simple(self, latents):
        """
        A simplified version of run() for the generator model.
        """
        labels = np.zeros([len(latents)] + self.label_inputs.shape[1:])
        feed_dict = {
            self.latent_inputs: latents,
            self.label_inputs: labels
        }
        result = tf.get_default_session().run(self.output, feed_dict)
        return result


    # Returns a list of (name, output_expr, trainable_vars) tuples corresponding to
    # individual layers of the network. Mainly intended to be used for reporting.
    def list_layers(self):
        patterns_to_ignore = ['/Setter', '/new_value', '/Shape', '/strided_slice', '/Cast', '/concat']
        all_ops = tf.get_default_graph().get_operations()
        all_ops = [op for op in all_ops if not any(p in op.name for p in patterns_to_ignore)]
        layers = []

        def recurse(scope, parent_ops, level):
            prefix = scope + '/'
            ops = [op for op in parent_ops if op.name == scope or op.name.startswith(prefix)]

            # Does not contain leaf nodes => expand immediate children.
            if level == 0 or all('/' in op.name[len(prefix):] for op in ops):
                visited = set()
                for op in ops:
                    suffix = op.name[len(prefix):]
                    if '/' in suffix:
                        suffix = suffix[:suffix.index('/')]
                    if suffix not in visited:
                        recurse(prefix + suffix, ops, level + 1)
                        visited.add(suffix)

            # Otherwise => interpret as a layer.
            else:
                layer_name = scope[len(self.scope)+1:]
                layer_output = ops[-1].outputs[0]
                layer_trainables = [op.outputs[0] for op in ops if op.type.startswith('Variable') and self.get_var_localname(op.name) in self.trainables]
                layers.append((layer_name, layer_output, layer_trainables))

        recurse(self.scope, all_ops, 0)
        return layers

    # Print a summary table of the network structure.
    def print_layers(self, title=None, hide_layers_with_no_params=False):
        if title is None: title = self.name
        print()
        print('%-28s%-12s%-24s%-24s' % (title, 'Params', 'OutputShape', 'WeightShape'))
        print('%-28s%-12s%-24s%-24s' % (('---',) * 4))

        total_params = 0
        for layer_name, layer_output, layer_trainables in self.list_layers():
            weights = [var for var in layer_trainables if var.name.endswith('/weight:0')]
            num_params = sum(np.prod(shape_to_list(var.shape)) for var in layer_trainables)
            total_params += num_params
            if hide_layers_with_no_params and num_params == 0:
                continue

            print('%-28s%-12s%-24s%-24s' % (
                layer_name,
                num_params if num_params else '-',
                layer_output.shape,
                weights[0].shape if len(weights) == 1 else '-'))

        print('%-28s%-12s%-24s%-24s' % (('---',) * 4))
        print('%-28s%-12s%-24s%-24s' % ('Total', total_params, '', ''))
        print()

    # Construct summary ops to include histograms of all trainable parameters in TensorBoard.
    def setup_weight_histograms(self, title=None):
        if title is None: title = self.name
        with tf.name_scope(None), tf.device(None), tf.control_dependencies(None):
            for localname, var in self.trainables.items():
                if '/' in localname:
                    p = localname.split('/')
                    name = title + '_' + p[-1] + '/' + '_'.join(p[:-1])
                else:
                    name = title + '_toplevel/' + localname
                tf.summary.histogram(name, var)

#----------------------------------------------------------------------------
