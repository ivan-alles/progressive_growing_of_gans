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

def absolute_name_scope(scope): # Forcefully enter the specified name scope, ignoring any surrounding scopes.
    return tf.name_scope(scope + '/')

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


class Network:
    """
    Unpickles a trained network to convert it to TF2. Only generator is supported.
    """
    # Run initializers for all variables defined by this network.
    def reset_vars(self):
        run([var.initializer for var in self.vars.values()])

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

    # Pickle import.
    def __setstate__(self, state):
        global UNPICKLE_COUNTER
        UNPICKLE_COUNTER += 1
        if UNPICKLE_COUNTER != 3:
            # Skip unused objects.
            return

        # Set basic fields.
        assert state['version'] == 2
        self.name = state['name']
        self.static_kwargs = state['static_kwargs']

        self.scope = tf.get_default_graph().unique_name(self.name.replace('/', '_'), mark_as_used=False)

        mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():
            with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
                assert tf.get_variable_scope().name == self.scope
                self.latent_inputs = tf.keras.Input(name='latents_in', shape=[None])
                self.label_inputs = tf.keras.Input(name='labels_in', shape=[None])
                self.output = networks2.G_paper(self.latent_inputs, self.label_inputs, **self.static_kwargs)


            self.vars = OrderedDict([(self.get_var_localname(var), var) for var in tf.global_variables(self.scope + '/')])
            self.trainables = OrderedDict(
                [(self.get_var_localname(var), var) for var in tf.trainable_variables(self.scope + '/')])

            self.reset_vars()
            set_vars({self.find_var(name): value for name, value in state['variables']})

            self.keras_model = tf.keras.Model(inputs=(self.latent_inputs, self.label_inputs), outputs=self.output)


    def run(self, latents):
        """
        Generate images.
        """
        labels = np.zeros([len(latents)] + self.label_inputs.shape[1:])
        feed_dict = {
            self.latent_inputs: latents,
            self.label_inputs: labels
        }
        result = tf.get_default_session().run(self.output, feed_dict)
        return result

    def run_keras(self, latents):
        """
        Generate images using keras.
        """
        labels = np.zeros([len(latents)] + self.label_inputs.shape[1:])
        result = self.keras_model.predict([latents, labels])
        return result

