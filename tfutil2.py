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

class Network:
    """
    Unpickles a trained network to convert it to TF2. Only generator is supported.
    """

    def __setstate__(self, state):
        """
        Unpickling.
        """
        global UNPICKLE_COUNTER
        UNPICKLE_COUNTER += 1
        if UNPICKLE_COUNTER != 3:
            # Skip unused objects.
            return

        # Set basic fields.
        assert state['version'] == 2

        # self.mirrored_strategy = tf.distribute.MirroredStrategy()
        # with self.mirrored_strategy.scope():

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            self.latents_in = tf.keras.Input(name='latents_in', shape=[512])
            self.label_in = tf.keras.Input(name='labels_in', shape=[None])
            self.output = networks2.G_paper(self.latents_in, self.label_in, **state['static_kwargs'])


        # @tf.function
        # def output_func(latents_in, label_in):
        #     return networks2.G_paper(latents_in, label_in, **self.static_kwargs)
        # self.output = output_func(self.latents_in, self.label_in)

        #self.vars = OrderedDict([(self.get_var_localname(var), var) for var in tf.global_variables(self.scope + '/')])
        # set_vars({self.find_var(name): value for name, value in state['variables']})

        variable_values = dict(state['variables'])

        operations = []
        for variable in tf.global_variables():
            if variable.name[-4] == '_':
                key = variable.name[:-4]
            else:
                key = variable.name[:-2]
            print(variable.name, key)
            value = variable_values[key]
            operations.append(variable.assign(value))

        tf.get_default_session().run(operations)


        # self.keras_model = tf.keras.Model(inputs=(self.latents_in, self.label_in), outputs=self.output)


    def run(self, latents):
        """
        Generate images.
        """
        labels = np.zeros([len(latents)] + self.label_in.shape[1:])
        feed_dict = {
            self.latents_in: latents,
            self.label_in: labels
        }
        result = tf.get_default_session().run(self.output, feed_dict)
        return result

    def run_keras(self, latents):
        """
        Generate images using keras.
        """
        labels = np.zeros([len(latents)] + self.label_in.shape[1:])
        result = self.keras_model.predict([latents, labels])
        return result

