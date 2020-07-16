# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

""" A port of the tfutil.py to make the generator work with TensorFlow 2. """

import re

import numpy as np
import tensorflow as tf
import networks2

# Use it to skip unpickling unnecessary objects.
UNPICKLE_COUNTER = 0

# TODO(ia): rework this, can be just a dictionary.
class UnpickledVariables:
    def __init__(self, variables):
        self._variables = dict(variables)

    def get(self, name):
        # TODO(ia): some variables are created with the same value. How to avoid duplicates?
        return self._variables[name]

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

        variables = UnpickledVariables(state['variables'])

        self.latents_in = tf.keras.Input(name='latents_in', shape=[512])
        output = networks2.G_paper(self.latents_in, variables, **state['static_kwargs'])
        self.keras_model = tf.keras.Model(inputs=self.latents_in, outputs=output)

