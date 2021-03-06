# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf

class UnpickledVariables:
    """
    A storage for variables values from a pickled model.
    """
    def __init__(self, variables):
        """
        Create object.
        :param variables a list of (name, value) tuples from the pickle file.
        """
        self._variables = dict(variables)

        # A prefix for variable names used to search in the storage.
        self.name_prefix = ''

        num_parameters = 0
        # Check min and max values of the variables, to find out whether a conversion to 16 bit is possible.
        # See https://www.tensorflow.org/js/guide/platform_environment
        min_var = 9e99
        min_nz_var = 9e99
        max_var = -9e99
        for value in self._variables.values():
            num_parameters += value.size
            min_value = abs(value.min())
            min_var = min(min_value, min_var)
            if min_value > 0:
                min_nz_var = min(min_value, min_nz_var)
            max_var = max(abs(value.max()), max_var)

        print(f'Total number of unpickled parameters: {num_parameters}, min: {min_var}, min nz: {min_nz_var}, max: {max_var}')

    def get(self, name):
        """
        Get value of the variable with the given name, taking name_prefix into account.
        """
        key = self.name_prefix + '/' + name
        value = self._variables[key]
        # Remove the variable to ensure that we read each one exactly once.
        del self._variables[key]
        return value


def lerp(a, b, t):
    return a + (b - a) * t


class PixelNormLayer(tf.keras.layers.Layer):
    """
    Pixelwise feature vector normalization.
    """
    def call(self, x):
        return x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + 1e-8)


def get_weight_std(shape, gain):
    fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)  # He init
    # print(f'{shape} {fan_in} {std}')
    return std


def make_conv2d(x, filters, kernel_size, variables, activation=None, gain=np.sqrt(2), factor=1, bias=0):
    """
    Create and initialize a Conv2D layer for given parameters.
    """

    conv_layer = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        activation=activation,
        padding='same',
        name=variables.name_prefix)
    y = conv_layer(x)

    std = get_weight_std((kernel_size, kernel_size, x.shape[-1], filters), gain=gain)
    weight_value = variables.get('weight') * std * factor
    bias_value = variables.get('bias') * factor + bias
    conv_layer.kernel.assign(weight_value)
    conv_layer.bias.assign(bias_value)

    return y


def G_paper(
    latents_in,                         # First input: Latent vectors [minibatch, latent_size].
    variables,                          # Unpickled variables
    num_channels        = 1,            # Number of output color channels. Overridden based on dataset.
    resolution          = 32,           # Output resolution. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    **kwargs                            # Used to check a possibly pickled unsupported in TF2 version values.
    ):
    """
    Generator network used in the paper.
    """

    def check_arg(name, expected_value):
        if name in kwargs and kwargs[name] != expected_value:
            raise ValueError('TF2 version does not support this value')

    check_arg('fused_scale', False)
    check_arg('use_leakyrelu', True)
    check_arg('use_pixelnorm', True)
    check_arg('pixelnorm_epsilon', 1e-8)
    check_arg('label_size', 0)
    check_arg('latent_size', None)
    check_arg('dtype', 'float32')
    check_arg('normalize_latents', True)
    check_arg('use_wscale', True)

    variables = UnpickledVariables(variables)

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage):
        return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    act = tf.nn.leaky_relu
    
    # Building blocks.
    def block(x, res):  # res = 2..resolution_log2
        name_prefix = f'{2**res}x{2**res}'
        if res == 2:  # 4x4
            x = PixelNormLayer()(x)

            variables.name_prefix = name_prefix + '/Dense'
            dense_layer = tf.keras.layers.Dense(nf(res - 1) * 16, input_shape=x.shape[1:],
                                                use_bias=False, name=variables.name_prefix)
            std = get_weight_std((x.shape[1], dense_layer.units), gain=np.sqrt(2)/4)
            x = dense_layer(x)

            weight_value = variables.get('weight') * std
            dense_layer.kernel.assign(weight_value)
            x = tf.keras.layers.Reshape((nf(res - 1), 4, 4))(x)
            x = tf.transpose(x, [0, 2, 3, 1])
            bias_value = variables.get('bias').reshape(1, 1, -1)
            bias = tf.Variable(bias_value, name=variables.name_prefix + '/bias')
            x = act(tf.keras.layers.Add()([x, bias]))
            x = PixelNormLayer()(x)

            variables.name_prefix = name_prefix + '/Conv'
            x = make_conv2d(x, nf(res - 1), 3, variables, act)
            x = PixelNormLayer()(x)

        else:  # 8x8 and up
            x = tf.keras.layers.UpSampling2D(name=name_prefix)(x)
            variables.name_prefix = name_prefix + '/Conv0'
            x = make_conv2d(x, nf(res - 1), 3, variables, act)
            x = PixelNormLayer()(x)

            variables.name_prefix = name_prefix + '/Conv1'
            x = make_conv2d(x, nf(res - 1), 3, variables, act)
            x = PixelNormLayer()(x)

        return x

    def to_rgb(x, res):  # res = 2..resolution_log2
        lod = resolution_log2 - res
        variables.name_prefix = f'ToRGB_lod{lod}'
        # As this is the last layer, apply the factor and bias to convert output range
        # from [-1, 1] to [0, 1].
        return make_conv2d(x, num_channels, 1, variables, None, gain=1,
                           factor=0.5,
                           bias=0.5)

    # Recursive structure: complex but efficient.
    def grow(x, res, lod):
        y = block(x, res)
        if lod > 0:
            img = grow(y, res + 1, lod - 1)
        else:
            img = to_rgb(y, res)
        return img

    images_out = grow(latents_in, 2, resolution_log2 - 2)
    return images_out
