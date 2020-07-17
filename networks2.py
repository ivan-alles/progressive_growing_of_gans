# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf

VARIABLES = None

#----------------------------------------------------------------------------

def lerp(a, b, t):
    return a + (b - a) * t

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(name_prefix, shape, gain=np.sqrt(2)):
    weight = tf.Variable(VARIABLES.get(name_prefix + '/weight'), name='weight')
    fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)  # He init
    print(f'{name_prefix} {shape} {fan_in} {std}')
    weight = weight * std
    return weight

#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d(x, name_prefix, fmaps, kernel, gain=np.sqrt(2)):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight(name_prefix, [kernel, kernel, x.shape[1], fmaps], gain=gain)
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x, name_prefix):
    b = tf.Variable(VARIABLES.get(name_prefix + '/bias'), name='bias')
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1])

class PixelNormLayer(tf.keras.layers.Layer):
    """
    Pixelwise feature vector normalization.
    """
    def call(self, x):
        return x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8)

def get_weight_std(shape, gain):
    fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)  # He init
    return std

def make_conv2d(x, filters, kernel_size, name_prefix, activation=None, gain=np.sqrt(2)):
    y = apply_bias(conv2d(x, name_prefix=name_prefix, fmaps=filters, kernel=kernel_size, gain=gain),
                   name_prefix=name_prefix)
    if activation is not None:
        y = activation(y)
    return y

#----------------------------------------------------------------------------
# Generator network used in the paper.

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

    global VARIABLES
    VARIABLES = variables

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage):
        return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    act = tf.nn.leaky_relu
    
    # Building blocks.
    def block(x, res): # res = 2..resolution_log2
        name_prefix = f'{2**res}x{2**res}'
        if res == 2:  # 4x4
            x = PixelNormLayer()(x)

            dense_layer = tf.keras.layers.Dense(nf(res - 1) * 16, input_shape=x.shape[1:], use_bias=False)
            std = get_weight_std((x.shape[1], dense_layer.units), gain=np.sqrt(2)/4)
            x = dense_layer(x)
            weight_value = VARIABLES.get(name_prefix + '/Dense/weight') * std
            dense_layer.kernel.assign(weight_value)
            x = tf.keras.layers.Reshape((nf(res - 1), 4, 4))(x)
            bias_value = VARIABLES.get(name_prefix + '/Dense/bias').reshape(1, -1, 1, 1)
            bias = tf.Variable(bias_value, name=name_prefix + '/Dense/bias')
            x = act(tf.keras.layers.Add()([x, bias]))
            x = PixelNormLayer()(x)

            x = make_conv2d(x, nf(res - 1), 3, name_prefix + '/Conv', act)
            x = PixelNormLayer()(x)
        else: # 8x8 and up
            x = tf.keras.layers.UpSampling2D(data_format='channels_first', name=name_prefix)(x)
            x = make_conv2d(x, nf(res - 1), 3, name_prefix + '/Conv0', act)
            x = PixelNormLayer()(x)

            x = make_conv2d(x, nf(res - 1), 3, name_prefix + '/Conv1', act)
            x = PixelNormLayer()(x)

        return x
    def torgb(x, res): # res = 2..resolution_log2
        lod = resolution_log2 - res
        return make_conv2d(x, num_channels, 1, f'ToRGB_lod{lod}', None, gain=1)

    # Recursive structure: complex but efficient.
    def grow(x, res, lod):
        y = block(x, res)
        if lod > 0:
            img = grow(y, res + 1, lod - 1)
        else:
            img = tf.keras.layers.UpSampling2D(size=2 ** lod, data_format='channels_first')(torgb(y, res))
        return img

    images_out = grow(latents_in, 2, resolution_log2 - 2)
    return images_out
