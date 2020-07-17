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

def get_weight(name_prefix, shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    weight = tf.Variable(VARIABLES.get(name_prefix + '/weight'), name='weight')
    if use_wscale:
        if fan_in is None:
            fan_in = np.prod(shape[:-1])
        std = gain / np.sqrt(fan_in)  # He init
        weight = weight * std
    return weight

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, name_prefix, fmaps, gain=np.sqrt(2), use_wscale=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d for d in x.shape[1:]])])
    w = get_weight(name_prefix, [x.shape[1], fmaps], gain=gain, use_wscale=use_wscale)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d(x, name_prefix, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight(name_prefix, [kernel, kernel, x.shape[1], fmaps], gain=gain, use_wscale=use_wscale)
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
    use_wscale          = True,         # Enable equalized learning rate?
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
            x = dense(x, name_prefix=name_prefix + '/Dense', fmaps=nf(res-1)*16, gain=np.sqrt(2)/4, use_wscale=use_wscale) # override gain to match the original Theano implementation
            x = tf.reshape(x, [-1, nf(res-1), 4, 4])
            x = PixelNormLayer()(act(apply_bias(x, name_prefix=name_prefix + '/Dense')))

            x = PixelNormLayer()(act(
                apply_bias(conv2d(x, name_prefix=name_prefix + '/Conv', fmaps=nf(res-1), kernel=3, use_wscale=use_wscale),
                name_prefix=name_prefix + '/Conv')))
        else: # 8x8 and up
            x = tf.keras.layers.UpSampling2D(data_format='channels_first', name=name_prefix)(x)
            x = PixelNormLayer()(act(apply_bias(
                conv2d(x, name_prefix=name_prefix + '/Conv0', fmaps=nf(res-1), kernel=3, use_wscale=use_wscale),
                name_prefix=name_prefix + '/Conv0')))

            x = PixelNormLayer()(act(apply_bias(
                conv2d(x, name_prefix=name_prefix + '/Conv1', fmaps=nf(res-1), kernel=3, use_wscale=use_wscale),
                name_prefix=name_prefix + '/Conv1')))
        return x
    def torgb(x, res): # res = 2..resolution_log2
        lod = resolution_log2 - res
        name_prefix = f'ToRGB_lod{lod}'
        return apply_bias(
            conv2d(x, name_prefix, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale),
            name_prefix)

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
