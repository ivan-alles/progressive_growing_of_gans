"""
    Extract the generator model from the original network.
    This file is based on the import_example.py.
"""
import os
import pickle
import sys

import numpy as np
import tensorflow as tf
import PIL.Image

import tfutil2

#  Needed for tf.compat.v1.placeholder
tf.compat.v1.disable_eager_execution()

#  Needed for the original code to run. The pre-trained models are pickled with the code.
#  We do not want to change it, so the functions they need to the tf module.
tf.get_default_graph = tf.compat.v1.get_default_graph
tf.AUTO_REUSE = tf.compat.v1.AUTO_REUSE
tf.placeholder = tf.compat.v1.placeholder
tf.variable_scope = tf.compat.v1.variable_scope
tf.get_variable_scope = tf.compat.v1.get_variable_scope
tf.get_variable = tf.compat.v1.get_variable
tf.rsqrt = tf.compat.v1.rsqrt
tf.initializers.random_normal = tf.compat.v1.initializers.random_normal
tf.global_variables = tf.compat.v1.global_variables
tf.trainable_variables = tf.compat.v1.trainable_variables
tf.get_default_session = tf.compat.v1.get_default_session
tf.assign = tf.compat.v1.assign


# Initialize TensorFlow session.
tf.compat.v1.InteractiveSession()

class MyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'tfutil' and name == 'Network':
            return tfutil2.Network
        return super().find_class(module, name)

input_file_name = sys.argv[1]

with open(input_file_name, 'rb') as file:
    tfutil2.UNPICKLE_COUNTER = 0
    unpickler = MyUnpickler(file)
    G, D, Gs = unpickler.load()

# Generate latent vectors.
latents = np.random.RandomState(1000).randn(1000, *Gs.latents_in.shape[1:])  # 1000 random latents
latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]  # hand-picked top-10

# Run the generator to produce a set of images.
# images = Gs.run(latents)
images = Gs.run_keras(latents)

# Convert images to PIL-compatible format.
images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC

# Save images as PNG.
for idx in range(images.shape[0]):
    PIL.Image.fromarray(images[idx], 'RGB').save('img%d.png' % idx)

base_name, ext = os.path.splitext(input_file_name)

if hasattr(Gs, 'keras_model'):
    tf.keras.utils.plot_model(Gs.keras_model, to_file=base_name + '.svg', dpi=50, show_shapes=True)
    # Gs.keras_model.save(base_name + '.tf')
