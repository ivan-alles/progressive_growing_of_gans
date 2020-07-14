"""
    Extract the generator model from the original network.
    This file is based on the import_example.py.
"""
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image

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

# Import official CelebA-HQ networks.
with open('karras2018iclr-celebahq-1024x1024.pkl', 'rb') as file:
    #  Replace calls like shape[0].value because shape now contains integers.
    text = file.read()
    text = text.replace(b'.value', b'      ')
    G, D, Gs = pickle.loads(text)
    del text

# Generate latent vectors.
latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:])  # 1000 random latents
latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]  # hand-picked top-10

# Run the generator to produce a set of images.
images = Gs.run_simple(latents)
# images = Gs.run_keras(latents)

# Convert images to PIL-compatible format.
images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC

# Save images as PNG.
for idx in range(images.shape[0]):
    PIL.Image.fromarray(images[idx], 'RGB').save('img%d.png' % idx)
