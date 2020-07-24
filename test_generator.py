"""
    Test generator inference from a saved tf.keras model.
"""
import sys
import os

import numpy as np
import tensorflow as tf
import PIL.Image

generator = tf.keras.models.load_model(sys.argv[1])

# Generate latent vectors.
latents = np.random.RandomState(1000).randn(1000, *generator.input.shape[1:])  # 1000 random latents
latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]  # hand-picked top-10

images = generator.predict(latents)

# Convert to bytes in range [0, 255]
images = np.rint(np.clip(images, 0, 1) * 255.0).astype(np.uint8)

# Save images as PNG.
for idx in range(images.shape[0]):
    PIL.Image.fromarray(images[idx], 'RGB').save('img%d.png' % idx)

