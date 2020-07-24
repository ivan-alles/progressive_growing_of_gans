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
images = Gs.keras_model.predict(latents)

# Convert to bytes in range [0, 255]
images = np.rint(np.clip(images, 0, 1) * 255.0).astype(np.uint8)

# Save images as PNG.
for idx in range(images.shape[0]):
    PIL.Image.fromarray(images[idx], 'RGB').save('img%d.png' % idx)

base_name, ext = os.path.splitext(input_file_name)

if hasattr(Gs, 'keras_model'):
    tf.keras.utils.plot_model(Gs.keras_model, to_file=base_name + '.svg', dpi=50, show_shapes=True)

    Gs.keras_model.save(base_name + '.tf')

    with open(base_name + '.txt', 'w') as f:
        Gs.keras_model.summary(print_fn=lambda l: print(l, file=f))

    # with open(base_name + '.json', 'w') as f:
    #     s = Gs.keras_model.to_json()
    #     f.write(s)


    # Gs.keras_model.save_weights(base_name + '-weights.tf')
