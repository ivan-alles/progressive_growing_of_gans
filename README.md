# A Port of the Progressive Growing of GANs Network to TensorFlow 2

This repo is a fork of the great work **[Progressive Growing of GANs for Improved Quality, Stability, and Variation
](https://github.com/tkarras/progressive_growing_of_gans)** by **Tero Karras** (NVIDIA), **Timo Aila** (NVIDIA), **Samuli Laine** (NVIDIA), **Jaakko Lehtinen** (NVIDIA and Aalto University).

I ported the generator model of the network to TensorFlow 2 format, using `tf.keras` layers wherever possible. Now the inference runs natively on TensorFlow 2. The other parts of the newtork and the training are not re-implemented.

## Setup

1. Optionally install the required drivers, etc. for **[TensorFlow 2](https://www.tensorflow.org/install/gpu)**.
2. Clone the repository.
3. Install the dependencies with `pipenv sync`.
4. Activate the pipenv environment with `pipenv shell`.

## Extract the generator model and save it as TensorFlow 2 Keras model

1. Download [`karras2018iclr-celebahq-1024x1024.pkl`](https://drive.google.com/open?id=188K19ucknC6wg1R6jbuPEhTq9zoufOx4) from [`networks/tensorflow-version`](https://drive.google.com/open?id=15hvzxt_XxuokSmj0uO4xxMTMWVc0cIMU) and place it in the same directory as the script.
2. Run `python extract_generator.py karras2018iclr-celebahq-1024x1024.pkl`
3. The script should generate 10 images. They are slightly different from originals, as we reimplemented the layers of the network.
4. The script will also save the TF2 generator as a `tf.keras` model under `karras2018iclr-celebahq-1024x1024.tf` folder.

## Test a TensorFlow 2 Keras model
5. To test the TF2 model, run `python test_generator.py karras2018iclr-celebahq-1024x1024.tf`. 
6. It should generate the same 10 images as before.
