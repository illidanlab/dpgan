## Wasserstein GAN

Tensorflow implementation of Wasserstein GAN.

How to run (an example):

MNIST data:
python wgan.py --data mnist --model mlp --gpus 0

CelebA data:
python wgan_facetest.py --data face_test --model dcgan --gpus 0
