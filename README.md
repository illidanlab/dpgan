## DPGAN

Tensorflow implementation of DPGAN.

How to run (an example):

```
MNIST data:
python wgan.py --data mnist --model mlp --gpus 0

CelebA data:
python wgan_facetest.py --data face_test --model dcgan --gpus 0

EHR data:
put medWGAN.py PATIENTS.csv.matrix in same folder:
python medWGAN.py PATIENTS.csv.matrix 'path to output' --data_type='binary'
```
