#!/usr/bin/env python

import sys
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ReLU, Lambda, Add
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_normal
import tensorflow.keras.backend as K

import tensorflow_datasets as tfds
import tensorflow_model_optimization as tfmot

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
      print(e)

ds_div2k = tfds.load('div2k', shuffle_files=True)

ds_train = ds_div2k['train']
ds_test = ds_div2k['validation']


def abpn(scale=3, in_channels=3, num_fea=28, m=4, out_channels=3):
    inp = Input(shape=(None, None, 3))
    upsample_func = Lambda(lambda x_list: tf.concat(x_list, axis=3))
    upsampled_inp = upsample_func([inp for x in range(scale**2)])

    # Feature extraction
    x = Conv2D(num_fea, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(inp)

    for i in range(m):
        x = Conv2D(num_fea, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)

    # Pixel-Shuffle
    x = Conv2D(out_channels*(scale**2), 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
    x = Conv2D(out_channels*(scale**2), 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
    x = Add()([upsampled_inp, x])

    depth_to_space = Lambda(lambda x: tf.nn.depth_to_space(x, scale))
    out = depth_to_space(x)
    clip_func = Lambda(lambda x: K.clip(x, 0., 255.))
    out = clip_func(out)

    return Model(inputs=inp, outputs=out, name='abpn')

model = abpn(2) # ABPN super resolution scale = 2
model = tfmot.quantization.keras.quantize_model(model)
model.load_weights('models/checkpoint/abpn_x2_ip_quant/checkpoint').expect_partial()

import imagepairs
ip_test = tfds.load('imagepairs', split='test')

psnrs = []
ssims = []

for a in ip_test:
    sr = tf.cast(model(tf.expand_dims(a['image'], 0)), tf.uint8)
    sr_q = tf.image.central_crop(sr, 0.25)
    hr_q = tf.image.central_crop(a['image_gt'], 0.25)
    psnrs.append(tf.image.psnr(sr_q, hr_q, max_val=255.0))
    ssims.append(tf.image.ssim(sr_q, hr_q, max_val=255.0))

print("ImagePairs (PSNR, SSIM) trained with ImagePairs:", tf.reduce_mean(psnrs), tf.reduce_mean(ssims))
