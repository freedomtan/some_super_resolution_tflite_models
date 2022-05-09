#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/freedomtan/some_super_resolution_tflite_models/blob/main/train_abpn.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ReLU, Lambda, Add
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_normal

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
import tensorflow.keras.backend as K

from tensorflow.data import AUTOTUNE

# Load ImagePairs from [TensorFlow Datasets](https://www.tensorflow.org/datasets)
import tensorflow_datasets as tfds
import imagepairs

ds_ip = tfds.load('imagepairs', shuffle_files=True)
ds_train = ds_ip['train']
ds_test = ds_ip['test']

# ABPN from the authors' [repo](https://github.com/NJU-Jet/SR_Mobile_Quantization), I fixed `upsample_in = ...`

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


# Preprocessing functions from @krasserm's super [resolution repo](https://github.com/krasserm/super-resolution)
# 
# patch size for training: 96x96 (48x48 -> 96x96)
# training batch size: 16

def random_crop(ds, hr_crop_size=96, scale=2):
    lr_img = ds['image']
    hr_img = ds['image_gt']
    
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped

def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)


def get_patches(ds_train):
    batch_size = 16
    repeat_count = None
    
    ds = ds_train.map(lambda ds: random_crop(ds), num_parallel_calls=AUTOTUNE)
    ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
    ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.cache()
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def wrapper(ds):
    return ds['image'], ds['image_gt']

def get_part(ds_test):
    batch_size = 1
    repeat_count = 1
    
    ds = ds_test.map(lambda ds: wrapper(ds), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds


model = abpn(2) # ABPN super resolution scale = 2
# if there is pre-trained checkpoint, do something like
# model.load_weights('/tmp/abpn_ip/checkpoint')
# otherwise, train from scratch

def psnr_fn(y, x):
    psnr = tf.image.psnr(x, y, max_val = 255.0)
    return psnr

mean_psnr = tf.keras.metrics.MeanMetricWrapper(fn=psnr_fn, name='mean_psnr')

learning_rate = PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5])
model.compile(Adam(learning_rate=learning_rate),loss='mean_absolute_error', metrics = [mean_psnr])

checkpoint_filepath = '/tmp/abpn_ip/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_mean_psnr',
    mode='max',
    save_best_only=True)

tb_callback = tf.keras.callbacks.TensorBoard('/tmp/abpn_ip/logs', update_freq=1)

ds_patches = get_patches(ds_train)
ds_test_batched = get_part(ds_test)

history = model.fit(
    ds_patches,
    epochs=300,
    steps_per_epoch=1000,
    validation_data = ds_test_batched.take(300),
    callbacks = [model_checkpoint_callback, tb_callback]
    )

