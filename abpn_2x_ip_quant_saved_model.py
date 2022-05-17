#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ReLU, Lambda, Add
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_normal
import tensorflow.keras.backend as K

import tensorflow_model_optimization as tfmot

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
model.load_weights('models/checkpoint/abpn_x2_ip/checkpoint').expect_partial()
model.save('models/saved_model/abpn_x2_ip')
converter = tf.lite.TFLiteConverter.from_saved_model('models/saved_model/abpn_x2_ip')
tflite_model = converter.convert()

# Save the model.
with open('models/tflite/abpn_x2_ip.tflite', 'wb') as f:
  f.write(tflite_model)

model = tfmot.quantization.keras.quantize_model(model)
model.load_weights('models/checkpoint/abpn_x2_ip_quant/checkpoint').expect_partial()
model.save('models/saved_model/abpn_x2_ip_quant')

converter = tf.lite.TFLiteConverter.from_saved_model('models/saved_model/abpn_x2_ip_quant')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_quant = converter.convert()

# Save the model.
with open('models/tflite_quant/abpn_x2_ip_quant.tflite', 'wb') as f:
  f.write(tflite_model_quant)
