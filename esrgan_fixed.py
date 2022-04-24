import tensorflow as tf
import numpy as np
from data import DIV2K

# 876 x 583 --> 3504 x 2332

model = tf.saved_model.load('models/saved_model/esrgan')
concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([1, 583, 876, 3])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

tflite_model = converter.convert()
with open('models/tflite/esrgan_fixed.tflite', 'wb') as f:
  f.write(tflite_model)

scale = 4
# Downgrade operator
downgrade = 'bicubic'
div2k_valid = DIV2K(scale=scale, subset='valid', downgrade=downgrade)
valid_ds = div2k_valid.dataset(batch_size=1, random_transform=False, repeat_count=1)

def representative_data_gen():
  for input_value in valid_ds.take(1):
    resized = tf.image.resize(input_value[0], (583, 876))
    yield [np.float32(resized)]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model_quant = converter.convert()

# Save the model
with open('models/tflite_quant/esrgan_fixed_quant.tflite', 'wb') as f:
  f.write(tflite_model_quant)
