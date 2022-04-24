import tensorflow as tf
import numpy as np
from data import DIV2K

converter = tf.lite.TFLiteConverter.from_saved_model('models/saved_model/abpn_x2')
tflite_model = converter.convert()
with open('models/tflite/abpn_x2.tflite', 'wb') as f:
  f.write(tflite_model)

# Super-resolution factor
scale = 2
# Downgrade operator
downgrade = 'bicubic'
div2k_valid = DIV2K(scale=scale, subset='valid', downgrade=downgrade)
valid_ds = div2k_valid.dataset(batch_size=1, random_transform=False, repeat_count=1)

def representative_data_gen():
  for input_value in valid_ds.take(100):
    # Model has only one input so each data point has one element.
    yield [np.float32(input_value[0])]

converter.representative_dataset = representative_data_gen
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model_quant = converter.convert()

# Save the model.
with open('models/tflite_quant/abpn_x2_quant.tflite', 'wb') as f:
  f.write(tflite_model_quant)

print("converted")
