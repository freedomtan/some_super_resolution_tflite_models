# some_super_resolution_tflite_models
Some super resolutions models converted to TFLite

# files
## models/saved_model
TensorFlow models in saved_model format. 
* models/saved_model/{[srgan](models/saved_model/srgan),[edsr](models/saved_model/edsr)}: from Martin Krasser's [super resolution](https://github.com/krasserm/super-resolution) repo
* [models/saved_model/wdsr](models/saved_model/wdsr): from [offical WDSR](https://github.com/ychfan/tf_estimator_barebone/blob/master/docs/super_resolution.md)
## models/tflite
fp32 tflite models

## models/tflite_quant
post-training full integer quantized tflite models

## scripts used to generate tflite models
[srgan.py](srgan.py), [edsr.py](edsr.py), [wdsr.py](wdsr.py)

## Jupyter notebooks used to check tflite models
[test srgan tflite.ipynb](test%20srgan%20tflite.ipynb), [test edsr tflite.ipynb](test%20edsr%20tflite.ipynb), [test wdsr tflite.ipynb](test%20wdsr%20tflite.ipynb)
