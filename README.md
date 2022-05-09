# some_super_resolution_tflite_models
Some super resolutions models converted to TFLite

# files
## models/saved_model
TensorFlow models in saved_model format. 
* models/saved_model/{[srgan](models/saved_model/srgan),[edsr](models/saved_model/edsr)}: from Martin Krasser's [super resolution](https://github.com/krasserm/super-resolution) repo
* [models/saved_model/wdsr](models/saved_model/wdsr): from [offical WDSR](https://github.com/ychfan/tf_estimator_barebone/blob/master/docs/super_resolution.md)

* [models/saved_model/esrgan](models/saved_model/esrgan): from [TFHub ESRGAN](https://tfhub.dev/captain-pool/esrgan-tf2/1)

## models/tflite
fp32 tflite models

## models/tflite_quant
post-training full integer quantized tflite models

## scripts used to generate tflite models
[srgan.py](srgan.py), [edsr.py](edsr.py), [wdsr.py](wdsr.py), [esrgan.py](esrgan.py)

## Jupyter notebooks used to check tflite models
[test srgan tflite.ipynb](test%20srgan%20tflite.ipynb), [test edsr tflite.ipynb](test%20edsr%20tflite.ipynb), [test wdsr tflite.ipynb](test%20wdsr%20tflite.ipynb), [test esrgan tflite.ipynb](test%20esrgan%20tflite.ipynb)

## other files
[data.py](data.py) from Martin Krasser's [super resolution](https://github.com/krasserm/super-resolution) repo, to use DIV2K dataset

[A self-contained Jupyter notebook that can be used train ABPN](train_abpn.ipynb)
* PSNR, SSIM evaluated with DIV2K: 33.37, 0.93
* PSNR, SSIM evaluated with ImagePairs: 22.06, 0.73

[train_abpn_ip.py](train_abpn_ip.py): a self-contained script to train ABPN with ImagePairs dataset
* (PSNR, SSIM) evaluated with ImagePairs: (23.95284, 0.76988685)

demo/*.png: low resolution cropped png from Martin Krasser's [super resolution](https://github.com/krasserm/super-resolution) repo
