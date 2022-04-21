```python
tfds.core.DatasetInfo(
    name='imagepairs',
    full_name='imagepairs/1.0.0',
    description="""
    * ImagePairs Dataset
    
    ImagePairs is composed of pairs of images of the exact same scene
    using two different cameras: one low-resolution image (1,752 × 1,166 pixels)
    and one high-resolution image that was exactly twice as big in each
    dimension (3,504 × 2,332 pixels).
    """,
    homepage='https://www.microsoft.com/applied-sciences/projects/imagepairs',
    data_path='/home/freedom/tensorflow_datasets/imagepairs/1.0.0',
    download_size=148.64 GiB,
    dataset_size=162.46 GiB,
    features=FeaturesDict({
        'image': Image(shape=(1146, 1737, 3), dtype=tf.uint8),
        'image_gt': Image(shape=(2292, 3474, 3), dtype=tf.uint8),
    }),
    supervised_keys=('image', 'image_gt'),
    disable_shuffling=False,
    splits={
        'test': <SplitInfo num_examples=2827, num_shards=512>,
        'train': <SplitInfo num_examples=8591, num_shards=1024>,
    },
    citation="""@article{https://doi.org/10.48550/arxiv.2004.08513,
      doi = {10.48550/ARXIV.2004.08513},
      url = {https://arxiv.org/abs/2004.08513},
      author = {Joze, Hamid Reza Vaezi and Zharkov, Ilya and Powell, Karlton and Ringler, Carl and Liang, Luming and Roulston, Andy and Lutz, Moshe and Pradeep, Vivek},
      keywords = {Image and Video Processing (eess.IV), Computer Vision and Pattern Recognition (cs.CV), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
      title = {ImagePairs: Realistic Super Resolution Dataset via Beam Splitter Camera Rig},
      publisher = {arXiv},
      year = {2020},
      copyright = {arXiv.org perpetual, non-exclusive license}
    }""",
)
```
