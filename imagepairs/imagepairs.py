"""imagepairs dataset."""
import os

import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
* ImagePairs Dataset

ImagePairs is composed of pairs of images of the exact same scene
using two different cameras: one low-resolution image (1,752 × 1,166 pixels)
and one high-resolution image that was exactly twice as big in each
dimension (3,504 × 2,332 pixels).
"""

_CITATION = """
@article{https://doi.org/10.48550/arxiv.2004.08513,
  doi = {10.48550/ARXIV.2004.08513},
  url = {https://arxiv.org/abs/2004.08513},
  author = {Joze, Hamid Reza Vaezi and Zharkov, Ilya and Powell, Karlton and Ringler, Carl and Liang, Luming and Roulston, Andy and Lutz, Moshe and Pradeep, Vivek},
  keywords = {Image and Video Processing (eess.IV), Computer Vision and Pattern Recognition (cs.CV), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {ImagePairs: Realistic Super Resolution Dataset via Beam Splitter Camera Rig},
  publisher = {arXiv},
  year = {2020},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
"""


class Imagepairs(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for imagepairs dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(1146, 1737, 3)),
            'image_gt': tfds.features.Image(shape=(2292, 3474, 3)),
        }),
        supervised_keys=('image', 'image_gt'),  # Set to `None` to disable
        homepage='https://www.microsoft.com/applied-sciences/projects/imagepairs',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # train_path = dl_manager.download_and_extract('https://download.microsoft.com/download/4/3/b/43ba4267-3d16-4fc7-a6f3-6590ba61d884/ImagePairs - Training Data (14 of 14).zip')
    train_set = []
    for i in range(14):
      train_set.append(f'https://download.microsoft.com/download/4/3/b/43ba4267-3d16-4fc7-a6f3-6590ba61d884/ImagePairs - Training Data ({i+1} of 14).zip')

    test_set = []
    for i in range(5):
      test_set.append(f'https://download.microsoft.com/download/3/0/f/30f85f48-82f1-4637-9827-cf9d2f96f335/ImagePairs - Test Data ({i+1} of 5).zip')

    test_path = dl_manager.download_and_extract(test_set)
    train_path = dl_manager.download_and_extract(train_set)

    return {
        'train': self._generate_examples(train_path),
        'test': self._generate_examples(test_path),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    if not isinstance(path, list):
      for root, _, files in tf.io.gfile.walk(path):
        for f in files:
          if f.endswith("_gt.png"):
            g = f.replace("_gt.png", ".png")
            # print(f, g)
            yield f, {
              'image': root + "/" + g,
              'image_gt': root + "/" + f,
            }
    else:
      for a in path:
        # print(type(a))
        for root, _, files in tf.io.gfile.walk(a):
          for f in files:
            if f.endswith("_gt.png"):
              g = f.replace("_gt.png", ".png")
              # print(f, g)
              if os.path.exists(root + "/" + g):
                yield f, {
                  'image': root + "/" + g,
                  'image_gt': root + "/" + f,
                }
              else:
                continue
