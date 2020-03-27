# Adding a Dataset of Your Own to TFDS

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20Deployment/Course%203%20-%20TensorFlow%20Datasets/Week%204/Exercises/TFDS_Week4_Exercise.ipynb">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
    Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20Deployment/Course%203%20-%20TensorFlow%20Datasets/Week%204/Exercises/TFDS_Week4_Exercise.ipynb">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub</a>
  </td>
</table>

## Setup


```python
try:
    %tensorflow_version 2.x
except:
    pass
```


```python
import os
import textwrap
import scipy.io
import pandas as pd

import tensorflow as tf

print("\u2022 Using TensorFlow Version:", tf.__version__)
```

## IMDB Faces Dataset

This is the largest publicly available dataset of face images with gender and age labels for training.

Source: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

The IMDb Faces dataset provides a separate .mat file which can be loaded with Matlab containing all the meta information. The format is as follows:  
**dob**: date of birth (Matlab serial date number)  
**photo_taken**: year when the photo was taken  
**full_path**: path to file  
**gender**: 0 for female and 1 for male, NaN if unknown  
**name**: name of the celebrity  
**face_location**: location of the face (bounding box)  
**face_score**: detector score (the higher the better). Inf implies that no face was found in the image and the face_location then just returns the entire image  
**second_face_score**: detector score of the face with the second highest score. This is useful to ignore images with more than one face. second_face_score is NaN if no second face was detected.  
**celeb_names**: list of all celebrity names  
**celeb_id**: index of celebrity name

Here you can download the raw images and the metadata. We also provide a version with the cropped faces (with 40% margin). This version is much smaller.


```python
# Download and extract the IMDB Faces dataset
!wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar
!tar xf imdb_crop.tar
```

Next, let's inspect the dataset

## Exploring the Data


```python
# Inspect the directory structure
files = os.listdir('imdb_crop')
print(textwrap.fill(' '.join(sorted(files)), 80))
```

**NOTE:** In the code below we have set `/content/` as the path to the `/imdb_crop/imdb.mat` file. This will work in Google's Colab environment without any modifications. However, if you are running this notebook locally, you should change `/content/` to the appropriate path to the `/imdb_crop/imdb.mat` file on your computer.


```python
# Inspect the meta data
meta = scipy.io.loadmat('/content/imdb_crop/imdb.mat')
```


```python
meta
```

## Extraction

Let's clear up the clutter by going to the metadata's most useful key (imdb) and start exploring all the other keys inside it


```python
root = meta['imdb'][0, 0]
```


```python
desc = root.dtype.descr
desc
```


```python
# EXERCISE: Fill in the missing code below.

full_path = root["full_path"][0]

# Do the same for other attributes
names = # YOUR CODE HERE
dob = # YOUR CODE HERE
gender = # YOUR CODE HERE
photo_taken = # YOUR CODE HERE
face_score = # YOUR CODE HERE
face_locations = # YOUR CODE HERE
second_face_score = # YOUR CODE HERE
celeb_names = # YOUR CODE HERE
celeb_ids = # YOUR CODE HERE

print('Filepaths: {}\n\n'
      'Names: {}\n\n'
      'Dates of birth: {}\n\n'
      'Genders: {}\n\n'
      'Years when the photos were taken: {}\n\n'
      'Face scores: {}\n\n'
      'Face locations: {}\n\n'
      'Second face scores: {}\n\n'
      'Celeb IDs: {}\n\n'
      .format(full_path, names, dob, gender, photo_taken, face_score, face_locations, second_face_score, celeb_ids))
```


```python
print('Celeb names: {}\n\n'.format(celeb_names))
```

Display all the distinct keys and their corresponding values


```python
names = [x[0] for x in desc]
names
```


```python
values = {key: root[key][0] for key in names}
values
```

## Cleanup

Pop out the celeb names as they are not relevant for creating the records.


```python
del values['celeb_names']
names.pop(names.index('celeb_names'))
```

Let's see how many values are present in each key


```python
for key, value in values.items():
    print(key, len(value))
```

## Dataframe

Now, let's try examining one example from the dataset. To do this, let's load all the attributes that we've extracted just now into a Pandas dataframe


```python
df = pd.DataFrame(values, columns=names)
df.head()
```

The Pandas dataframe may contain some Null values or nan. We will have to filter them later on.


```python
df.isna().sum()
```

# TensorFlow Datasets

TFDS provides a way to transform all those datasets into a standard format, do the preprocessing necessary to make them ready for a machine learning pipeline, and provides a standard input pipeline using `tf.data`.

To enable this, each dataset implements a subclass of `DatasetBuilder`, which specifies:

* Where the data is coming from (i.e. its URL). 
* What the dataset looks like (i.e. its features).  
* How the data should be split (e.g. TRAIN and TEST). 
* The individual records in the dataset.

The first time a dataset is used, the dataset is downloaded, prepared, and written to disk in a standard format. Subsequent access will read from those pre-processed files directly.

## Clone the TFDS Repository

The next step will be to clone the GitHub TFDS Repository. For this particular notebook, we will clone a particular version of the repository. You can clone the repository by running the following command:


```python
!git clone https://github.com/tensorflow/datasets.git -b v1.2.0
```

Next, we set the current working directory to `/content/datasets`.

**NOTE:** Here we have set `/content/` as the path to the `/datasets/` directory. This will work in Google's Colab environment without any modifications. However, if you are running this notebook locally, you should change `/content/` to the appropriate path to the `/datasets/` directory on your computer.


```python
cd /content/datasets
```

Now we will use IPython's `%%writefile` in-built magic command to write whatever is in the current cell into a file. To create or overwrite a file you can use:
```
%%writefile filename
```

Let's see an example:


```python
%%writefile something.py
x = 10
```

Now that the file has been written, let's inspect its contents.


```python
!cat something.py
```

## Define the Dataset with `GeneratorBasedBuilder`

Most datasets subclass `tfds.core.GeneratorBasedBuilder`, which is a subclass of `tfds.core.DatasetBuilder` that simplifies defining a dataset. It works well for datasets that can be generated on a single machine. Its subclasses implement:

* `_info`: builds the DatasetInfo object describing the dataset


* `_split_generators`: downloads the source data and defines the dataset splits


* `_generate_examples`: yields (key, example) tuples in the dataset from the source data

In this exercise, you will use the `GeneratorBasedBuilder`.

### EXERCISE: Fill in the missing code below.


```python
%%writefile tensorflow_datasets/image/imdb_faces.py

# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IMDB Faces dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """\
Follow the URL below and write a description on IMDB Faces dataset.
"""

_URL = ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/")
_DATASET_ROOT_DIR = # Put the name of the dataset root directory here
_ANNOTATION_FILE = # Put the name of annotation file here (.mat file)


_CITATION = """\
Follow the URL and paste the citation of IMDB Faces dataset.
"""

# Source URL of the IMDB faces dataset
_TARBALL_URL = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar"

class ImdbFaces(tfds.core.GeneratorBasedBuilder):
  """IMDB Faces dataset."""

  VERSION = tfds.core.Version("0.1.0")
  
  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        # Describe the features of the dataset by following this url
        # https://www.tensorflow.org/datasets/api_docs/python/tfds/features
        features=tfds.features.FeaturesDict({
            "image": # Create a tfds Image feature here
            "gender":  # Create a tfds Class Label feature here for the two classes (Female, Male)
            "dob": # YOUR CODE HERE
            "photo_taken": # YOUR CODE HERE
            "face_location": # Create a tfds Bounding box feature here
            "face_score": # YOUR CODE HERE
            "second_face_score": # YOUR CODE HERE
            "celeb_id": # YOUR CODE HERE
        }),
        supervised_keys=("image", "gender"),
        urls=[_URL],
        citation=_CITATION)

  def _split_generators(self, dl_manager):
    # Download the dataset and then extract it.
    download_path = dl_manager.download([_TARBALL_URL])
    extracted_path = dl_manager.download_and_extract([_TARBALL_URL])

    # Parsing the mat file which contains the list of train images
    def parse_mat_file(file_name):
      with tf.io.gfile.GFile(file_name, "rb") as f:
        # Add a lazy import for scipy.io and import the loadmat method to 
        # load the annotation file
        dataset = # YOUR CODE HERE
      return dataset

    # Parsing the mat file by using scipy's loadmat method
    # Pass the path to the annotation file using the downloaded/extracted paths above
    meta = parse_mat_file(# YOUR CODE HERE)

    # Get the names of celebrities from the metadata
    celeb_names = # YOUR CODE HERE

    # Create tuples out of the distinct set of genders and celeb names
    self.info.features['gender'].names = # YOUR CODE HERE
    self.info.features['celeb_id'].names = # YOUR CODE HERE

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "image_dir": extracted_path[0],
                "metadata": meta,
            })
    ]

  def _get_bounding_box_values(self, bbox_annotations, img_width, img_height):
    """Function to get normalized bounding box values.

    Args:
      bbox_annotations: list of bbox values in kitti format
      img_width: image width
      img_height: image height

    Returns:
      Normalized bounding box xmin, ymin, xmax, ymax values
    """

    ymin = bbox_annotations[0] / img_height
    xmin = bbox_annotations[1] / img_width
    ymax = bbox_annotations[2] / img_height
    xmax = bbox_annotations[3] / img_width
    return ymin, xmin, ymax, xmax
  
  def _get_image_shape(self, image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    shape = image.shape[:2]
    return shape

  def _generate_examples(self, image_dir, metadata):
    # Add a lazy import for pandas here (pd)
    pd = # YOUR CODE HERE

    # Extract the root dictionary from the metadata so that you can query all the keys inside it
    root = metadata[0, 0]

    """Extract image names, dobs, genders,  
               face locations, 
               year when the photos were taken,
               face scores (second face score too),
               celeb ids
    """
    image_names = root["full_path"][0]
        
    # Do the same for other attributes (dob, genders etc)
    dobs = # YOUR CODE HERE
    genders = # YOUR CODE HERE
    face_locations = # YOUR CODE HERE
    photo_taken_years = # YOUR CODE HERE
    face_scores = # YOUR CODE HERE
    second_face_scores = # YOUR CODE HERE
    celeb_id = # YOUR CODE HERE
        
    # Now create a dataframe out of all the features like you've seen before
    df = # YOUR CODE HERE

    # Filter dataframe by only having the rows with face_scores > 1.0
    df = # YOUR CODE HERE


    # Remove any records that contain Nulls/NaNs by checking for NaN with .isna()
    df = df[~df['genders'].isna()]
    df = # YOUR CODE HERE

    # Cast genders to integers so that mapping can take place
    df.genders = # YOUR CODE HERE

    # Iterate over all the rows in the dataframe and map each feature
    for _, row in df.iterrows():
      # Extract filename, gender, dob, photo_taken, 
      # face_score, second_face_score and celeb_id
      filename = os.path.join(image_dir, _DATASET_ROOT_DIR, row['image_names'][0])
      gender = row['genders']
      dob = row['dobs']
      photo_taken = row['photo_taken_years']
      face_score = row['face_scores']
      second_face_score = row['second_face_scores']
      celeb_id = row['celeb_ids']

      # Get the image shape
      image_width, image_height = self._get_image_shape(filename)
      # Normalize the bounding boxes by using the face coordinates and the image shape
      bbox = self._get_bounding_box_values(row['face_locations'][0], 
                                           image_width, image_height)

      # Yield a feature dictionary 
      yield filename, {
          "image": filename,
          "gender": gender,
          "dob": dob,
          "photo_taken": photo_taken,
          "face_location": # Create a bounding box (BBox) object out of the coordinates extracted
          "face_score": face_score,
          "second_face_score": second_face_score,
          "celeb_id": celeb_id
      }

```

## Add an Import for Registration

All subclasses of `tfds.core.DatasetBuilder` are automatically registered when their module is imported such that they can be accessed through `tfds.builder` and `tfds.load`.

If you're contributing the dataset to `tensorflow/datasets`, you must add the module import to its subdirectory's `__init__.py` (e.g. `image/__init__.py`), as shown below:


```python
%%writefile tensorflow_datasets/image/__init__.py
# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image datasets."""

from tensorflow_datasets.image.abstract_reasoning import AbstractReasoning
from tensorflow_datasets.image.aflw2k3d import Aflw2k3d
from tensorflow_datasets.image.bigearthnet import Bigearthnet
from tensorflow_datasets.image.binarized_mnist import BinarizedMNIST
from tensorflow_datasets.image.binary_alpha_digits import BinaryAlphaDigits
from tensorflow_datasets.image.caltech import Caltech101
from tensorflow_datasets.image.caltech_birds import CaltechBirds2010
from tensorflow_datasets.image.cats_vs_dogs import CatsVsDogs
from tensorflow_datasets.image.cbis_ddsm import CuratedBreastImagingDDSM
from tensorflow_datasets.image.celeba import CelebA
from tensorflow_datasets.image.celebahq import CelebAHq
from tensorflow_datasets.image.chexpert import Chexpert
from tensorflow_datasets.image.cifar import Cifar10
from tensorflow_datasets.image.cifar import Cifar100
from tensorflow_datasets.image.cifar10_corrupted import Cifar10Corrupted
from tensorflow_datasets.image.clevr import CLEVR
from tensorflow_datasets.image.coco import Coco
from tensorflow_datasets.image.coco2014_legacy import Coco2014
from tensorflow_datasets.image.coil100 import Coil100
from tensorflow_datasets.image.colorectal_histology import ColorectalHistology
from tensorflow_datasets.image.colorectal_histology import ColorectalHistologyLarge
from tensorflow_datasets.image.cycle_gan import CycleGAN
from tensorflow_datasets.image.deep_weeds import DeepWeeds
from tensorflow_datasets.image.diabetic_retinopathy_detection import DiabeticRetinopathyDetection
from tensorflow_datasets.image.downsampled_imagenet import DownsampledImagenet
from tensorflow_datasets.image.dsprites import Dsprites
from tensorflow_datasets.image.dtd import Dtd
from tensorflow_datasets.image.eurosat import Eurosat
from tensorflow_datasets.image.flowers import TFFlowers
from tensorflow_datasets.image.food101 import Food101
from tensorflow_datasets.image.horses_or_humans import HorsesOrHumans
from tensorflow_datasets.image.image_folder import ImageLabelFolder
from tensorflow_datasets.image.imagenet import Imagenet2012
from tensorflow_datasets.image.imagenet2012_corrupted import Imagenet2012Corrupted
from tensorflow_datasets.image.kitti import Kitti
from tensorflow_datasets.image.lfw import LFW
from tensorflow_datasets.image.lsun import Lsun
from tensorflow_datasets.image.mnist import EMNIST
from tensorflow_datasets.image.mnist import FashionMNIST
from tensorflow_datasets.image.mnist import KMNIST
from tensorflow_datasets.image.mnist import MNIST
from tensorflow_datasets.image.mnist_corrupted import MNISTCorrupted
from tensorflow_datasets.image.omniglot import Omniglot
from tensorflow_datasets.image.open_images import OpenImagesV4
from tensorflow_datasets.image.oxford_flowers102 import OxfordFlowers102
from tensorflow_datasets.image.oxford_iiit_pet import OxfordIIITPet
from tensorflow_datasets.image.patch_camelyon import PatchCamelyon
from tensorflow_datasets.image.pet_finder import PetFinder
from tensorflow_datasets.image.quickdraw import QuickdrawBitmap
from tensorflow_datasets.image.resisc45 import Resisc45
from tensorflow_datasets.image.rock_paper_scissors import RockPaperScissors
from tensorflow_datasets.image.scene_parse_150 import SceneParse150
from tensorflow_datasets.image.shapes3d import Shapes3d
from tensorflow_datasets.image.smallnorb import Smallnorb
from tensorflow_datasets.image.so2sat import So2sat
from tensorflow_datasets.image.stanford_dogs import StanfordDogs
from tensorflow_datasets.image.stanford_online_products import StanfordOnlineProducts
from tensorflow_datasets.image.sun import Sun397
from tensorflow_datasets.image.svhn import SvhnCropped
from tensorflow_datasets.image.uc_merced import UcMerced
from tensorflow_datasets.image.visual_domain_decathlon import VisualDomainDecathlon
from tensorflow_datasets.image.voc import Voc2007

# EXERCISE: Import your dataset module here

# YOUR CODE HERE
```

## URL Checksums

If you're contributing the dataset to `tensorflow/datasets`, add a checksums file for the dataset. On first download, the DownloadManager will automatically add the sizes and checksums for all downloaded URLs to that file. This ensures that on subsequent data generation, the downloaded files are as expected.


```python
!touch tensorflow_datasets/url_checksums/imdb_faces.txt
```

## Build the Dataset


```python
# EXERCISE: Fill in the name of your dataset.
# The name must be a string.
DATASET_NAME = # YOUR CODE HERE
```

We then run the `download_and_prepare` script locally to build it, using the following command:

```
%%bash -s $DATASET_NAME
python -m tensorflow_datasets.scripts.download_and_prepare \
  --register_checksums \
  --datasets=$1
```

**NOTE:** It may take more than 30 minutes to download the dataset and then write all the preprocessed files as TFRecords. Due to the enormous size of the data involved, we are not going to run the above code here. However, if you have enough disk space, you are welcome to run it locally or in a Colab. 

## Load the Dataset

Once the dataset is built you can load it in the usual way, by using `tfds.load`, as shown below:

```python
import tensorflow_datasets as tfds
dataset, info = tfds.load('imdb_faces', with_info=True)
```

**Note:** Since we didn't build the `imdb_faces` dataset due to its size, we are unable to run the above code. However, if you had enough disk space to build the `imdb_faces` dataset, you are welcome to run it locally or in a Colab.

## Explore the Dataset

Once the dataset is loaded, you can explore it by using the following loop:

```python
for feature in tfds.as_numpy(dataset['train']):
  for key, value in feature.items():
    if key == 'image':
      value = value.shape
    print(key, value)
  break
```

**Note:** Since we didn't build the `imdb_faces` dataset due to its size, we are unable to run the above code. However, if you had enough disk space to build the `imdb_faces` dataset, you are welcome to run it locally or in a Colab.

The expected output from the code block shown above should be:

```python
>>>
celeb_id 12387
dob 722957
face_location [1.         0.56327355 1.         1.        ]
face_score 4.0612864
gender 0
image (96, 97, 3)
photo_taken 2007
second_face_score 3.6680346
```

# Next steps for publishing

**Double-check the citation**  

It's important that DatasetInfo.citation includes a good citation for the dataset. It's hard and important work contributing a dataset to the community and we want to make it easy for dataset users to cite the work.

If the dataset's website has a specifically requested citation, use that (in BibTex format).

If the paper is on arXiv, find it there and click the bibtex link on the right-hand side.

If the paper is not on arXiv, find the paper on Google Scholar and click the double-quotation mark underneath the title and on the popup, click BibTeX.

If there is no associated paper (for example, there's just a website), you can use the BibTeX Online Editor to create a custom BibTeX entry (the drop-down menu has an Online entry type).
  

**Add a test**   

Most datasets in TFDS should have a unit test and your reviewer may ask you to add one if you haven't already. See the testing section below.   
**Check your code style**  

Follow the PEP 8 Python style guide, except TensorFlow uses 2 spaces instead of 4. Please conform to the Google Python Style Guide,

Most importantly, use tensorflow_datasets/oss_scripts/lint.sh to ensure your code is properly formatted. For example, to lint the image directory
See TensorFlow code style guide for more information.

**Add release notes**
Add the dataset to the release notes. The release note will be published for the next release.

**Send for review!**
Send the pull request for review.

For more information, visit https://www.tensorflow.org/datasets/add_dataset
