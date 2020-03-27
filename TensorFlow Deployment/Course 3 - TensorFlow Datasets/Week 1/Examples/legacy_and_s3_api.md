# Legacy and S3 APIs

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20Deployment/Course%203%20-%20TensorFlow%20Datasets/Week%201/Examples/legacy_and_s3_api.ipynb">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
    Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20Deployment/Course%203%20-%20TensorFlow%20Datasets/Week%201/Examples/legacy_and_s3_api.ipynb">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub</a>
  </td>
</table>

In this notebook, we'll take a look at the Legacy and S3 APIs for TensorFlow datasets. We'll explore both the Legacy API which is object-based and the new API which was nicknamed S3 for Split, Slices, and Strings. The new S3 API will eventually become the default API for TensorFlow Datasets.

## Setup

We'll start by importing TensorFlow and TensorFlow Datasets.


```
try:
    %tensorflow_version 2.x
except:
    pass
```


```
import tensorflow as tf
import tensorflow_datasets as tfds

print("\u2022 Using TensorFlow Version:", tf.__version__)
```

## Merging Splits with the Legacy API

In the Legacy API, you can merge splits together by adding them together, as shown below:


```
all_splits = tfds.Split.TRAIN + tfds.Split.TEST

ds = tfds.load("mnist", split=all_splits)

print("Number of Records: {:,}".format(len(list(ds))))
```

## Subsplitting with the Legacy API

With the Legacy API, we can use the `subsplit` method to divide the datasets. In the example below, we divide the training set into four splits by specifying the number of subsplits, with the argument `k=4`.


```
s1, s2, s3, s4 = tfds.Split.TRAIN.subsplit(k=4)

dataset_split_1 = tfds.load("mnist", split=s1)
dataset_split_2 = tfds.load("mnist", split=s2)
dataset_split_3 = tfds.load("mnist", split=s3)
dataset_split_4 = tfds.load("mnist", split=s4)

print(len(list(dataset_split_1)))
print(len(list(dataset_split_2)))
print(len(list(dataset_split_3)))
print(len(list(dataset_split_4)))
```

We can also perform the same operation, by specifying a percentage slice in the `subsplit` method instead. In the example below, we divide the training set into four splits by specifying a percentage slice, with `tfds.percent`.


```
s1 = tfds.Split.TRAIN.subsplit(tfds.percent[0:25])
s2 = tfds.Split.TRAIN.subsplit(tfds.percent[25:50])
s3 = tfds.Split.TRAIN.subsplit(tfds.percent[50:75])
s4 = tfds.Split.TRAIN.subsplit(tfds.percent[75:100])

dataset_split_1 = tfds.load("mnist", split=s1)
dataset_split_2 = tfds.load("mnist", split=s2)
dataset_split_3 = tfds.load("mnist", split=s3)
dataset_split_4 = tfds.load("mnist", split=s4)

print(len(list(dataset_split_1)))
print(len(list(dataset_split_2)))
print(len(list(dataset_split_3)))
print(len(list(dataset_split_4)))
```

## Using the New S3 API

Before using the new S3 API, we must first find out whether the MNIST dataset implements the new S3 API. In the cell below we indicate that we want to use version `3.*.*` of the MNIST dataset.


```
mnist_builder = tfds.builder("mnist:3.*.*")

print(mnist_builder.version.implements(tfds.core.Experiment.S3))
```

We can see that the code above printed `True`, which means that version `3.*.*` of the MNIST dataset supports the new S3 API.

Now, let's see how we can use the S3 API to download the MNIST dataset and specify the splits we want use. In the code below we download the `train` and `test` splits of the MNIST dataset and then we print their size. We will see that there are 60,000 records in the training set and 10,000 in the test set.


```
train_ds, test_ds = tfds.load('mnist:3.*.*', split=['train', 'test'])

print(len(list(train_ds)))
print(len(list(test_ds)))
```

In the S3 API we can use strings to specify the slicing instructions. For example, in the cell below we will merge the training and test sets by passing the string `’train+test'` to the `split` argument.


```
combined = tfds.load('mnist:3.*.*', split='train+test')

print(len(list(combined)))
```

We can also use Python style list slicers to specify the data we want. For example, we can specify that we want to take the first 10,000 records of the `train` split with the string `'train[:10000]'`, as shown below:


```
first10k = tfds.load('mnist:3.*.*', split='train[:10000]')

print(len(list(first10k)))
```

The S3 API, also allows us to specify the percentage of the data we want to use. For example, we can select the first 20\% of the training set with the string `'train[:20%]'`, as shown below:


```
first20p = tfds.load('mnist:3.*.*', split='train[:20%]')

print(len(list(first20p)))
```

We can see that `first20p` contains 12,000 records, which is indeed 20\% the total number of records in the training set. Recall that the training set contains 60,000 records. 

Because the slices are string-based we can use loops, like the ones shown below, to slice up the dataset and make some pretty complex splits. For example, the loops below create 10 complimentary validation and training sets (each loop returns a list with 5 data sets).


```
val_ds = tfds.load('mnist:3.*.*', split=['train[{}%:{}%]'.format(k, k+20) for k in range(0, 100, 20)])

train_ds = tfds.load('mnist:3.*.*', split=['train[:{}%]+train[{}%:]'.format(k, k+20) for k in range(0, 100, 20)])
```


```
val_ds
```


```
train_ds
```


```
print(len(list(val_ds)))
print(len(list(train_ds)))
```

The S3 API also allows us to compose new datasets by using pieces from different splits. For example, we can create a new dataset from the first 10\% of the test set and the last 80\% of the training set, as shown below.


```
composed_ds = tfds.load('mnist:3.*.*', split='test[:10%]+train[-80%:]')

print(len(list(composed_ds)))
```
