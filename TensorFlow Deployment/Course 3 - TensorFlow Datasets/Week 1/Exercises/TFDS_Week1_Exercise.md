# TFDS and Rock, Paper, Scissors

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20Deployment/Course%203%20-%20TensorFlow%20Datasets/Week%201/Exercises/TFDS_Week1_Exercise.ipynb">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
    Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20Deployment/Course%203%20-%20TensorFlow%20Datasets/Week%201/Exercises/TFDS_Week1_Exercise.ipynb">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub</a>
  </td>
</table>

In this week's exercise you will be working with TFDS and the rock-paper-scissors dataset. You'll do a few tasks such as exploring the info of the dataset in order to figure out the name of the splits. You'll also write code to see if the dataset supports the new S3 API before creating your own versions of the dataset.

## Setup


```
try:
    %tensorflow_version 2.x
except:
    pass
```


```
# Use all imports
import tensorflow as tf
import tensorflow_datasets as tfds

print("\u2022 Using TensorFlow Version:", tf.__version__)
```

## Extract the Rock, Paper, Scissors Dataset

In the cell below, you will extract the `rock_paper_scissors` dataset and then print its info. Take note of the splits, what they're called, and their size.


```
# EXERCISE: Use tfds.load to extract the rock_paper_scissors dataset.

data, info = # YOUR CODE HERE
print(info)
```


```
# EXERCISE: In the space below, write code that iterates through the splits
# without hardcoding any keys. The code should extract 'test' and 'train' as
# the keys, and then print out the number of items in the dataset for each key. 
# HINT: num_examples property is very useful here.

for # YOUR CODE HERE:
  print(# YOUR CODE HERE)

# EXPECTED OUTPUT
# test:372
# train:2520
```

## Use the New S3 API

Before using the new S3 API, you must first find out whether the `rock_paper_scissors` dataset implements the new S3 API. In the cell below you should use version `3.*.*` of the `rock_paper_scissors` dataset.


```
# EXERCISE: In the space below, use the tfds.builder to fetch the
# rock_paper_scissors dataset and check to see if it supports the
# new S3 API. 
# HINT: The builder should 'implement' something

rps_builder = # YOUR CODE HERE (Include the following arguments in your code: "rock_paper_scissors:3.*.*")

print(# YOUR CODE HERE)

# Expected output:
# True
```

## Create New Datasets with the S3 API

Sometimes datasets are too big for prototyping. In the cell below, you will create a smaller dataset, where instead of using all of the training data and all of the test data, you instead have a `small_train` and `small_test` each of which are comprised of the first 10% of the records in their respective datasets.


```
# EXERCISE: In the space below, create two small datasets, `small_train`
# and `small_test`, each of which are comprised of the first 10% of the
# records in their respective datasets.

small_train = # YOUR CODE HERE (Include the following arguments in your code: "rock_paper_scissors:3.*.*")
small_test = # YOUR CODE HERE (Include the following arguments in your code: "rock_paper_scissors:3.*.*")

# No expected output yet, that's in the next cell
```


```
# EXERCISE: Print out the size (length) of the small versions of the datasets.

print(# YOUR CODE HERE)
print(# YOUR CODE HERE)

# Expected output
# 252
# 37
```

The original dataset doesn't have a validation set, just training and testing sets. In the cell below, you will use TFDS to create new datasets according to these rules:

* `new_train`: The new training set should be the first 90% of the original training set.


* `new_test`: The new test set should be the first 90% of the original test set.


* `validation`: The new validation set should be the last 10% of the original training set + the last 10% of the original test set.


```
# EXERCISE: In the space below, create 3 new datasets according to
# the rules indicated above.

new_train = # YOUR CODE HERE (Include the following arguments in your code: "rock_paper_scissors:3.*.*")
print(# YOUR CODE HERE)

new_test = # YOUR CODE HERE (Include the following arguments in your code: "rock_paper_scissors:3.*.*")
print(# YOUR CODE HERE)

validation = # YOUR CODE HERE (Include the following arguments in your code: "rock_paper_scissors:3.*.*")
print(# YOUR CODE HERE)

# Expected output
# 2268
# 335
# 289
```
