# Classify Structured Data

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20Deployment/Course%203%20-%20TensorFlow%20Datasets/Week%202/Exercises/TFDS_Week2_Exercise.ipynb">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
    Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20Deployment/Course%203%20-%20TensorFlow%20Datasets/Week%202/Exercises/TFDS_Week2_Exercise.ipynb">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub</a>
  </td>
</table>

## Import TensorFlow and Other Libraries


```
try:
    %tensorflow_version 2.x
except:
    pass
```


```
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow import feature_column

from sklearn.model_selection import train_test_split

print("\u2022 Using TensorFlow Version:", tf.__version__)
```

## Use Pandas to Create a Dataframe

[Pandas](https://pandas.pydata.org/) is a Python library with many helpful utilities for loading and working with structured data. We will use Pandas to download the dataset from a URL and load it into a dataframe.


```
URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
dataframe.head()
```

## Split the Dataframe Into Train, Validation, and Test Sets

The dataset we downloaded was a single CSV file. We will split this into train, validation, and test sets.


```
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')
```

## Create an Input Pipeline Using `tf.data`

Next, we will wrap the dataframes with [tf.data](https://www.tensorflow.org/guide/datasets). This will enable us  to use feature columns as a bridge to map from the columns in the Pandas dataframe to features used to train the model. If we were working with a very large CSV file (so large that it does not fit into memory), we would use tf.data to read it from disk directly.


```
# EXERCISE: A utility method to create a tf.data dataset from a Pandas Dataframe.

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    
    # Use Pandas dataframe's pop method to get the list of targets.
    labels = # YOUR CODE HERE
    
    # Create a tf.data.Dataset from the dataframe and labels.
    ds = # YOUR CODE HERE
    
    if shuffle:
        # Shuffle dataset.
        ds = # YOUR CODE HERE
        
    # Batch dataset with specified batch_size parameter.
    ds = # YOUR CODE HERE
    
    return ds
```


```
batch_size = 5 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
```

## Understand the Input Pipeline

Now that we have created the input pipeline, let's call it to see the format of the data it returns. We have used a small batch size to keep the output readable.


```
for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['age'])
    print('A batch of targets:', label_batch )
```

We can see that the dataset returns a dictionary of column names (from the dataframe) that map to column values from rows in the dataframe.

## Create Several Types of Feature Columns

TensorFlow provides many types of feature columns. In this section, we will create several types of feature columns, and demonstrate how they transform a column from the dataframe.


```
# Try to demonstrate several types of feature columns by getting an example.
example_batch = next(iter(train_ds))[0]
```


```
# A utility method to create a feature column and to transform a batch of data.
def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column, dtype='float64')
    print(feature_layer(example_batch).numpy())
```

### Numeric Columns

The output of a feature column becomes the input to the model (using the demo function defined above, we will be able to see exactly how each column from the dataframe is transformed). A [numeric column](https://www.tensorflow.org/api_docs/python/tf/feature_column/numeric_column) is the simplest type of column. It is used to represent real valued features. 


```
# EXERCISE: Create a numeric feature column out of 'age' and demo it.
age = # YOUR CODE HERE

demo(age)
```

In the heart disease dataset, most columns from the dataframe are numeric.

### Bucketized Columns

Often, you don't want to feed a number directly into the model, but instead split its value into different categories based on numerical ranges. Consider raw data that represents a person's age. Instead of representing age as a numeric column, we could split the age into several buckets using a [bucketized column](https://www.tensorflow.org/api_docs/python/tf/feature_column/bucketized_column). 


```
# EXERCISE: Create a bucketized feature column out of 'age' with
# the following boundaries and demo it.
boundaries = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]

age_buckets = # YOUR CODE HERE 

demo(age_buckets)
```

Notice the one-hot values above describe which age range each row matches.

### Categorical Columns

In this dataset, thal is represented as a string (e.g. 'fixed', 'normal', or 'reversible'). We cannot feed strings directly to a model. Instead, we must first map them to numeric values. The categorical vocabulary columns provide a way to represent strings as a one-hot vector (much like you have seen above with age buckets). 

**Note**: You will probably see some warning messages when running some of the code cell below. These warnings have to do with software updates and should not cause any errors or prevent your code from running.


```
# EXERCISE: Create a categorical vocabulary column out of the
# above mentioned categories with the key specified as 'thal'.
thal = # YOUR CODE HERE

# EXERCISE: Create an indicator column out of the created categorical column.
thal_one_hot = # YOUR CODE HERE

demo(thal_one_hot)
```

The vocabulary can be passed as a list using [categorical_column_with_vocabulary_list](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_list), or loaded from a file using [categorical_column_with_vocabulary_file](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_file).

### Embedding Columns

Suppose instead of having just a few possible strings, we have thousands (or more) values per category. For a number of reasons, as the number of categories grow large, it becomes infeasible to train a neural network using one-hot encodings. We can use an embedding column to overcome this limitation. Instead of representing the data as a one-hot vector of many dimensions, an [embedding column](https://www.tensorflow.org/api_docs/python/tf/feature_column/embedding_column) represents that data as a lower-dimensional, dense vector in which each cell can contain any number, not just 0 or 1. You can tune the size of the embedding with the `dimension` parameter.


```
# EXERCISE: Create an embedding column out of the categorical
# vocabulary you just created (thal). Set the size of the 
# embedding to 8, by using the dimension parameter.

thal_embedding = # YOUR CODE HERE


demo(thal_embedding)
```

### Hashed Feature Columns

Another way to represent a categorical column with a large number of values is to use a [categorical_column_with_hash_bucket](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_hash_bucket). This feature column calculates a hash value of the input, then selects one of the `hash_bucket_size` buckets to encode a string. When using this column, you do not need to provide the vocabulary, and you can choose to make the number of hash buckets significantly smaller than the number of actual categories to save space.


```
# EXERCISE: Create a hashed feature column with 'thal' as the key and 
# 1000 hash buckets.
thal_hashed = # YOUR CODE HERE

demo(feature_column.indicator_column(thal_hashed))
```

### Crossed Feature Columns
Combining features into a single feature, better known as [feature crosses](https://developers.google.com/machine-learning/glossary/#feature_cross), enables a model to learn separate weights for each combination of features. Here, we will create a new feature that is the cross of age and thal. Note that `crossed_column` does not build the full table of all possible combinations (which could be very large). Instead, it is backed by a `hashed_column`, so you can choose how large the table is.


```
# EXERCISE: Create a crossed column using the bucketized column (age_buckets),
# the categorical vocabulary column (thal) previously created, and 1000 hash buckets.
crossed_feature = # YOUR CODE HERE

demo(feature_column.indicator_column(crossed_feature))
```

## Choose Which Columns to Use

We have seen how to use several types of feature columns. Now we will use them to train a model. The goal of this exercise is to show you the complete code needed to work with feature columns. We have selected a few columns to train our model below arbitrarily.

If your aim is to build an accurate model, try a larger dataset of your own, and think carefully about which features are the most meaningful to include, and how they should be represented.


```
dataframe.dtypes
```

You can use the above list of column datatypes to map the appropriate feature column to every column in the dataframe.


```
# EXERCISE: Fill in the missing code below
feature_columns = []

# Numeric Cols.
# Create a list of numeric columns. Use the following list of columns
# that have a numeric datatype: ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca'].
numeric_columns = # YOUR CODE HERE

for header in numeric_columns:
    # Create a numeric feature column  out of the header.
    numeric_feature_column = # YOUR CODE HERE
    
    feature_columns.append(numeric_feature_column)

# Bucketized Cols.
# Create a bucketized feature column out of the age column (numeric column)
# that you've already created. Use the following boundaries:
# [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
age_buckets = # YOUR CODE HERE

feature_columns.append(age_buckets)

# Indicator Cols.
# Create a categorical vocabulary column out of the categories
# ['fixed', 'normal', 'reversible'] with the key specified as 'thal'.
thal = # YOUR CODE HERE

# Create an indicator column out of the created thal categorical column
thal_one_hot = # YOUR CODE HERE

feature_columns.append(thal_one_hot)

# Embedding Cols.
# Create an embedding column out of the categorical vocabulary you
# just created (thal). Set the size of the embedding to 8, by using
# the dimension parameter.
thal_embedding = # YOUR CODE HERE

feature_columns.append(thal_embedding)

# Crossed Cols.
# Create a crossed column using the bucketized column (age_buckets),
# the categorical vocabulary column (thal) previously created, and 1000 hash buckets.
crossed_feature = # YOUR CODE HERE

# Create an indicator column out of the crossed column created above to one-hot encode it.
crossed_feature = # YOUR CODE HERE

feature_columns.append(crossed_feature)
```

### Create a Feature Layer

Now that we have defined our feature columns, we will use a [DenseFeatures](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/DenseFeatures) layer to input them to our Keras model.


```
# EXERCISE: Create a Keras DenseFeatures layer and pass the feature_columns you just created.
feature_layer = # YOUR CODE HERE
```

Earlier, we used a small batch size to demonstrate how feature columns worked. We create a new input pipeline with a larger batch size.


```
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
```

## Create, Compile, and Train the Model


```
model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=100)
```


```
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
```
