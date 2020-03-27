```
import json
import tensorflow as tf
import csv
import random
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers


embedding_dim = 100
max_length = 16
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size=#Your dataset size here. Experiment using smaller values (i.e. 16000), but don't forget to train on at least 160000 to see the best effects
test_portion=.1

corpus = []

```


```
# Note that I cleaned the Stanford dataset to remove LATIN1 encoding to make it easier for Python CSV reader
# You can do that yourself with:
# iconv -f LATIN1 -t UTF8 training.1600000.processed.noemoticon.csv -o training_cleaned.csv
# I then hosted it on my site to make it easier to use in this notebook

!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/training_cleaned.csv \
    -O /tmp/training_cleaned.csv

num_sentences = 0

with open("/tmp/training_cleaned.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
      # Your Code here. Create list items where the first item is the text, found in row[5], and the second is the label. Note that the label is a '0' or a '4' in the text. When it's the former, make
      # your label to be 0, otherwise 1. Keep a count of the number of sentences in num_sentences
        list_item=[]
        # YOUR CODE HERE
        num_sentences = num_sentences + 1
        corpus.append(list_item)

```

    --2019-06-07 17:53:35--  https://storage.googleapis.com/laurencemoroney-blog.appspot.com/training_cleaned.csv
    Resolving storage.googleapis.com (storage.googleapis.com)... 173.194.192.128, 2607:f8b0:4001:c1d::80
    Connecting to storage.googleapis.com (storage.googleapis.com)|173.194.192.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 238942690 (228M) [application/octet-stream]
    Saving to: ‘/tmp/training_cleaned.csv’
    
    /tmp/training_clean 100%[===================>] 227.87M   221MB/s    in 1.0s    
    
    2019-06-07 17:53:36 (221 MB/s) - ‘/tmp/training_cleaned.csv’ saved [238942690/238942690]
    



```
print(num_sentences)
print(len(corpus))
print(corpus[1])

# Expected Output:
# 1600000
# 1600000
# ["is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!", 0]
```

    1600000
    1600000
    ["is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!", 0]



```
sentences=[]
labels=[]
random.shuffle(corpus)
for x in range(training_size):
    sentences.append(# YOUR CODE HERE)
    labels.append(# YOUR CODE HERE)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(# YOUR CODE HERE)

word_index = tokenizer.word_index
vocab_size=len(# YOUR CODE HERE)

sequences = tokenizer.texts_to_sequences(# YOUR CODE HERE)
padded = pad_sequences(# YOUR CODE HERE)

split = int(test_portion * training_size)

test_sequences = padded[# YOUR CODE HERE]
training_sequences = padded[# YOUR CODE HERE]
test_labels = labels[# YOUR CODE HERE]
training_labels = labels[# YOUR CODE HERE]
```


```
print(vocab_size)
print(word_index['i'])
# Expected Output
# 138858
# 1
```

    138858
    1



```
# Note this is the 100 dimension version of GloVe from Stanford
# I unzipped and hosted it on my site to make this notebook easier
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt \
    -O /tmp/glove.6B.100d.txt
embeddings_index = {};
with open('/tmp/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;
```

    --2019-06-07 17:55:30--  https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt
    Resolving storage.googleapis.com (storage.googleapis.com)... 64.233.183.128, 2607:f8b0:4001:c12::80
    Connecting to storage.googleapis.com (storage.googleapis.com)|64.233.183.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 347116733 (331M) [text/plain]
    Saving to: ‘/tmp/glove.6B.100d.txt’
    
    /tmp/glove.6B.100d. 100%[===================>] 331.04M   160MB/s    in 2.1s    
    
    2019-06-07 17:55:33 (160 MB/s) - ‘/tmp/glove.6B.100d.txt’ saved [347116733/347116733]
    



```
print(len(embeddings_matrix))
# Expected Output
# 138859
```

    138859



```
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),
    # YOUR CODE HERE - experiment with combining different types, such as convolutions and LSTMs
])
model.compile(# YOUR CODE HERE)
model.summary()

num_epochs = 50
history = model.fit(training_sequences, training_labels, epochs=num_epochs, validation_data=(test_sequences, test_labels), verbose=2)

print("Training Complete")

```


```
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])

plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])

plt.figure()


# Expected Output
# A chart where the validation loss does not increase sharply!
```
