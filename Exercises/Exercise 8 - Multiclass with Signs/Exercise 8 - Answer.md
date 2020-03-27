```
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import files
```

The data for this exercise is available at: https://www.kaggle.com/datamunge/sign-language-mnist/home

Sign up and download to find 2 CSV files: sign_mnist_test.csv and sign_mnist_train.csv -- You will upload both of them using this button before you can continue.



```
uploaded=files.upload()
```



<input type="file" id="files-9979a0a3-4162-45c4-98fc-4e133ab5ad52" name="files[]" multiple disabled />
<output id="result-9979a0a3-4162-45c4-98fc-4e133ab5ad52">
 Upload widget is only available when the cell has been executed in the
 current browser session. Please rerun this cell to enable.
 </output>
 <script src="/nbextensions/google.colab/files.js"></script> 


    Saving sign_mnist_test.csv to sign_mnist_test.csv
    Saving sign_mnist_train.csv to sign_mnist_train.csv



```
def get_data(filename):
    with open(filename) as training_file:
        csv_reader = csv.reader(training_file, delimiter=',')
        first_line = True
        temp_images = []
        temp_labels = []
        for row in csv_reader:
            if first_line:
                # print("Ignoring first line")
                first_line = False
            else:
                temp_labels.append(row[0])
                image_data = row[1:785]
                image_data_as_array = np.array_split(image_data, 28)
                temp_images.append(image_data_as_array)
        images = np.array(temp_images).astype('float')
        labels = np.array(temp_labels).astype('float')
    return images, labels


training_images, training_labels = get_data('sign_mnist_train.csv')
testing_images, testing_labels = get_data('sign_mnist_test.csv')

print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

```

    (27455, 28, 28)
    (27455,)
    (7172, 28, 28)
    (7172,)



```
training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    rescale=1. / 255)

print(training_images.shape)
print(testing_images.shape)
```

    (27455, 28, 28, 1)
    (7172, 28, 28, 1)



```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(26, activation=tf.nn.softmax)])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_datagen.flow(training_images, training_labels, batch_size=32),
                              steps_per_epoch=len(training_images) / 32,
                              epochs=15,
                              validation_data=validation_datagen.flow(testing_images, testing_labels, batch_size=32),
                              validation_steps=len(testing_images) / 32)

model.evaluate(testing_images, testing_labels)

```

    Epoch 1/15
    857/857 [==============================] - 21s 24ms/step - loss: 2.8540 - acc: 0.1562 - val_loss: 1.9644 - val_acc: 0.3100
    Epoch 2/15
    857/857 [==============================] - 46s 54ms/step - loss: 2.1131 - acc: 0.2188 - val_loss: 1.5558 - val_acc: 0.4855
    Epoch 3/15
    857/857 [==============================] - 40s 47ms/step - loss: 1.7423 - acc: 0.3438 - val_loss: 1.1643 - val_acc: 0.6115
    Epoch 4/15
    857/857 [==============================] - 30s 35ms/step - loss: 1.4933 - acc: 0.4688 - val_loss: 1.1572 - val_acc: 0.6064
    Epoch 5/15
    857/857 [==============================] - 22s 26ms/step - loss: 1.3521 - acc: 0.7500 - val_loss: 0.8973 - val_acc: 0.7167
    Epoch 6/15
    857/857 [==============================] - 21s 24ms/step - loss: 1.2332 - acc: 0.5625 - val_loss: 0.8082 - val_acc: 0.7200
    Epoch 7/15
    857/857 [==============================] - 44s 51ms/step - loss: 1.1631 - acc: 0.5938 - val_loss: 0.8671 - val_acc: 0.7352
    Epoch 8/15
    857/857 [==============================] - 58s 67ms/step - loss: 1.0857 - acc: 0.7188 - val_loss: 0.7608 - val_acc: 0.7949
    Epoch 9/15
    857/857 [==============================] - 23s 26ms/step - loss: 1.0197 - acc: 0.6562 - val_loss: 0.6978 - val_acc: 0.7674
    Epoch 10/15
    857/857 [==============================] - 28s 33ms/step - loss: 0.9711 - acc: 0.6875 - val_loss: 0.7027 - val_acc: 0.7984
    Epoch 11/15
    857/857 [==============================] - 53s 61ms/step - loss: 0.9076 - acc: 0.5625 - val_loss: 0.5784 - val_acc: 0.8238
    Epoch 12/15
    857/857 [==============================] - 64s 75ms/step - loss: 0.8764 - acc: 0.5938 - val_loss: 0.6079 - val_acc: 0.8133
    Epoch 13/15
    857/857 [==============================] - 28s 33ms/step - loss: 0.8410 - acc: 0.7500 - val_loss: 0.4547 - val_acc: 0.8182
    Epoch 14/15
    857/857 [==============================] - 22s 26ms/step - loss: 0.8045 - acc: 0.5312 - val_loss: 0.2415 - val_acc: 0.8496
    Epoch 15/15
    857/857 [==============================] - 47s 55ms/step - loss: 0.7836 - acc: 0.7188 - val_loss: 0.3857 - val_acc: 0.8425
    7172/7172 [==============================] - 4s 596us/step - loss: 6.9243 - acc: 0.5661





    [6.92426086682151, 0.56609035]




```
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```


![png](Exercise%208%20-%20Answer_files/Exercise%208%20-%20Answer_6_0.png)



![png](Exercise%208%20-%20Answer_files/Exercise%208%20-%20Answer_6_1.png)

