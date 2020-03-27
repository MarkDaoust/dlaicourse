```
# Import all the necessary files!
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
```


```
# Download the inception v3 weights
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

# Import the inception model  
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
  layer.trainable = False
  
# Print the model summary
pre_trained_model.summary()

# Expected Output is extremely large, but should end with:

#batch_normalization_v1_281 (Bat (None, 3, 3, 192)    576         conv2d_281[0][0]                 
#__________________________________________________________________________________________________
#activation_273 (Activation)     (None, 3, 3, 320)    0           batch_normalization_v1_273[0][0] 
#__________________________________________________________________________________________________
#mixed9_1 (Concatenate)          (None, 3, 3, 768)    0           activation_275[0][0]             
#                                                                 activation_276[0][0]             
#__________________________________________________________________________________________________
#concatenate_5 (Concatenate)     (None, 3, 3, 768)    0           activation_279[0][0]             
#                                                                 activation_280[0][0]             
#__________________________________________________________________________________________________
#activation_281 (Activation)     (None, 3, 3, 192)    0           batch_normalization_v1_281[0][0] 
#__________________________________________________________________________________________________
#mixed10 (Concatenate)           (None, 3, 3, 2048)   0           activation_273[0][0]             
#                                                                 mixed9_1[0][0]                   
#                                                                 concatenate_5[0][0]              
#                                                                 activation_281[0][0]             
#==================================================================================================
#Total params: 21,802,784
#Trainable params: 0
#Non-trainable params: 21,802,784
```


```
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Expected Output:
# ('last layer output shape: ', (None, 7, 7, 768))
```

    ('last layer output shape: ', (None, 7, 7, 768))



```
# Define a Callback class that stops training once accuracy reaches 99.9%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.999):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True

      
```


```
from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)           

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

model.summary()

# Expected output will be large. Last few lines should be:

# mixed7 (Concatenate)            (None, 7, 7, 768)    0           activation_248[0][0]             
#                                                                  activation_251[0][0]             
#                                                                  activation_256[0][0]             
#                                                                  activation_257[0][0]             
# __________________________________________________________________________________________________
# flatten_4 (Flatten)             (None, 37632)        0           mixed7[0][0]                     
# __________________________________________________________________________________________________
# dense_8 (Dense)                 (None, 1024)         38536192    flatten_4[0][0]                  
# __________________________________________________________________________________________________
# dropout_4 (Dropout)             (None, 1024)         0           dense_8[0][0]                    
# __________________________________________________________________________________________________
# dense_9 (Dense)                 (None, 1)            1025        dropout_4[0][0]                  
# ==================================================================================================
# Total params: 47,512,481
# Trainable params: 38,537,217
# Non-trainable params: 8,975,264

```


```
# Get the Horse or Human dataset
!wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip -O /tmp/horse-or-human.zip

# Get the Horse or Human Validation dataset
!wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip -O /tmp/validation-horse-or-human.zip 
  
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import zipfile

local_zip = '//tmp/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/training')
zip_ref.close()

local_zip = '//tmp/validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/validation')
zip_ref.close()
```

    --2019-03-26 15:58:22--  https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip
    Resolving storage.googleapis.com (storage.googleapis.com)... 74.125.141.128, 2607:f8b0:400c:c06::80
    Connecting to storage.googleapis.com (storage.googleapis.com)|74.125.141.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 149574867 (143M) [application/zip]
    Saving to: ‘/tmp/horse-or-human.zip’
    
    /tmp/horse-or-human 100%[===================>] 142.65M   153MB/s    in 0.9s    
    
    2019-03-26 15:58:23 (153 MB/s) - ‘/tmp/horse-or-human.zip’ saved [149574867/149574867]
    
    --2019-03-26 15:58:24--  https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip
    Resolving storage.googleapis.com (storage.googleapis.com)... 74.125.141.128, 2607:f8b0:400c:c06::80
    Connecting to storage.googleapis.com (storage.googleapis.com)|74.125.141.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 11480187 (11M) [application/zip]
    Saving to: ‘/tmp/validation-horse-or-human.zip’
    
    /tmp/validation-hor 100%[===================>]  10.95M  55.6MB/s    in 0.2s    
    
    2019-03-26 15:58:24 (55.6 MB/s) - ‘/tmp/validation-horse-or-human.zip’ saved [11480187/11480187]
    



```
train_horses_dir = os.path.join(train_dir, 'horses') # Directory with our training horse pictures
train_humans_dir = os.path.join(train_dir, 'humans') # Directory with our training humans pictures
validation_horses_dir = os.path.join(validation_dir, 'horses') # Directory with our validation horse pictures
validation_humans_dir = os.path.join(validation_dir, 'humans')# Directory with our validation humanas pictures

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)

print(len(train_horses_fnames))
print(len(train_humans_fnames))
print(len(validation_horses_fnames))
print(len(validation_humans_fnames))

# Expected Output:
# 500
# 527
# 128
# 128
```

    500
    527
    128
    128



```
# Define our example directories and files
train_dir = '/tmp/training'
validation_dir = '/tmp/validation'

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 20,
                                                    class_mode = 'binary', 
                                                    target_size = (150, 150))     

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory( validation_dir,
                                                          batch_size  = 20,
                                                          class_mode  = 'binary', 
                                                          target_size = (150, 150))

# Expected Output:
# Found 1027 images belonging to 2 classes.
# Found 256 images belonging to 2 classes.
```

    Found 1027 images belonging to 2 classes.
    Found 256 images belonging to 2 classes.



```
# Run this and see how many epochs it should take before the callback
# fires, and stops training at 99.9% accuracy
# (It should take less than 100 epochs)
callbacks = myCallback()
history = model.fit(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 100,
            epochs = 100,
            validation_steps = 50,
            verbose = 2,
            callbacks=[callbacks])
```

    Epoch 1/100
    13/13 [==============================] - 3s 262ms/step - loss: 0.0030 - acc: 1.0000
     - 17s - loss: 0.2578 - acc: 0.8987 - val_loss: 0.0030 - val_acc: 1.0000
    Epoch 2/100
    13/13 [==============================] - 2s 144ms/step - loss: 0.0015 - acc: 1.0000
     - 14s - loss: 0.1260 - acc: 0.9591 - val_loss: 0.0015 - val_acc: 1.0000
    Epoch 3/100
    13/13 [==============================] - 2s 146ms/step - loss: 0.0238 - acc: 0.9883
     - 15s - loss: 0.0757 - acc: 0.9640 - val_loss: 0.0238 - val_acc: 0.9883
    Epoch 4/100
    13/13 [==============================] - 2s 147ms/step - loss: 2.9810e-04 - acc: 1.0000
     - 15s - loss: 0.0805 - acc: 0.9737 - val_loss: 2.9810e-04 - val_acc: 1.0000
    Epoch 5/100
    13/13 [==============================] - 2s 144ms/step - loss: 0.0057 - acc: 0.9961
     - 15s - loss: 0.0448 - acc: 0.9864 - val_loss: 0.0057 - val_acc: 0.9961
    Epoch 6/100
    13/13 [==============================] - 2s 142ms/step - loss: 0.1663 - acc: 0.9688
     - 15s - loss: 0.0480 - acc: 0.9844 - val_loss: 0.1663 - val_acc: 0.9688
    Epoch 7/100
    13/13 [==============================] - 2s 143ms/step - loss: 0.0099 - acc: 0.9961
     - 15s - loss: 0.0460 - acc: 0.9844 - val_loss: 0.0099 - val_acc: 0.9961
    Epoch 8/100
    13/13 [==============================] - 2s 144ms/step - loss: 0.0609 - acc: 0.9883
     - 15s - loss: 0.0327 - acc: 0.9912 - val_loss: 0.0609 - val_acc: 0.9883
    Epoch 9/100
    13/13 [==============================] - 2s 147ms/step - loss: 0.0260 - acc: 0.9922
     - 15s - loss: 0.0434 - acc: 0.9825 - val_loss: 0.0260 - val_acc: 0.9922
    Epoch 10/100
    13/13 [==============================] - 2s 146ms/step - loss: 0.0502 - acc: 0.9883
     - 15s - loss: 0.0237 - acc: 0.9932 - val_loss: 0.0502 - val_acc: 0.9883
    Epoch 11/100
    13/13 [==============================] - 2s 146ms/step - loss: 0.0959 - acc: 0.9883
     - 15s - loss: 0.0587 - acc: 0.9796 - val_loss: 0.0959 - val_acc: 0.9883
    Epoch 12/100
    13/13 [==============================] - 2s 147ms/step - loss: 0.1125 - acc: 0.9883
     - 15s - loss: 0.0211 - acc: 0.9912 - val_loss: 0.1125 - val_acc: 0.9883
    Epoch 13/100
    13/13 [==============================] - 2s 143ms/step - loss: 0.2694 - acc: 0.9570
     - 15s - loss: 0.0463 - acc: 0.9854 - val_loss: 0.2694 - val_acc: 0.9570
    Epoch 14/100
    13/13 [==============================] - 2s 150ms/step - loss: 0.4562 - acc: 0.9531
     - 15s - loss: 0.0365 - acc: 0.9864 - val_loss: 0.4562 - val_acc: 0.9531
    Epoch 15/100
    13/13 [==============================] - 2s 147ms/step - loss: 0.1317 - acc: 0.9805
     - 15s - loss: 0.0220 - acc: 0.9893 - val_loss: 0.1317 - val_acc: 0.9805
    Epoch 16/100
    13/13 [==============================] - 2s 145ms/step - loss: 0.1389 - acc: 0.9844
     - 15s - loss: 0.0317 - acc: 0.9903 - val_loss: 0.1389 - val_acc: 0.9844
    Epoch 17/100
    13/13 [==============================] - 2s 146ms/step - loss: 0.0720 - acc: 0.9844
     - 15s - loss: 0.0545 - acc: 0.9854 - val_loss: 0.0720 - val_acc: 0.9844
    Epoch 18/100
    13/13 [==============================] - 2s 147ms/step - loss: 0.0197 - acc: 0.9922
     - 15s - loss: 0.0156 - acc: 0.9951 - val_loss: 0.0197 - val_acc: 0.9922
    Epoch 19/100
    13/13 [==============================] - 2s 146ms/step - loss: 0.0529 - acc: 0.9922
     - 15s - loss: 0.0138 - acc: 0.9951 - val_loss: 0.0529 - val_acc: 0.9922
    Epoch 20/100
    13/13 [==============================] - 2s 151ms/step - loss: 0.0437 - acc: 0.9883
     - 15s - loss: 0.0092 - acc: 0.9971 - val_loss: 0.0437 - val_acc: 0.9883
    Epoch 21/100
    13/13 [==============================] - 2s 146ms/step - loss: 0.4589 - acc: 0.9414
     - 15s - loss: 0.0281 - acc: 0.9912 - val_loss: 0.4589 - val_acc: 0.9414
    Epoch 22/100
    13/13 [==============================] - 2s 147ms/step - loss: 0.2179 - acc: 0.9727
     - 15s - loss: 0.0200 - acc: 0.9893 - val_loss: 0.2179 - val_acc: 0.9727
    Epoch 23/100
    13/13 [==============================] - 2s 145ms/step - loss: 0.1490 - acc: 0.9844
     - 15s - loss: 0.0489 - acc: 0.9903 - val_loss: 0.1490 - val_acc: 0.9844
    Epoch 24/100
    13/13 [==============================] - 2s 141ms/step - loss: 0.2060 - acc: 0.9766
     - 15s - loss: 0.0181 - acc: 0.9932 - val_loss: 0.2060 - val_acc: 0.9766
    Epoch 25/100
    13/13 [==============================] - 2s 144ms/step - loss: 0.1785 - acc: 0.9805
     - 15s - loss: 0.0137 - acc: 0.9951 - val_loss: 0.1785 - val_acc: 0.9805
    Epoch 26/100
    13/13 [==============================] - 2s 144ms/step - loss: 0.0258 - acc: 0.9922
     - 15s - loss: 0.0175 - acc: 0.9951 - val_loss: 0.0258 - val_acc: 0.9922
    Epoch 27/100
    13/13 [==============================] - 2s 146ms/step - loss: 0.0623 - acc: 0.9883
     - 15s - loss: 0.0279 - acc: 0.9942 - val_loss: 0.0623 - val_acc: 0.9883
    Epoch 28/100
    13/13 [==============================] - 2s 146ms/step - loss: 0.1110 - acc: 0.9844
     - 15s - loss: 0.0200 - acc: 0.9942 - val_loss: 0.1110 - val_acc: 0.9844
    Epoch 29/100
    13/13 [==============================] - 2s 145ms/step - loss: 0.0566 - acc: 0.9883
     - 15s - loss: 0.0039 - acc: 0.9971 - val_loss: 0.0566 - val_acc: 0.9883
    Epoch 30/100
    13/13 [==============================] - 2s 146ms/step - loss: 0.0114 - acc: 0.9961
     - 15s - loss: 0.0343 - acc: 0.9903 - val_loss: 0.0114 - val_acc: 0.9961
    Epoch 31/100
    13/13 [==============================] - 2s 145ms/step - loss: 0.3286 - acc: 0.9648
     - 15s - loss: 0.0209 - acc: 0.9922 - val_loss: 0.3286 - val_acc: 0.9648
    Epoch 32/100
    13/13 [==============================] - 2s 146ms/step - loss: 0.0234 - acc: 0.9883
     - 15s - loss: 0.0195 - acc: 0.9922 - val_loss: 0.0234 - val_acc: 0.9883
    Epoch 33/100
    13/13 [==============================] - 2s 148ms/step - loss: 0.1478 - acc: 0.9844
     - 15s - loss: 0.0251 - acc: 0.9922 - val_loss: 0.1478 - val_acc: 0.9844
    Epoch 34/100
    13/13 [==============================] - 2s 141ms/step - loss: 0.1752 - acc: 0.9766
     - 15s - loss: 0.0262 - acc: 0.9922 - val_loss: 0.1752 - val_acc: 0.9766
    Epoch 35/100
    13/13 [==============================] - 2s 150ms/step - loss: 0.3704 - acc: 0.9648
     - 15s - loss: 0.0181 - acc: 0.9922 - val_loss: 0.3704 - val_acc: 0.9648
    Epoch 36/100
    13/13 [==============================] - 2s 149ms/step - loss: 0.0575 - acc: 0.9883
     - 15s - loss: 0.0038 - acc: 0.9981 - val_loss: 0.0575 - val_acc: 0.9883
    Epoch 37/100
    13/13 [==============================] - 2s 153ms/step - loss: 0.3265 - acc: 0.9727
     - 15s - loss: 0.0094 - acc: 0.9971 - val_loss: 0.3265 - val_acc: 0.9727
    Epoch 38/100
    13/13 [==============================] - 2s 159ms/step - loss: 0.4095 - acc: 0.9570
     - 15s - loss: 0.0215 - acc: 0.9951 - val_loss: 0.4095 - val_acc: 0.9570
    Epoch 39/100
    13/13 [==============================] - 2s 159ms/step - loss: 0.2819 - acc: 0.9727
     - 15s - loss: 0.0218 - acc: 0.9951 - val_loss: 0.2819 - val_acc: 0.9727
    Epoch 40/100
    13/13 [==============================] - 2s 164ms/step - loss: 0.2788 - acc: 0.9727
     - 15s - loss: 0.0102 - acc: 0.9971 - val_loss: 0.2788 - val_acc: 0.9727
    Epoch 41/100
    13/13 [==============================] - 2s 149ms/step - loss: 0.4275 - acc: 0.9609
     - 15s - loss: 0.0080 - acc: 0.9951 - val_loss: 0.4275 - val_acc: 0.9609
    Epoch 42/100
    13/13 [==============================] - 2s 146ms/step - loss: 0.3639 - acc: 0.9609
     - 15s - loss: 0.0344 - acc: 0.9922 - val_loss: 0.3639 - val_acc: 0.9609
    Epoch 43/100
    13/13 [==============================] - 2s 149ms/step - loss: 0.3545 - acc: 0.9609
     - 15s - loss: 0.0116 - acc: 0.9961 - val_loss: 0.3545 - val_acc: 0.9609
    Epoch 44/100
    13/13 [==============================] - 2s 148ms/step - loss: 0.4563 - acc: 0.9531
     - 15s - loss: 0.0066 - acc: 0.9971 - val_loss: 0.4563 - val_acc: 0.9531
    Epoch 45/100
    13/13 [==============================] - 2s 148ms/step - loss: 0.2358 - acc: 0.9766
     - 15s - loss: 0.0122 - acc: 0.9942 - val_loss: 0.2358 - val_acc: 0.9766
    Epoch 46/100
    13/13 [==============================] - 2s 151ms/step - loss: 0.2340 - acc: 0.9688
     - 15s - loss: 0.0320 - acc: 0.9951 - val_loss: 0.2340 - val_acc: 0.9688
    Epoch 47/100
    13/13 [==============================] - 2s 149ms/step - loss: 0.3101 - acc: 0.9648
     - 15s - loss: 0.0174 - acc: 0.9942 - val_loss: 0.3101 - val_acc: 0.9648
    Epoch 48/100
    13/13 [==============================] - 2s 148ms/step - loss: 0.3406 - acc: 0.9609
     - 15s - loss: 0.0100 - acc: 0.9971 - val_loss: 0.3406 - val_acc: 0.9609
    Epoch 49/100
    13/13 [==============================] - 2s 150ms/step - loss: 0.3289 - acc: 0.9609
     - 15s - loss: 0.0151 - acc: 0.9961 - val_loss: 0.3289 - val_acc: 0.9609
    Epoch 50/100
    13/13 [==============================] - 2s 148ms/step - loss: 0.3019 - acc: 0.9609
     - 15s - loss: 0.0058 - acc: 0.9981 - val_loss: 0.3019 - val_acc: 0.9609
    Epoch 51/100
    13/13 [==============================] - 2s 146ms/step - loss: 0.6211 - acc: 0.9492
     - 15s - loss: 0.0141 - acc: 0.9971 - val_loss: 0.6211 - val_acc: 0.9492
    Epoch 52/100
    13/13 [==============================] - 2s 151ms/step - loss: 0.2269 - acc: 0.9766
     - 15s - loss: 0.0432 - acc: 0.9912 - val_loss: 0.2269 - val_acc: 0.9766
    Epoch 53/100
    13/13 [==============================] - 2s 149ms/step - loss: 0.2337 - acc: 0.9648
     - 15s - loss: 0.0216 - acc: 0.9942 - val_loss: 0.2337 - val_acc: 0.9648
    Epoch 54/100
    13/13 [==============================] - 2s 150ms/step - loss: 0.4217 - acc: 0.9531
     - 15s - loss: 0.0060 - acc: 0.9981 - val_loss: 0.4217 - val_acc: 0.9531
    Epoch 55/100
    13/13 [==============================] - 2s 146ms/step - loss: 0.5074 - acc: 0.9531
     - 15s - loss: 0.0146 - acc: 0.9922 - val_loss: 0.5074 - val_acc: 0.9531
    Epoch 56/100
    13/13 [==============================] - 2s 144ms/step - loss: 0.5438 - acc: 0.9531
     - 15s - loss: 0.0072 - acc: 0.9981 - val_loss: 0.5438 - val_acc: 0.9531
    Epoch 57/100
    13/13 [==============================] - 2s 148ms/step - loss: 0.3292 - acc: 0.9609
     - 15s - loss: 0.0331 - acc: 0.9922 - val_loss: 0.3292 - val_acc: 0.9609
    Epoch 58/100
    13/13 [==============================] - 2s 149ms/step - loss: 0.7246 - acc: 0.9453
     - 15s - loss: 0.0188 - acc: 0.9961 - val_loss: 0.7246 - val_acc: 0.9453
    Epoch 59/100
    13/13 [==============================] - 2s 149ms/step - loss: 0.7312 - acc: 0.9453
     - 15s - loss: 0.0355 - acc: 0.9922 - val_loss: 0.7312 - val_acc: 0.9453
    Epoch 60/100
    13/13 [==============================] - 2s 146ms/step - loss: 0.4679 - acc: 0.9492
     - 15s - loss: 0.0325 - acc: 0.9883 - val_loss: 0.4679 - val_acc: 0.9492
    Epoch 61/100
    13/13 [==============================] - 2s 148ms/step - loss: 0.7518 - acc: 0.9453
     - 15s - loss: 0.0203 - acc: 0.9951 - val_loss: 0.7518 - val_acc: 0.9453
    Epoch 62/100
    13/13 [==============================] - 2s 147ms/step - loss: 0.5631 - acc: 0.9492
     - 15s - loss: 0.0086 - acc: 0.9971 - val_loss: 0.5631 - val_acc: 0.9492
    Epoch 63/100
    13/13 [==============================] - 2s 146ms/step - loss: 0.4888 - acc: 0.9492
     - 15s - loss: 0.0148 - acc: 0.9981 - val_loss: 0.4888 - val_acc: 0.9492
    Epoch 64/100
    13/13 [==============================] - 2s 157ms/step - loss: 0.5083 - acc: 0.9492
     - 15s - loss: 0.0093 - acc: 0.9961 - val_loss: 0.5083 - val_acc: 0.9492
    Epoch 65/100
    13/13 [==============================] - 2s 161ms/step - loss: 0.3420 - acc: 0.9609
     - 15s - loss: 0.0407 - acc: 0.9932 - val_loss: 0.3420 - val_acc: 0.9609
    Epoch 66/100
    13/13 [==============================] - 2s 156ms/step - loss: 0.4816 - acc: 0.9492
     - 15s - loss: 0.0147 - acc: 0.9961 - val_loss: 0.4816 - val_acc: 0.9492
    Epoch 67/100
    13/13 [==============================] - 2s 158ms/step - loss: 0.4060 - acc: 0.9531
     - 15s - loss: 0.0065 - acc: 0.9981 - val_loss: 0.4060 - val_acc: 0.9531
    Epoch 68/100
    13/13 [==============================] - 2s 161ms/step - loss: 0.3314 - acc: 0.9609
     - 15s - loss: 0.0212 - acc: 0.9922 - val_loss: 0.3314 - val_acc: 0.9609
    Epoch 69/100
    13/13 [==============================] - 2s 148ms/step - loss: 0.6393 - acc: 0.9492
    
    Reached 99.9% accuracy so cancelling training!
     - 15s - loss: 0.0027 - acc: 0.9990 - val_loss: 0.6393 - val_acc: 0.9492



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
plt.legend(loc=0)
plt.figure()


plt.show()
```


![png](Exercise%207%20-%20Answer_files/Exercise%207%20-%20Answer_9_0.png)



    <Figure size 576x396 with 0 Axes>

