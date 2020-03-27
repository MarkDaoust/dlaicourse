```python
!pip install tensorflowjs
```


```python
import numpy as np
import tensorflow as tf

print('\u2022 Using TensorFlow Version:', tf.__version__)
```


```python
model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[1])  
])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)
```


```python
print(model.predict([10.0]))
```


```python
import time
saved_model_path = "./{}.h5".format(int(time.time()))

model.save(saved_model_path)
```


```python
!tensorflowjs_converter --input_format=keras {saved_model_path} ./
```
