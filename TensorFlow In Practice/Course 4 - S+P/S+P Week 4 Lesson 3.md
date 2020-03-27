```
!pip install tensorflow==2.0.0b1
```


```
import tensorflow as tf
print(tf.__version__)
```

    2.0.0-beta1



```
import numpy as np
import matplotlib.pyplot as plt
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
```


```
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/Sunspots.csv \
    -O /tmp/sunspots.csv
```

    --2019-07-01 15:07:56--  https://storage.googleapis.com/laurencemoroney-blog.appspot.com/Sunspots.csv
    Resolving storage.googleapis.com (storage.googleapis.com)... 74.125.195.128, 2607:f8b0:400e:c08::80
    Connecting to storage.googleapis.com (storage.googleapis.com)|74.125.195.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 70827 (69K) [application/octet-stream]
    Saving to: ‘/tmp/sunspots.csv’
    
    /tmp/sunspots.csv   100%[===================>]  69.17K  --.-KB/s    in 0.001s  
    
    2019-07-01 15:07:57 (95.3 MB/s) - ‘/tmp/sunspots.csv’ saved [70827/70827]
    



```
import csv
time_step = []
sunspots = []

with open('/tmp/sunspots.csv') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  next(reader)
  for row in reader:
    sunspots.append(float(row[2]))
    time_step.append(int(row[0]))

series = np.array(sunspots)
time = np.array(time_step)
plt.figure(figsize=(10, 6))
plot_series(time, series)
```


![png](S%2BP%20Week%204%20Lesson%203_files/S%2BP%20Week%204%20Lesson%203_4_0.png)



```
split_time = 3000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 60
batch_size = 32
shuffle_buffer_size = 1000


```


```
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset
```


```
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20, input_shape=[window_size], activation="relu"), 
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-7, momentum=0.9))
model.fit(dataset,epochs=100,verbose=0)



```




    <tensorflow.python.keras.callbacks.History at 0x7f4d79894e10>




```
forecast=[]
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]


plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)
plot_series(time_valid, results)
```


![png](S%2BP%20Week%204%20Lesson%203_files/S%2BP%20Week%204%20Lesson%203_8_0.png)



```
tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
```




    15.0819645


