from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train[1:5000], y_train[1:5000], epochs=5, validation_data=(x_train[5000:6000], y_train[5000:6000]))

model.evaluate(x_test,  y_test, verbose=2)

labels = model.predict(x_train[6000:7000])

a=0
b=0
for i in range(len(labels)):
    if (np.argmax(labels[i]) == y_train[6000+i]):
        a = a+1
    else:
        b = b+1

print (a/1000)
