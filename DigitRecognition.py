import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.dataset.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalise(x_train, axis = 1)
x_test = tf.keras.utils.normalise(x_test, axis = 1)

model = tf.keras.models.Sequential()
model.add(tf.Keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.Keras.layers.Dense(128, activation='relu'))
model.add(tf.Keras.layers.Dense(128, activation='relu'))
model.add(tf.Keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorial_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train)

model.fit(x_train, y_train, epoch= 3 )

model.save('unknown.model')

model = tf.keras.models.load_models('handwritten.model')

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)




