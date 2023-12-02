#MultiLayer Perceptron - Neural Network - Natural Computing
# 30/01/23


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# print(train_images.shape)   
# print(test_images.shape)
# print(train_images[0])
# print(train_labels)

categories = ['t-shirt', 'pants', 'sweather', 'dress', 'coat', 'sandal', 'shirt', 'shoes', 'handbag', 'boot']

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#Transforming to 0-1 scale  
#Adjust the image to be better for ReLU and Sigmoid (grey scale)
train_images = train_images/255.0
test_images = test_images/255.0

# Plotting the first 25 images
plt.figure(figsize=(10, 10))
for i in range(25):
  plt.subplot(5, 5, i + 1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap = plt.cm.binary)
  plt.xlabel(categories[train_labels[i]])
plt.show()

#Obs: how you format the data for activation functions
# Sigmoid (0 - 1) and hyperbolic tangent (-1 - 1)

#Object Oriented
modelo = keras.Sequential()
modelo.add(keras.layers.Flatten(input_shape = (28,28))) #input
modelo.add(keras.layers.Dense(128, activation = 'relu')) #middle layer
modelo.add(keras.layers.Dense(64, activation = 'relu'))
modelo.add(keras.layers.Dense(10, activation = 'softmax')) #output
#n = neuron number

#Compile the model
modelo.compile(optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = ['accuracy']
)

#Train the model
modelo.fit(train_images, train_labels, epochs = 10)

