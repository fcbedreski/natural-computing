import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

#fine: apenas a classe da imagem
#coarse: a superclasse da imagem
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

#normalização 
train_images = train_images/255.0 
test_images = test_images/255.0 

#armazenamento das classes
aux_list = []
fine_file = open("fine.txt", "r")
data = fine_file.readlines()
	
for line in data: 
	aux_class = line.split(",")
	aux_list.append(aux_class)
	
classes = aux_list[0]

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


#Object oriented
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape = (32,32,3))) #Cifar100 input
model.add(keras.layers.Dense(1024, activation = 'relu')) #middle layers
model.add(keras.layers.Dense(512, activation = 'relu')) 
model.add(keras.layers.Dense(256, activation = 'relu')) 
model.add(keras.layers.Dense(128, activation = 'relu')) 
model.add(keras.layers.Dense(100, activation = 'softmax')) #output

#Compile the model
model.compile(optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = ['accuracy']
)

#train the model
modelo.fit(train_images, train_labels, epochs = 50)

#Plotting the first 25 images
plt.figure(figsize=(10, 10))
for i in range(25):
  plt.subplot(5, 5, i + 1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap = plt.cm.binary)
  plt.xlabel(classes[int(train_labels[i])])
plt.show()
