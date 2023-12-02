#Convolutional Neural Network (CNN) for image processing
# 13/02/23 - Monday

import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt 

#Loading the data
(train_images, train_labels), (test_images, test_labels) = 
datasets.cifar10.load_data()
train_images = train_images/255.0 
test_images = test_images/255.0 

#Be careful with normalization: dividing too much by 255 tends to zero (black) and the image disappears. Divide only once. 

#Cifar10 and Cifar100 is about the number of categories

categories = ['aircraft', 'car', 'bird', 'cat', 'hart', 'dog', 'frog', 'horse', 'boat', 'truck']

#Plotting of colorful images

plt.figure(figsize=(10,10))

for i in range (25) 
	plt.subplot(5, 5, i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False) 
	plt.imshow(train_images[i])
	plt.xlabel(categories[train_labels[i][0]])

plt.show

#Layers
modelo = models.Sequential()

modelo.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32, 32, 3))) #First 'input_shape' parameter: neuron number
modelo.add(layers.MaxPooling2D((2,2)))
modelo.add(layers.Conv2D(64, (3,3), activation = 'relu')) 
modelo.add(layers.MaxPooling2D((2,2)))
modelo.add(layers.Conv2D(64, (3,3), activation = 'relu'))
modelo.summary()

modelo.add(layers.Flatten())
modelo.add(layers.Dense(64, activation = 'relu'))
modelo.add(layers.Dense(10, activation = 'softmax')) #Softmax normalize the data - activation function

modelo.summary() #Check the differences



#15/02/23 - Wednesday - CNN: continued ---------------------------------------------------------------------

modelo.compile(optimizer:'adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

#Training and validating at the same time - at each training epoch, it performs a validation - training slows down, 
#but is ideal for training with little data 
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val-accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.4, 1.0])
plt.legend(loc='lower right')


