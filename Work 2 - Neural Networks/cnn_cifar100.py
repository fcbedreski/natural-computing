#Convolutional Neural Network CNN with Cifar100 dataset

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

# Normalize pixels to have values between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#Class storage
aux_list = []
fine_file = open("fine.txt", "r")
data = fine_file.readlines()

for line in data:
	fine_class = line.split(",")
	aux_list.append(fine_class)

fine_classes = aux_list[0]

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(fine_classes[int(train_labels[i])])
plt.show()

#Convolutional base creation
model = models.Sequential()
model.add(layers.Conv2D(256, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(100, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(100, (3, 3), activation='relu'))

#To show the model architecture
#model.summary()

#On the top -> dense layers
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(100, activation = 'softmax')) #softmax normalize the data

#Compare the differences
#model.summary()

#Compile and test the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

historic = model.fit(train_images, train_labels, epochs=30,
                    validation_data=(test_images, test_labels))


#Model evaluation
plt.plot(historic.history['accuracy'], label='accuracy')
plt.plot(historic.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_accuracy = model.evaluate(test_images,  test_labels, verbose=2)


print(test_accuracy)
#1st Result: 0.7192000150680542 - aprox 70% of accuracy with Cifar10, 3 layers of 32 and 64 neurons
