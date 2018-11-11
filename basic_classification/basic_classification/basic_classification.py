# The following is an exploration of tensorflow keras api using the fashion
# MINST data set. The tutorial is available:
# https://www.tensorflow.org/tutorials/keras/basic_classification

# TensorFlow and tf.keras
import tensorflow as tf
import matplotlib
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Each label corresponds to an integer 0-9 corresponding to these class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#exploring the data
#train_images.shape
#len(train_labels)
#train_labels

#test_images.shape
#len(test_labels)

#plt.figure()
#plt.imshow(train_images[0]) #to inspect the first image
#plt.colorbar()
#plt.grid(False)
#plt.show() #don't forget this, same from ML for trading

#process the pixel values to be in 0 - 1 range by 
train_images = train_images / 255.0
test_images = test_images / 255.0

#data verification
#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])
#plt.show()

#building the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #transform the 28x28 array to a 1D linear array , first layer
    keras.layers.Dense(128, activation=tf.nn.relu), #fully connected nerural layer
    keras.layers.Dense(10, activation=tf.nn.softmax) #fully connected neural layer of softmax scores (probabilities)
])

model.compile(optimizer=tf.train.AdamOptimizer(), #update
              loss='sparse_categorical_crossentropy', #the function we wan to minimize
              metrics=['accuracy']) #what metric do we train on

#training
model.fit(train_images, train_labels, epochs=5)

#testing
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

#making predictions
predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()