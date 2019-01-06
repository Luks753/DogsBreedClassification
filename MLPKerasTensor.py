import keras
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.utils import to_categorical
from keras.datasets import mnist

'''
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
 
print('Training data shape : ', train_images.shape, train_labels.shape)
 
print('Testing data shape : ', test_images.shape, test_labels.shape)
 
# Find the unique numbers from the train labels
classes = np.unique(train_labels)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)
 
plt.figure(figsize=[10,5])

 
# Display the first image in training data
plt.subplot(121)
plt.imshow(train_images[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_labels[0]))
 
# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_images[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_labels[0]))

plt.show()'''


#'''
data_train = np.loadtxt('basetreino.txt', dtype=float, delimiter=',')
data_test = np.loadtxt('baseteste.txt', dtype=float, delimiter=',')
data_train_labels = np.loadtxt('labels_basetreino.txt', dtype=int, delimiter='\n')
data_test_labels = np.loadtxt('labels_baseteste.txt', dtype=int, delimiter='\n')

print('Training data shape : ', data_train.shape, data_train_labels.shape)
 
print('Testing data shape : ', data_test.shape, data_test_labels.shape)

teste = data_train[0].reshape(32,32)

# Find the unique numbers from the train labels
classes = np.unique(data_train_labels)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

plt.figure(figsize=[10,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(teste[:,:], cmap='gray')
plt.title("Ground Truth : {}".format(teste))

plt.show()
#'''





