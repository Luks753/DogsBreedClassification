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

# Change the labels from integer to categorical data
train_labels_one_hot = to_categorical(data_train_labels)
test_labels_one_hot = to_categorical(data_test_labels)

# Display the change for category label using one-hot encoding
print('Original label 0 : ', data_train_labels[0])
print('After conversion to categorical ( one-hot ) : ', train_labels_one_hot[0])

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(np.prod(teste.shape),)))
model.add(Dense(512, activation='relu'))
model.add(Dense(nClasses, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(data_train, train_labels_one_hot, batch_size=256, epochs=20, verbose=1, 
                   validation_data=(data_test, test_labels_one_hot))

plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.show()

plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.show()

[test_loss, test_acc] = model.evaluate(data_test, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# Predict the most likely class
print(model.predict_classes(data_test[[0],:]))

