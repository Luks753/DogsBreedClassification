import keras
import numpy as np
import matplotlib.pyplot as plt
import MLPagain as mlp
import pickle,gzip
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense


#'''
# As 'mnist.pkl.gz' was created in Python2, 'latin1' encoding is needed to loaded in Python3
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

# Plot random examples
examples = np.random.randint(10000, size=8)
n_examples = len(examples)
plt.figure()
for ix_example in range(n_examples):
    tmp = np.reshape(train_set[0][examples[ix_example],:], [28,28])
    ax = plt.subplot(1,n_examples, ix_example + 1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.title(str(train_set[1][examples[ix_example]]))
    plt.imshow(tmp, cmap='gray')


# Training data
train_X = valid_set[0]
train_y = valid_set[1]  
print('Shape of training set: ' + str(train_X.shape))

# change y [1D] to Y [2D] sparse array coding class
n_examples = len(train_y)
labels = np.unique(train_y)
train_Y = np.zeros((n_examples, len(labels)))
for ix_label in range(len(labels)):
    # Find examples with with a Label = lables(ix_label)
    ix_tmp = np.where(train_y == labels[ix_label])[0]
    train_Y[ix_tmp, ix_label] = 1


# Test data
test_X = test_set[0]
test_y = test_set[1] 
print('Shape of test set: ' + str(test_X.shape))

# change y [1D] to Y [2D] sparse array coding class
n_examples = len(test_y)
labels = np.unique(test_y)
test_Y = np.zeros((n_examples, len(labels)))
for ix_label in range(len(labels)):
    # Find examples with with a Label = lables(ix_label)
    ix_tmp = np.where(test_y == labels[ix_label])[0]
    test_Y[ix_tmp, ix_label] = 1

# Creating the MLP object initialize the weights
mlp_classifier = mlp.Mlp(size_layers = [784, 25, 10, 10], 
                         act_funct   = 'relu',
                         reg_lambda  = 0,
                         bias_flag   = True)
print(mlp_classifier)

# Training with Backpropagation and 400 iterations
iterations = 100
loss = np.zeros([iterations,1])

for ix in range(iterations):
    mlp_classifier.train(train_X, train_Y, 1)
    Y_hat = mlp_classifier.predict(train_X)
    y_tmp = np.argmax(Y_hat, axis=1)
    y_hat = labels[y_tmp]
    
    loss[ix] = (0.5)*np.square(y_hat - train_y).mean()

# Ploting loss vs iterations
plt.figure()
ix = np.arange(iterations)
plt.plot(ix, loss)

# Training Accuracy
Y_hat = mlp_classifier.predict(train_X)
y_tmp = np.argmax(Y_hat, axis=1)
y_hat = labels[y_tmp]

print(y_hat)
acc = np.mean(1 * (y_hat == train_y))
print('Training Accuracy: ' + str(acc*100))

# Test Accuracy
Y_hat = mlp_classifier.predict(test_X)
y_tmp = np.argmax(Y_hat, axis=1)
y_hat = labels[y_tmp]

print(y_hat)
print(test_y)
acc = np.mean(1 * (y_hat == test_y))
print('Testing Accuracy: ' + str(acc*100))#'''

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


'''
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


# Change the labels from integer to categorical data
train_labels_one_hot = to_categorical(data_train_labels)
test_labels_one_hot = to_categorical(data_test_labels)

# Display the change for category label using one-hot encoding
print('Original label 0 : ', data_train_labels[0])
print('After conversion to categorical ( one-hot ) : ', train_labels_one_hot[0])

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(np.prod(teste.shape),)))
model.add(Dense(512, activation='relu'))
model.add(Dense(nClasses, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(data_train, train_labels_one_hot, batch_size=256, epochs=1000, verbose=1, 
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

#'''



