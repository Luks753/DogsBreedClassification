import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from PIL import Image
import cv2
import MLPagain
     
    

class MLP():
    def __init__(self, size_layers, act_funct='sigmoid', reg_lambda=0, bias_flag=True):
        self.size_layers = size_layers
        self.n_layers    = len(size_layers)
        self.act_f       = act_funct
        self.lambda_r    = reg_lambda
        self.bias_flag   = bias_flag
 
        self.initialize_weights()
        
    def initialize_weights(self):
        self.weights = []
        size_next_layers = self.size_layers.copy()
        size_next_layers.pop(0)
        for size_layer, size_next_layer in zip(self.size_layers, size_next_layers):
            epsilon = 4.0 * np.sqrt(6) / np.sqrt(size_layer + size_next_layer)
            if self.bias_flag:  
                theta_tmp = epsilon * ( (np.random.rand(size_next_layer, size_layer + 1) * 2.0 ) - 1)
            else:
                theta_tmp = epsilon * ( (np.random.rand(size_next_layer, size_layer) * 2.0 ) - 1)                              
            self.weights.append(theta_tmp)
        print(self.weights)
        return self.weights
        
    
    
    def logistic_function(self,x):
        return .5 * (1 + np.tanh(.5 * x))
        
    
    
class preProcess():
    
    def __init__(self):
        print ("ok")

    def processimg(self,path):
        png = []
        for image_path in sorted(glob.glob(r""+path+"")):
            img = cv2.imread(image_path).astype('uint8')
            I = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            height, width = I.shape
            I = np.reshape(I,height*width)
            png.append(I)

        return np.asarray(png)
        
        
    def modifyimg(self,path):
        i = 1
        for image_path in sorted(glob.glob(r""+path+"")):
            tm = Image.open(image_path).convert('LA')
            tm = tm.resize((32,32))
            name = os.path.dirname(image_path)
            tm.save(name+str(i)+'.png')
            i = i+1

if __name__ == "__main__":
    
    #FAZENDO TESTE USANDO A REDE PRONTA DO ARQUIVO MLPagain
    data_train = np.loadtxt('basetreino.txt', dtype=int, delimiter=',')
    data_test = np.loadtxt('baseteste.txt', dtype=int, delimiter=',')
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
    
    n_examples = len(data_train_labels)
    
    plt.figure(figsize=[10,5])
    
    # Display the first image in training data
    plt.subplot(121)
    plt.imshow(teste[:,:], cmap='gray')
    plt.title("Ground Truth : {}".format(teste))
    
    #plt.show()
    #'''
    
    # Change the labels from integer to categorical data
    train_labels_one_hot = to_categorical(data_train_labels)
    test_labels_one_hot = to_categorical(data_test_labels)
    
    # Display the change for category label using one-hot encoding
    print('Original label 0 : ', data_train_labels[0])
    print('After conversion to categorical ( one-hot ) : ', train_labels_one_hot)
        
    mlp_classifier = MLPagain.Mlp(size_layers = [1024, 512, 5], 
                                  reg_lambda  = 0,
                                  bias_flag   = True)
    print(mlp_classifier)
    
    epochs = 10
    loss = np.zeros([epochs,1])
    
    for ix in range(epochs):
        mlp_classifier.train(data_train, train_labels_one_hot, 1)
        Y_hat = mlp_classifier.predict(data_train)
        y_tmp = np.argmax(Y_hat, axis=1)
        y_hat = classes[y_tmp]
        
        loss[ix] = (0.5)*np.square(y_hat - data_train_labels).mean()
    
    # Ploting loss vs iterations
    plt.figure()
    ix = np.arange(epochs)
    plt.plot(ix, loss)
    
    # Training Accuracy
    Y_hat = mlp_classifier.predict(data_train)
    y_tmp = np.argmax(Y_hat, axis=1)
    y_hat = classes[y_tmp]
    
    acc = np.mean(1 * (y_hat == data_train_labels))
    print('Training Accuracy: ' + str(acc*100))
    
    # Test Accuracy
    Y_hat = mlp_classifier.predict(data_test)
    y_tmp = np.argmax(Y_hat, axis=1)
    y_hat = classes[y_tmp]
    
    acc = np.mean(1 * (y_hat == data_test_labels))
    print(y_hat)
    print('Testing Accuracy: ' + str(acc*100))

    
    
    
    
