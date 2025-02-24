import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from PIL import Image
import cv2
import MLPagain
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
     

class MLP():
    def __init__(self, size_layers, learning_rate=0.2, reg_lambda=0, bias_flag=True):
        self.size_layers = size_layers
        self.n_layers    = len(size_layers)
        self.l_rate      = learning_rate
        self.act_f       = 'sigmoid'
        self.lambda_r    = reg_lambda
        self.bias   = bias_flag
 
        self.initialize_weights()
        print(self.weights)
        
    def initialize_weights(self):
        self.weights = []
        size_next_layers = self.size_layers.copy()
        size_next_layers.pop(0)
    
        for size_layer, size_next_layer in zip(self.size_layers, size_next_layers):
            if self.bias:  
                neuron_weights = 2.0 * np.random.rand(size_next_layer, size_layer + 1) - 1
            else:
                neuron_weights = 2.0 * np.random.rand(size_next_layer, size_layer) - 1                            
            self.weights.append(neuron_weights) 
        return self.weights
        
    
    def train2(self, X, Y, iterations=400, reset=False):
        '''
        Given X (feature matrix) and y (class vector)
        Updates the Theta Weights by running Backpropagation N tines
        Arguments:
            X          : Feature matrix [n_examples, n_features]
            Y          : Sparse class matrix [n_examples, classes]
            iterations : Number of times Backpropagation is performed
                default = 400
            reset      : If set, initialize Theta Weights before training
                default = False
        '''
        n_examples = Y.shape[0]
#        self.labels = np.unique(y)
#        Y = np.zeros((n_examples, len(self.labels)))
#        for ix_label in range(len(self.labels)):
#            # Find examples with with a Label = lables(ix_label)
#           ix_tmp = np.where(y == self.labels[ix_label])[0]
#            Y[ix_tmp, ix_label] = 1

        if reset:
            self.initialize_weights()
        for iteration in range(iterations):
            self.gradients = self.backpropagation(X, Y) 
            self.gradients_vector = self.unroll_weights(self.gradients)
            self.theta_vector = self.unroll_weights(self.theta_weights)
            self.theta_vector = self.theta_vector - self.gradients_vector
            self.theta_weights = self.roll_weights(self.theta_vector)

    def train(self, X, Y):

        n_examples = X.shape[0]
        inputs = []
        outputs = []
        Z = []
        C = []
        # Feedforward
        print(self.n_layers)
        # PASSA AS IMAGENS UMA POR UMA
        for neuron in X:
            print(neuron)
            inputs, Z = self.feedforward(neuron, self.weights)
            if(Z < 0.5):
                outputs.append(0)
            else:
                outputs.append(1)
                
            print("A: ", inputs)
            print("Z: ",Z)                
        print(outputs)
        
        

            

    def logistic_function(self,x):
        return .5 * (1 + np.tanh(.5 * x))


    def update_weights(self, index, output, expected_output, neuron):
        for i in neuron:
            value = self.l_rate*(expected_output - output)*neuron[i]
        self.weights[index] = self.weights[index] + value


    def feedforward(self, X, weights):
        new_input=[]
        f = lambda x: self.logistic_function(x)
        
        for n in range(self.n_layers-1):
            W = weights[n]
            for i in range(W.shape[0]):
                value = 0
                for j in range(np.prod(X.shape)):
                    value = value + X[j]*W[i][j]
                value = f(value)
                new_input.append(value)
            X=np.asarray(new_input)
        return new_input, X[j+1]                   
    
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
            tm = tm.resize((200,200))
            name = os.path.dirname(image_path)
            tm.save(name+str(i)+'.png')
            i = i+1

if __name__ == "__main__":
    
    
    '''
    pr = preProcess()
    pr.modifyimg("F:\Testar\*\*.jpg")
    I = pr.processimg("F:\Testar\*.png")
    np.savetxt('basetesteAV.txt', I.astype(int), fmt='%.0f',delimiter=',', newline='\n')

    Dir2="nome"
    i=-1
    Labels = []
    for imagepath in sorted(glob.glob(r"F:\Testar\*\*.jpg")):
        Dir1 = os.path.dirname(imagepath)
        if(Dir1 != Dir2):
            Dir2 = os.path.dirname(imagepath)
            i = i + 1
            #fim do if
        Labels.append(i)
    #fim do for
    
    Im = np.asarray(Labels)
    np.savetxt('labels_basetesteAV.txt', Im.astype(int), fmt='%.0f',delimiter=',', newline='\n')

    
    
'''
    data_train = np.loadtxt('basetreinoAV.txt', dtype=int, delimiter=',')
    data_test = np.loadtxt('basetesteAV.txt', dtype=int, delimiter=',')
    data_train_labels = np.loadtxt('labels_basetreinoAV.txt', dtype=int, delimiter='\n')
    data_test_labels = np.loadtxt('labels_basetesteAV.txt', dtype=int, delimiter='\n')
    
    print('Training data shape : ', data_train.shape, data_train_labels.shape)
    print('Testing data shape : ', data_test.shape, data_test_labels.shape)
    
    teste = data_train[0].reshape(200,200)
    
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
    plt.title("Exemplo de imagem usada")
    
    #plt.show()

    
    # Change the labels from integer to categorical data
    train_labels_one_hot = to_categorical(data_train_labels)
    test_labels_one_hot = to_categorical(data_test_labels)
    
    # Display the change for category label using one-hot encoding
    print('Original label 0 : ', data_train_labels[0])
    print('After conversion to categorical ( one-hot ) : ', train_labels_one_hot[0])
        
    mlp_classifier = MLPagain.Mlp(size_layers = [40000, 2], 
                                  reg_lambda  = 0,
                                  bias_flag   = False)
    
    
    epochs = 500
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
    plt.show()
    # Training Accuracy
    Y_hat = mlp_classifier.predict(data_train)
    y_tmp = np.argmax(Y_hat, axis=1)
    y_hat = classes[y_tmp]
    
    acc = np.mean(1 * (y_hat == data_train_labels))
    error = np.mean(1 * (y_hat != data_train_labels))
    print("---------------------------")
    print('Training Error: ' + str(error*100))
    print('Training Accuracy: ' + str(acc*100))
    
    
    # Test Accuracy
    Y_hat = mlp_classifier.predict(data_test)
    y_tmp = np.argmax(Y_hat, axis=1)
    y_hat = classes[y_tmp]
    
    acctest = np.mean(1 * (y_hat == data_test_labels))
    errortest = np.mean(1 * (y_hat != data_test_labels))
    print("-----------TEST INFO-----------")    
    print('Test Error: ' + str(errortest*100))
    print('Test Accuracy: ' + str(acctest*100))#'''
    
    true_class1 = 0
    false_class1 = 0
    true_class2 = 0
    false_class2 = 0

    
    for i in range(len(data_test_labels)):
        if(data_test_labels[i]==0):
            if(y_hat[i] == 0):
                true_class1 = true_class1 + 1
            else:
                false_class1 = false_class1 + 1
        else:
            if(y_hat[i] == 1):
                true_class2 = true_class2 + 1
            else:
                false_class2 = false_class2 + 1
    
    
    
    
    total_exemplos1 = true_class1 + false_class1
    total_exemplos2 = true_class2 + false_class2
    total_previsto1 = true_class1 + false_class2
    total_previsto2 = true_class2 + false_class1
    
    prec_rot = true_class1/total_previsto1
    prec_shi = true_class2/total_previsto2
    rec_rot  = true_class1/total_exemplos1
    rec_shi  = true_class1/total_exemplos2
    f1_rot   = ((prec_rot * rec_rot)/(prec_rot + rec_rot)) * 2
    f1_shi   = ((prec_shi * rec_shi)/(prec_shi + rec_shi)) * 2
    f1_total = (f1_rot + f1_shi)/2
    
    print("F1 score(Total): ", f1_total)
    print("---------------------------")
    print("******Rottweiler******")    
    print("Precision(Rottweiler): ", prec_rot)
    print("Recall(Rottweiler): ", rec_rot)
    print("F1 score(Rottweiler): ", f1_rot)
    print("---------------------------")
    print("******Shitzu******") 
    print("Precision(Shitzu): ", prec_shi)
    print("Recall(Shitzu): ", rec_shi)
    print("F1 score(Shitzu): ", f1_shi)
    print("---------------------------")
    
    
    
    
    #Cria a tabela
    x = PrettyTable(["Matriz de confusao", "Rottweiler", "Shitzu", "Total de elementos"])

    #Alinha as colunas
    x.align["Matriz de confusao"] = "l"
    x.align["Rottweiler"] = "l"
    x.align["Shitzu"] = "r"
    x.align["Total de elementos"] = "r"

    #Deixa um espaco entre a borda das colunas e o conteudo (default)
    x.padding_width = 1
    
    x.add_row(["Rottweiler",true_class1,false_class1, total_exemplos1]) 
    x.add_row(["Shitzu",false_class2,true_class2, total_exemplos2])
    x.add_row(["Total previstos",total_previsto1, total_previsto2," "]) 
    print(x)                

    
    