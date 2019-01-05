import pickle
import numpy as np
import os
import glob
from scipy import misc
from PIL import Image
from imageop import rgb2grey
import csv
import cv2

#class layers:
    

class MLP():
    
    def logistic_function(self,x):
        return .5 * (1 + np.tanh(.5 * x))
        
    
    
class preProcess():

    def processimg(self,path):
        png = [];
        for image_path in sorted(glob.glob(r""+path+"")):
            img = cv2.imread(image_path).astype('uint8')
            I = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            height, width = I.shape
            I = np.reshape(I,height*width)
            png.append(I)
        
        return np.asarray(png)
        
        
    def modifyimg(self,path):
        i = 0
        for image_path in sorted(glob.glob(r""+path+"")):
            tm = Image.open(image_path).convert('LA')
            tm = tm.resize((50,50))
            name = os.path.dirname(image_path)
            tm.save(name+str(i)+'.png')
            i = i+1
    