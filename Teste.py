import pickle
import numpy
import os
import glob
from scipy import misc
from PIL import Image
from imageop import rgb2grey
import csv
import cv2

salvar = open("metabasedogs.txt", "w")
misc = misc
np = numpy
png = [];
Image = Image
i = 0
count = 1

outF = open("myOutFile.txt", "w")



if(i<1):
    for image_path in sorted(glob.glob(r"F:\Treinar\*\*.png")):
        outF = open("myOutFile.txt", "w")
        img = cv2.imread(image_path).astype('uint8')
        I = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #print(I)
        height, width = I.shape
        I = np.reshape(I,height*width)
        #print(np.prod(I.shape))        
        #I = np.append(I, i)
        for i in range(np.size(I)):
            # write line to output file
            outF.write('@attribute \"pixel'+str(i)+'\" integer')
            outF.write("\n")
        print(I)
        i = i+5
        outF.close()
        #png.append()
        '''png.append(I)
        if(count%100 == 0):
            print(os.path.dirname(image_path))
            print(i)
            print(count)
            print("if"+str(count%100)) 
            i = i+1  
            print(i)
        count = count + 1
        tm = Image.open(image_path).convert('LA')
        tm = tm.resize((50,50))
        name = os.path.dirname(image_path)
        tm = np.resize(tm, (1))
        tm.save(name+str(i)+'.png')
        #png.append(misc.imread(image_path));
        #i = i+1
        #png.append(vector)
'''
im = np.asarray(png)

#print(im)
#pickle.dump(im,salvar)
#np.savetxt('basedefinitiva.txt', im.astype(int), fmt='%.0f',delimiter=',', newline='\n')
