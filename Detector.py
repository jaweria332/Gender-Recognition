#Importing necessary libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPooling2D,Activation,Dropout,Dense,Flatten
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import random

#Initializing parameters
epochs=20
lr=1e-2
batch_size=64
img_dim=(48,48,3)

labels=[]
data=[]
#Loading image file
img_file=[f for f in glob.glob("E:\\6th Semester\\AI Practical\\04 Gender Recognition\\Gender-Recognition\\dataset"+"/**/*",recursive=True) if not os.path.isdir(f)]
random.shuffle(img_file)

#Convert image to array and label the category
for img in img_file:
    #Reading the image
    image=cv2.imread(img)
    #resizing
    image=cv2.resize(image,(img_dim[0],img_dim[1]))
    #Converting images to array
    image=img_to_array(image)
    #Append resulting array into data (variable defined above)
    data.append(image)

    label=img.split(os.path.sep)[-2]
    if label=='woman':
        label=1
    else:
        label=0

    #[[1] [0] [1] ......]
    labels.append([label])


# Lets begin the data preprocessing
data=np.array(data,dtype="float")/255.0
labels=np.array(label)

#Splitting dataset
X_train, X_test,Y_train,Y_test=train_test_split(data,labels,test_size=0.2,random_state=42)
Y_train=to_categorical(Y_train,num_classes=2)
Y_test=train_test_split(Y_test,num_classes=2)
