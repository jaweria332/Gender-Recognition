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
lr=1e-3
batch_size=64
img_dim=(48,48,3)

labels=[]
data=[]
#Loading image file
img_file=[f for f in glob.glob(r"E:\\6th Semester\\AI Practical\\04 Gender Recognition\\Gender-Recognition\\dataset"+"/**/*",recursive=True) if not os.path.isdir(f)]
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
labels=np.array(labels)

#Splitting dataset
X_train, X_test,Y_train,Y_test=train_test_split(data,labels,test_size=0.2,random_state=42)
Y_train=to_categorical(Y_train,num_classes=2)
Y_test=train_test_split(Y_test,num_classes=2)

#Augmenting dataset
aug=ImageDataGenerator(rotation_range=25,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")

#Defining a  model
def built(w,h,d,classes):
    model=Sequential()
    inputShape=(h,w,d)
    chanDim=-1

    if K.image_data_format()=="channels_first":
        inputShape=(d,h,w)
        chanDim=1

    model.add(Conv2D(32,(3,3),padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model



#Build a model
model=built(w=img_dim[0],h=img_dim[1],d=img_dim[2],classes=2)
opt=Adam(lr=lr, decay=lr/epochs)
model.compile(loss="binary_crossentropy",optimizer=opt, metrics=["accuracy"])

#Training a model
H = model.fit_generator(aug.flow(X_train,Y_train,batch_size=batch_size),validation_data=(X_test,Y_test),steps_per_epoch=len(X_train)//batch_size,epochs=epochs,verbose=1)

#Saving the model for later use
model.save('Detector.model')

#plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0,N),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,N),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,N),H.history["acc"],label="train_acc")
plt.plot(np.arange(0,N),H.history["val_acc"],label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

#Save plot to disk
plt.savefig('plot.png')