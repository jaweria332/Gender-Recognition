from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

#Load the model
model=load_model('Detector.model')

#Open webcam
webcam = cv2.VideoCapture(0)

#Defining classes
classes=["man","woman"]

#Reading webcam stream
while True:
    status,frame=webcam.read()

    #Detect face
    face,confidence=cv.detect_face(frame)

    #Loop through detected faces
    for idx, f in enumerate(face):
        #Get rectangle cordinates
        (X_str,Y_str,X_end,Y_end)=f[0],f[1],f[2],f[3]

        #draw rectangle over face
        cv2.rectangle(frame,(X_str,Y_str),(X_end,Y_end),(0.255,0),2)

        #Crop Region of Interest - detected face
        face_crop=np.copy(frame[Y_str:Y_end,X_str:X_end])

        if(face_crop.shape[0]<10) or (face_crop.shape[1]<10)<10:
            continue

        #Preprocessing our gender detection model
        face_crop=cv2.resize(face_crop,(96,96))
        face_crop=face_crop.astype("float")/255.0
        face_crop=img_to_array(face_crop)
        face_crop=np.expand_dims(face_crop,axis=0)

        #Apply gender detection on detected face
        conf=model.predict(face_crop)[0]

        #Get label for prediction
        idx=np.argmax(conf)
        label=classes[idx]

        label="{}: {:.2f}%".format(label,conf[idx]*100)

        Y=Y_str-10 if Y_str -10 >10 else Y_str +10

        #Write label
        cv2.putText(frame,label,(X_str,Y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)

    #Display output
    cv2.imshow("Gender Detection",frame)

    #Press Q to stop detection
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

#Release resources
webcam.release()
cv2.destroyAllWindows()