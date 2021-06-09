#imports
import numpy as np
import os
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt

import imageio
from imutils import face_utils

cap = cv2.VideoCapture(0)

#dlib predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#load model
from tensorflow import keras
model = keras.models.load_model('complete_saved_model/')

recording=False
text = "---"
frames = []

words = ['Begin', 'Choose', 'Connection', 'Navigation', 'Next', 'Previous', 'Start', 'Stop', 'Hello', 'Web']

import skimage
from skimage.transform import resize

count = 0
while True:
    #record
    _, frame = cap.read()
    height,width,color = frame.shape

    # frame to grayscale format
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)    

    x, y, w, h, left_lip, right_lip, bottom_lip, top_lip, margin_x, margin_y = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    key = cv2.waitKey(1)
    
    text_display = cv2.putText(frame, text , (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,255,255), 2)
    # record frames
    if recording:
        text = "Recording... "
        text_display = cv2.putText(frame, text , (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,255,255), 2)
        #loop coordinates
        for face in faces:
            # args: image, bounding box of the image
            landmarks = predictor(gray, face)
            left_lip = landmarks.part(48).x
            top_lip = landmarks.part(52).y
            right_lip = landmarks.part(54).x
            bottom_lip = landmarks.part(57).y
        
            landmarks = face_utils.shape_to_np(landmarks)
            name, i, j = 'mouth', 48, 68
        
        margin_x,margin_y = 5,7        
        (x, y, w, h) = cv2.boundingRect(np.array([landmarks[i:j]])) 
        
        roi = gray[y:y+h, x:x+w]
        roi = imutils.resize(roi, width = 250, inter=cv2.INTER_CUBIC) 
        
        cv2.rectangle(frame, (left_lip-margin_x, top_lip-margin_y), (right_lip+margin_x ,bottom_lip+margin_y),(0,255,0),2,-1)
        # save frames for new datasets
        cv2.imwrite("new_dataset/color%d.jpg" % count, roi)
        frames.append(np.array(roi))
        count+=1
        

    elif len(frames) > 0:
        sequence = []
        for frame in frames:
            frame = resize(frame, (100,100))
            frame = 255 * frame
            frame = frame.astype(np.uint8)
            sequence.append(frame)
        pad_array = [np.zeros((100, 100))]                            
        sequence.extend(pad_array * (22 - len(sequence)))
        sequence = np.array(sequence)
            
        #normalize
        np.seterr(divide='ignore', invalid='ignore') # ignore divide by 0 warning
        v_min = sequence.min(axis=(1, 2), keepdims=True)
        v_max = sequence.max(axis=(1, 2), keepdims=True)
        sequence = (sequence - v_min)/(v_max - v_min)
        sequence = np.nan_to_num(sequence)
            
        my_pred = sequence.reshape(1,22,100,100,1)
        ans = model.predict(my_pred)
#             index = np.argmax(ans,)
        text = "Predicted : " + words[np.argmax(ans,)]
        text_display = cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,255,255), 2)
        frames = []
        
    cv2.imshow("ALR, space- record, stop",frame)
    
    # Escape key, close cam
    if key ==27:
        cv2.destroyAllWindows()
        break
    # Spacebar , stop and record
    elif key == 32:
        if recording == True:
            recording = False
        else:
            recording = True