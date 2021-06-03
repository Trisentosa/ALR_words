#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib


# In[2]:


cap = cv2.VideoCapture(0)


# In[3]:


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# In[4]:


recording=False
record_n = 1
frames = []


# In[5]:


import time


# In[1]:


while True:
    _, frame = cap.read()
    height,width,color = frame.shape

    # grayscale format
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)    
    
    left_lip, top_lip,bottom_lip, right_lip =0,0,0,0
    margin_x,margin_y = 5,7
    #loop coordinates
    for face in faces:
        # face: [(313, 234) (492, 413)] -> [(topleft) (rightbottom)]
#         x1 = face.left()
#         y1= face.top()
#         x2 = face.right()
#         y2 = face.bottom()

        # args: image, bounding box of the image
        landmarks = predictor(gray, face)
        
        # lip is part 48 - 68
        # top left is (part x of n=49, part y of top 53) and bottom right is (part x of n=55, part y of bottom 58)
        
        left_lip = landmarks.part(48).x
        top_lip = landmarks.part(52).y
        right_lip = landmarks.part(54).x
        bottom_lip = landmarks.part(57).y
        
#         for n in range(48,68):
#             x = landmarks.part(n).x
#             y = landmarks.part(n).y
#             cv2.circle(frame, (x,y) , 3, (0,0,255))

        # create rectangle around lip
#         cv2.rectangle(frame, (left_lip-margin_x, top_lip-margin_y), (right_lip+margin_x ,bottom_lip+margin_y),(0,255,0),2,-1)
        # create rectangle around face
        #  image, topleft, bottom right, color, thicknes, color fullness
        # cv2.rectangle(frame, (x1,y1), (x2,y2),(0,255,0), 3, -1)

    roi = frame[top_lip-margin_y : bottom_lip+margin_y, left_lip-margin_x : right_lip+margin_x]
    
    key = cv2.waitKey(1)
    
    # record frames
    if recording:
        timer = 1-(time.time() - start)
        text = " Timer: " + str(timer)
        text_display = cv2.putText(frame, text , (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2)
        print(str(timer))
        cv2.rectangle(frame, (left_lip-margin_x, top_lip-margin_y), (right_lip+margin_x ,bottom_lip+margin_y),(0,255,0),2,-1)
        frames.append(np.array(roi))
        if(timer <= 0):
            recording = False   
            print("stop capturing")
    else:
        text = " Timer: 0.00"
        text_display = cv2.putText(frame, text , (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2)

    cv2.imshow("Frame",frame)
    
    # escape key, close cam
    if key ==27:
        cv2.destroyAllWindows()
        break
    elif key == 32:
        recording = True
        start = time.time()
        print(start)


# In[ ]:


ind_jump = len(frames)//10
index = 0
cut_frame = []
for n in range(0,10):
    cut_frame.append(frames[index])
    index += ind_jump


# In[ ]:


plt.imshow(cut_frame[0])


# In[ ]:


plt.imshow(cut_frame[2])


# In[ ]:


plt.imshow(cut_frame[3])


# In[ ]:


plt.imshow(cut_frame[5])


# In[ ]:


plt.imshow(cut_frame[7])


# In[ ]:


plt.imshow(cut_frame[9])


# In[ ]:




