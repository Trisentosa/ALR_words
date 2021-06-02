import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    
    # grayscale format
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)    
    for face in faces:
        # face: [(313, 234) (492, 413)] -> [(topleft) (rightbottom)]
        # extract coordinates of face
        x1 = face.left()
        y1= face.top()
        x2 = face.right()
        y2 = face.bottom()

        # args: image, bounding box of the image
        landmarks = predictor(gray, face)
        # print(landmarks)
        
        # lip is part 48 - 68
        
        for n in range(48,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x,y) , 3, (0,0,255))
            print("point x of ", n, " : ", x)
            print("point y of ", y, " : ", y)

        # create rectangle around face
        #  image, topleft, bottom right, color, thicknes, color fullness
        # cv2.rectangle(frame, (x1,y1), (x2,y2),(0,255,0), 3, -1)

    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)
    
    # escape key, close cam
    if key ==27:
        break