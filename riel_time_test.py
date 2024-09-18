import cv2
import numpy as np
import tensorflow as tf
from deepface import DeepFace


video=cv2.VideoCapture(0)
#find the faces
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces= faceDetect.detectMultiScale(gray, 1.3, 3)
    for x,y,w,h in faces:
        sub_face_img=gray[y:y+h, x:x+w]
        #resized=cv2.resize(sub_face_img,(48,48))
        #normalize=resized/255.0
        result=DeepFace.analyze(frame,actions=['emotion'])
        emo=result[0]['dominant_emotion']
        accuracy_of_dominant_emotion = result[0]['emotion'][emo]
        print(emo,": ",accuracy_of_dominant_emotion)  
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, emo, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()