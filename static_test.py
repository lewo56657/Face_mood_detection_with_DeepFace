import tensorflow 
import matplotlib.pyplot  as plt
from  deepface import DeepFace
import cv2
import numpy as np
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#labels_dict={0:'sinirli',1:'iğrenmiş', 2:'korkmuş', 3:'mutlu',4:'normal',5:'uzgun',6:'şaşırmış'}

img=cv2.imread("C:/Users/levent/Downloads/data2/surprised/images113.jpg")
rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
faces= faceDetect.detectMultiScale(img, 1.3, 3)
for x,y,w,h in faces:
    sub_face_img=rgb[y:y+h, x:x+w]
    #resized=cv2.resize(sub_face_img,(48,48))
    #normalize=resized/255.0
    #reshaped=np.reshape(normalize, (1, 48, 48, 1))
    result=DeepFace.analyze(img,actions=['emotion'])
    emo=result[0]['dominant_emotion']
    accuracy_of_dominant_emotion = result[0]['emotion'][emo]
    print(emo,": ",accuracy_of_dominant_emotion)  
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 1)
    cv2.rectangle(img,(x,y),(x+w,y+h),(50,50,255),2)
    cv2.rectangle(img,(x,y-40),(x+w,y),(50,50,255),-1)
    cv2.putText(img,emo, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

cv2.imshow("Frame",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

     