import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImageAtendence'
images= []

className = []
myList =os.listdir(path)



for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])
print(className)

def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendence(name):
    with open('attend/Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



encodeListKnow = findEncoding(images)

print('encoding complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgc = cv2.resize(img,(0,0),None,0.25,0.25)
    imgc = cv2.cvtColor(imgc, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(imgc)
    encodeCurrentFrame = face_recognition.face_encodings(imgc,facesCurrentFrame)

    for encodeface,faceLoc in zip(encodeCurrentFrame,facesCurrentFrame):
        matches =face_recognition.compare_faces(encodeListKnow,encodeface)
        faceDic = face_recognition.face_distance(encodeListKnow,encodeface)
        #print(faceDic)
        matchindex = np.argmin(faceDic)


        if matches[matchindex]:


            name = className[matchindex].upper()
           # print(name)

            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 =y1*4,x2*4,y2*4,x1*4

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)

            markAttendence(name)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)