import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('ImageBasic/elonmask.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)


imgElonTest = face_recognition.load_image_file('ImageBasic/elonmask task.jpg')
imgElonTest = cv2.cvtColor(imgElonTest,cv2.COLOR_BGR2RGB)

facelocation = face_recognition.face_locations(imgElon)[0]
encodeElon =face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(facelocation[3],facelocation[0]),(facelocation[1],facelocation[2]),(255,0,255),2)

facelocationTest = face_recognition.face_locations(imgElonTest)[0]
encodeElonTest =face_recognition.face_encodings(imgElonTest)[0]
cv2.rectangle(imgElonTest,(facelocationTest[3],facelocationTest[0]),(facelocationTest[1],facelocationTest[2]),(255,0,255),2)

result = face_recognition.compare_faces([encodeElon],encodeElonTest)

faceDis = face_recognition.face_distance([encodeElon],encodeElonTest)

print(result,faceDis)

cv2.putText(imgElonTest,f'{result} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)



cv2.waitKey(0)