import cv2
import numpy as np
import face_recognition

imgElon= face_recognition.load_image_file('ImagesBasic/Bill Gates.jpg')
imgElon= cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

imgtest= face_recognition.load_image_file('ImagesBasic/Bill_gates.jpg')
imgtest= cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facelocTest = face_recognition.face_locations(imgtest)[0]
encodeTest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis= face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
cv2.putText(imgtest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Elon Mask',imgElon)
cv2.imshow('Elon Test',imgtest)
cv2.waitKey(0)