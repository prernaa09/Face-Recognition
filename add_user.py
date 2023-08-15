import cv2, os, sys
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

sampleNumber = 0
id = 1
while(True):
	ret, img = cam.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = faceDetect.detectMultiScale(gray,1.8,5)

	for (x,y,w,h) in faces:
		sampleNumber +=1
		cv2.imwrite('Images/User.'+str(id)+'.'+str(sampleNumber)+'.jpg',gray[y-2:y+h+2,x-2:x+w+2])
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	cv2.imshow('face',img)
	print(sampleNumber)
	if sampleNumber >=300:
		break
	if cv2.waitKey(1) == ord('q'):
		break
cam.release()
cv2.destroyAllWindows()