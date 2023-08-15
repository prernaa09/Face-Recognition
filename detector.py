import cv2, sys
import numpy as n

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('traningData.yml')
		

while(True):
	ret, img = cam.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = faceDetect.detectMultiScale(gray,1.8,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		id, conf = rec.predict(gray[y:y+h,x:x+w])
		print(id)
		if(id == 1) :
			cv2.putText(img, str("Prerna"), (x+10,y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
		elif(id == 2) :
			cv2.putText(img, str("Prerna"), (x+10,y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
		else :
			cv2.putText(img, str("Person-3"), (x+10,y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)

	cv2.imshow('face',img)
	if cv2.waitKey(1) == ord('q'):
		break
cam.release()
cv2.destroyAllWindows()

