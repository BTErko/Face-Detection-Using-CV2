import numpy as np
import cv2 as cv

def videoDetector(fileName):

	video = cv.VideoCapture('TestImagesAndVideo/TestVideo/' + fileName)
	video.set(cv.CAP_PROP_FPS, 20)
	classifier = cv.CascadeClassifier('faceClassifier.xml')

	vwidth = video.get(cv.CAP_PROP_FRAME_WIDTH)
	vheight = video.get(cv.CAP_PROP_FRAME_HEIGHT)

	fourcc = cv.VideoWriter_fourcc(*'mp4v')
	out = cv.VideoWriter('Detected/Video/' + fileName, fourcc, 20, (int(vwidth), int(vheight)))

	while True:
		exist, frame = video.read()
		
		if not exist:
			break
	
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

		faces = classifier.detectMultiScale(gray, 1.1, 15)
	
		for (x,y,w,h) in faces:
			cv.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)

		out.write(frame)	
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

	video.release()
	out.release()
	cv.destroyAllWindows()


videoDetector('test_video.mp4')