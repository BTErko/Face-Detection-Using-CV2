# import libraries
import cv2 as cv
import numpy as np

# read image from file
filename = 'test_image2.jpg'

img = cv.imread('TestImagesAndVideo/TestImages/' + filename) 

# convert image to gray

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) 

classifier = cv.CascadeClassifier('faceClassifier.xml') 

facerect = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) 

for (x,y,w,h) in facerect:
    cv.rectangle(img,(x,y), (x+w,y+h), (0,255,0),thickness=2) 

cv.imwrite('Detected/Images/' + filename, img) # save the image with detected face

cv.waitKey(0)
cv.destroyAllWindows()