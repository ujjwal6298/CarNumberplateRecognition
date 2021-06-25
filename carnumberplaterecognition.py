import cv2
import imutils
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img=cv2.imread(r'C:\Users\Utsav\Downloads\car3.jpg',cv2.IMREAD_COLOR)
# we can also use cv2.VideoCapture(0) is for opening system webcamera
# cv2.videocapture(1) is for any portable camera
#resizing image
img=cv2.resize(img,(600,400))
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#converting it into gray scale image
gray=cv2.bilateralFilter(gray,13,15,15)  #blurring the image

edged=cv2.Canny(gray,30,200)  #perform edge detection
contours=cv2.findContours(edged.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours=imutils.grab_contours(contours)
contours=sorted(contours, key=cv2.contourArea, reverse= True)[:10]
screenCnt= None

for c in contours:
    #approximate the contour
    peri=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.018*peri,True)
    #if our approximated contour has four points, then we can assume that
    # we have found our screen
    if len(approx)==4:
        screenCnt=approx
        break

if screenCnt is None:
    detected=0
    print("No contour detected")
else:
    detected=1


if detected==1:
    cv2.drawContours(img,[screenCnt],-1,(0,0,255),3)

#masking the part other than the number plate
mask=np.zeros(gray.shape,np.uint8)
new_image=cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image=cv2.bitwise_and(img,img,mask=mask)

# now crop
(x,y)=np.where(mask==255)
(topx,topy)=(np.min(x),np.min(y))
(bottomx,bottomy)=(np.max(x),np.max(y))
Cropped=gray[topx:bottomx+1,topy:bottomy+1]

# read the number plate
text=pytesseract.image_to_string(Cropped,config='--psm 11')
print("geeky_ujjwal's Licence plate Recognition\n")
print("Detected Licence plate number is:",text)
img=cv2.resize(img,(500,300))
Cropped=cv2.resize(Cropped,(400,200))
cv2.imshow('car',img)
cv2.imshow('Cropped',Cropped)


cv2.waitKey(0)
cv2.destroyAllWindows()







