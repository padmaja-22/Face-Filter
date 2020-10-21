import cv2
import numpy as np
import dlib

webcam=True
cap=cv2.VideoCapture(0)

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def empty(a):
    pass

cv2.namedWindow("BGR")
cv2.resizeWindow("BGR",640,340)
cv2.createTrackbar("Blue","BGR",0,255,empty)
cv2.createTrackbar("Green","BGR",0,255,empty)
cv2.createTrackbar("Red","BGR",0,255,empty)



def createBound(img,points,scale=5,masked=False,cropped=True):
    if masked:
        mask=np.zeros_like(img)
        mask=cv2.fillPoly(mask,[points],(255,255,255))

        img=cv2.bitwise_and(img,mask) #adding img and mask
        #cv2.imshow("Mask",img)
    if cropped:
        bounding_box=cv2.boundingRect(points)
        x,y,w,h=bounding_box
        imgCrop=img[y:y+h,x:x+w]
        imgCrop=cv2.resize(imgCrop,(0,0),None,scale,scale)
        return imgCrop
    else:
        return mask
while True:
    if webcam: success,img=cap.read()
    else:img = cv2.imread('test.jpg')
    img=cv2.resize(img,(0,0),None,0.5,0.5)
    imgNew=img.copy()


    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detector(imgGray)

    for face in faces:
        x1,y1=face.left(),face.top()
        x2,y2=face.right(),face.bottom()
        #imgNew=cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
        landmarks=predictor(imgGray,face)
        myPoints=[]
        for n in range(68):
            x=landmarks.part(n).x
            y=landmarks.part(n).y
            #cv2.circle(imgNew,(x,y),2,(50,50,255),cv2.FILLED)
            #cv2.putText(imgNew,str(n),(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.4,(50,50,255),1)
            myPoints.append([x,y])
        myPoints=np.array(myPoints)
        #leftEye=createBound(img,myPoints[36:42])
        #cv2.imshow('Left Eye',leftEye)
        lips=createBound(img,myPoints[48:62],3,masked=True,cropped=False)
        imgcolorlips=np.zeros_like(lips)  #gives a black image
        b=cv2.getTrackbarPos("Blue","BGR")
        g=cv2.getTrackbarPos("Green","BGR")
        r=cv2.getTrackbarPos("Red","BGR")
        imgcolorlips[:]=b,g,r  #gives a purple color
        imgcolorlips=cv2.bitwise_and(imgcolorlips,lips)
        imgcolorlips=cv2.GaussianBlur(imgcolorlips,(7,7),10)
        imgNewGray=cv2.cvtColor(imgNew,cv2.COLOR_BGR2GRAY)
        imgNewGray=cv2.cvtColor(imgNewGray,cv2.COLOR_GRAY2BGR)
        imgcolorlips=cv2.addWeighted(imgNewGray,1,imgcolorlips,0.4,0)
        cv2.imshow("BGR",imgcolorlips )
        #cv2.imshow('Lips',lips)
    #cv2.imshow("Picture",imgNew)
    cv2.waitKey(1)
