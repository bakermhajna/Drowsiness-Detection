import cv2
import urllib.request as urlreq
import os
import matplotlib.pyplot as plt
import pygame
from scipy.spatial import distance
from imutils import face_utils
import imutils



def getimg():
    # save picture's url in pics_url variable
    pics_url = "https://upload.wikimedia.org/wikipedia/commons/2/2a/Human_faces.jpg"

    # save picture's name as pic
    pic = "image.jpg"

    # chech if picture is in working directory
    if (pic in os.listdir(os.curdir)):
        print("Picture exists")
    else:
        # download picture from url and save locally as image.jpg
        urlreq.urlretrieve(pics_url, pic)
        print("Picture downloaded")
    
    return pic

def imgcoloring(frame):
    image = cv2.imread(frame)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

def facedetection(image):
        # save face detection algorithm's url in haarcascade_url variable
    haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

    # save face detection algorithm's name as haarcascade
    haarcascade = "haarcascade_frontalface_alt2.xml"

    # chech if file is in working directory
    if (haarcascade in os.listdir(os.curdir)):
        print("File exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml
        urlreq.urlretrieve(haarcascade_url, haarcascade)
        print("File downloaded")

        # create an instance of the Face Detection Cascade Classifier
    detector = cv2.CascadeClassifier(haarcascade)
    
    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces=detector.detectMultiScale(image)
    facelandmark(faces,image)
    
    return faces

def facelandmark(faces,image_gray):
        # save facial landmark detection model's url in LBFmodel_url variable
    LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

    # save facial landmark detection model's name as LBFmodel
    LBFmodel = "LFBmodel.yaml"

    # check if file is in working directory
    if (LBFmodel in os.listdir(os.curdir)):
        print("File exists")
    else:
        # download picture from url and save locally as lbfmodel.yaml
        urlreq.urlretrieve(LBFmodel_url, LBFmodel)
        print("File downloaded")
    
        # create an instance of the Facial landmark Detector with the model
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel)

    # Detect landmarks on "image_gray"
    _, landmarks = landmark_detector.fit(image_gray, faces)
    return landmarks
    for landmark in landmarks:
        for x,y in landmark[0]:
            # display landmarks on "image_cropped"
            # with white colour in BGR and thickness 1
            cv2.circle(image_gray,(x,y),1,(255,255,255),1)
    # plt.axis("off")
    plt.imshow(image_gray)
    plt.show()
    # print coordinates of detected landmarks
    print("landmarks LBF\n", landmarks)


# pic=getimg()
image=imgcoloring("C:\\Users\\kha12\\Desktop\\test2\\image.jpg")
faces=facedetection(image)


