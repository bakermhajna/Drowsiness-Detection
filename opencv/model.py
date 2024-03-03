import cv2
import urllib.request as urlreq
import os
import pygame
from scipy.spatial import distance
from imutils import face_utils
import threading

def loadfacedetectionmodel():
    haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    haarcascadePath = "Drowsiness-Detection\\opencv\\assets\\haarcascade_frontalface_alt2.xml"
    if (os.path.exists(haarcascadePath)):
        print("File exists")
    else:
        urlreq.urlretrieve(haarcascade_url, haarcascadePath)
        print("File downloaded")
    detector = cv2.CascadeClassifier(haarcascadePath)
    return detector.detectMultiScale


def loadfacelandmarkmodel():
    LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
    LBFmodelPath = "Drowsiness-Detection\\opencv\\assets\\LFBmodel.yaml"
    if (os.path.exists(LBFmodelPath)):
        print("File exists")
    else:
        urlreq.urlretrieve(LBFmodel_url, LBFmodelPath)
        print("File downloaded")
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodelPath)
    return landmark_detector.fit

def init_sound():
    pygame.mixer.init()
    pygame.mixer.music.load('Drowsiness-Detection\\opencv\\assets\\emergency-alarm.mp3')  # Replace with the path to your sound file

def play_sound():
    pygame.mixer.music.play()

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	

if __name__ == "__main__":
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    thresh = 0.19
    frame_check = 10
    counter=0
    sound_thread_started=False
    init_sound()
    
    detector=loadfacedetectionmodel()
    landmarksmodel=loadfacelandmarkmodel()
    cap=cv2.VideoCapture(0) 
    while True:
        ret, frame=cap.read()
        # frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=detector(gray)
        landmarks=[]
        try:
            landmarks=landmarksmodel(gray,faces)
        except:
            print("there is no face")
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue
        leftEye = landmarks[1][0][0][lStart:lEnd]
        rightEye = landmarks[1][0][0][rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        print(ear)
        # leftEyeHull = leftEye.astype(int)
        # rightEyeHull = rightEye.astype(int)
        # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if ear < thresh:
            counter += 1
            if counter == frame_check:
                
                sound_thread = threading.Thread(target=play_sound)
                sound_thread.start()
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10,325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                
        else:
            counter = 0
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    cap.release() 





