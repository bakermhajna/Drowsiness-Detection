from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np
import urllib.request as urlreq
import os
from pygame import mixer
from scipy.spatial import distance
from imutils import face_utils
import threading



Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
    Image:
        id: kivy_image
    ToggleButton:
        text: 'Play'
        on_press: root.toggle_camera()
        size_hint_y: None
        height: '48dp'
    Button:
        text: 'Capture'
        size_hint_y: None
        height: '48dp'
        on_press: root.capture_Image()
''')

class CameraClick(BoxLayout):
    def __init__(self, **kwargs):
        super(CameraClick, self).__init__(**kwargs)
        self.capture = None
        self.is_camera_playing = False
        self.kivy_image = self.ids.kivy_image
        self.texture = None
        self.model=Model()

    def toggle_camera(self):
        if self.is_camera_playing:
            self.capture.release()
            self.is_camera_playing = False
        else:
            self.capture = cv2.VideoCapture(0)  # Use 0 for default camera, you can change it accordingly
            self.is_camera_playing = True
            Clock.schedule_interval(self.update_image, 1 / 30.0)  # Update the image every 1/30 seconds

    def update_image(self, dt):
        if self.is_camera_playing:
            ret, frame = self.capture.read()
            if ret:
                self.model.model_function(frame)
                self.texture = self.convert_frame_to_texture(frame)
                self.kivy_image.texture = self.texture

    def capture_Image(self):
        if self.is_camera_playing:
            # Perform additional capture logic if needed
            pass

    def convert_frame_to_texture(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip the image vertically (OpenCV captures in reverse order)
        frame_rgb = np.flipud(frame_rgb)

        # Convert to a Kivy texture
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(frame_rgb.tostring(), colorfmt='rgb', bufferfmt='ubyte')

        return texture

class TestCamera(App):
    def build(self):
        return CameraClick()


class Model:
    def __init__(self):
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]   #מחזיר שתי נקודות התחלתית וסופית לעין שמאל 
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        self.thresh = 0.19  #
        self.frame_check = 10  # בדיקת frames
        self.counter=0  # ספירת תמונות שנסגרה העין
        self.init_sound() # אתחול קובץ אודיו
        self.detector=self.loadfacedetectionmodel()
        self.landmarksmodel=self.loadfacelandmarkmodel()
    
    
    def model_function(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# מחזיר תמונה שחור לבן
        faces=self.detector(gray) #מחזיר מערך בגודל 4
        landmarks=[]
        try:
            landmarks=self.landmarksmodel(gray,faces)
        except:
            print("there is no face")
            return
        leftEye = landmarks[1][0][0][self.lStart:self.lEnd] # מחזירה 6 נקוקות של העין
        rightEye = landmarks[1][0][0][self.rStart:self.rEnd] # arr =[1,2,3,4,5,6,7,8,9]  newarr=arr[2:4]  = [3,4,5]
        leftEAR = self.eye_aspect_ratio(leftEye) # מחזיר יחס רוחב לגובה לעין
        rightEAR = self.eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        print(ear)
        if ear < self.thresh:
            self.counter += 1
            if self.counter == self.frame_check:
                sound_thread = threading.Thread(target=self.play_sound)
                sound_thread.start()
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10,325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)        
        else:
            self.counter = 0
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF


    def loadfacedetectionmodel(self):
        haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
        haarcascadePath = "opencv\\assets\\haarcascade_frontalface_alt2.xml"
        if (os.path.exists(haarcascadePath)):
            print("File exists")
        else:
            urlreq.urlretrieve(haarcascade_url, haarcascadePath)
            print("File downloaded")
        detector = cv2.CascadeClassifier(haarcascadePath)
        return detector.detectMultiScale


    def loadfacelandmarkmodel(self):
        LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
        LBFmodelPath = "opencv\\assets\\LFBmodel.yaml"
        if (os.path.exists(LBFmodelPath)):
            print("File exists")
        else:
            urlreq.urlretrieve(LBFmodel_url, LBFmodelPath)
            print("File downloaded")
        landmark_detector  = cv2.face.createFacemarkLBF()
        landmark_detector.loadModel(LBFmodelPath)
        return landmark_detector.fit

    def init_sound(self):
        mixer.init()
        mixer.music.load('opencv\\assets\\emergencyAlarm.mp3')  # Replace with the path to your sound file

    def play_sound(self):
        mixer.music.play()

    def eye_aspect_ratio(self,eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear


if __name__ == '__main__':
    TestCamera().run()
