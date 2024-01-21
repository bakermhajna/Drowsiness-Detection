from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np

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

    def toggle_camera(self):
        if self.is_camera_playing:
            self.capture.release()
            self.is_camera_playing = False
        else:
            self.capture = cv2.VideoCapture(1)  # Use 0 for default camera, you can change it accordingly
            self.is_camera_playing = True
            Clock.schedule_interval(self.update_image, 1 / 30.0)  # Update the image every 1/30 seconds

    def update_image(self, dt):
        if self.is_camera_playing:
            ret, frame = self.capture.read()
            if ret:
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

if __name__ == '__main__':
    TestCamera().run()
