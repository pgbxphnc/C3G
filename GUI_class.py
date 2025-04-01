from Yolo_GUI_UI import Ui_Form_Yolo
from PyQt6.QtWidgets import *
from GUI_function import *

class MyWindow(Ui_Form_Yolo, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("YOLO")
        self.setFixedSize(800, 800)

        self.myinit()
        self.link_my_class()
        self.link_my_button()

    def myinit(self):
        self.current_model = None
        self.train_dialog = None

        self.current_image = None

        self.current_video = None

    def link_my_class(self):
        self.Model = Model(self)
        self.Image = Image(self)
        self.Video = Video(self)

    def link_my_button(self):
        self.pushButton_model_select.clicked.connect(self.Model.select_a_model)
        self.pushButton_model_train.clicked.connect(self.Model.train_a_model)

        self.pushButton_image_select.clicked.connect(self.Image.select_a_image)
        self.pushButton_image_detection.clicked.connect(self.Image.start_detection)

        self.pushButton_video_select.clicked.connect(self.Video.select_a_video)
        self.pushButton_video_select_webcam.clicked.connect(self.Video.select_the_webcam)
        self.pushButton_video_detection.clicked.connect(self.Video.start_detection)
        self.pushButton_video_stop.clicked.connect(self.Video.stop_detection)


