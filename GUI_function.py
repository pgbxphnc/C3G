from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

from train_dialog import Ui_Dialog_model_train
import os
import shutil
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np


class Model():
    def __init__(self, MyWindow):
        self.MyWindow = MyWindow
        self.pre_load_model()

    def pre_load_model(self):
        # 预加载模型
        model_path = "./test/model/yolo12.pt"
        if os.path.exists(model_path):
            self.MyWindow.current_model = YOLO(model_path)
            self.MyWindow.label_model_current_model.setText(f"Current Model: {model_path}")
        else:
            self.MyWindow.current_model = None
            self.MyWindow.label_model_current_model.setText("No Model Loaded")


    def select_a_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.MyWindow, "Select an existing Model", "./", "Model File (*.pt)"
        )
        if file_path:
            normalized_path = os.path.normpath(file_path)
            self.MyWindow.current_model = YOLO(normalized_path)
            self.MyWindow.label_model_current_model.setText(f"Current Model: {normalized_path}")

    def train_a_model(self):
        if self.MyWindow.train_dialog == None:

            self.MyWindow.train_dialog = QDialog(self.MyWindow)
            ui = Ui_Dialog_model_train()
            ui.setupUi(self.MyWindow.train_dialog)

        self.MyWindow.train_dialog.setWindowTitle("Train a Model")
        self.MyWindow.train_dialog.exec()




class Image():
    def __init__(self, MyWindow):
        self.MyWindow = MyWindow

    def select_a_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.MyWindow, "Select an Image", "./", "Image File (*.png *.jpg *.jpeg)"
        )
        if file_path:
            normalized_path = os.path.normpath(file_path)
            self.MyWindow.label_image_filepath.setText(f"Current Image: {normalized_path}")

            suffix = file_path.split(".")[-1]
            save_path = os.path.join("./test/tmp", "tmp_image." + suffix)
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)  # 如果不存在，则创建该文件夹
            # 将图像转移到images目录下并且修改为英文的形式
            shutil.copy(file_path, save_path)
            self.MyWindow.current_image = normalized_path

            current_image = cv2.imread(save_path)
            traget_size = [self.MyWindow.label_image_original.width(), self.MyWindow.label_image_original.height()]
            current_image = cv2.resize(current_image, (traget_size[0], traget_size[1]))
            resize_image_path = "./test/tmp/tmp_image_resize.jpg"
            # self.MyWindow.current_image = resize_image_path# 将图像进行缩放
            cv2.imwrite(resize_image_path, current_image)                             # 给变量进行赋值方便后面实际进行读取
            # 将图像显示在界面上并将预测的文字内容进行初始化
            self.MyWindow.label_image_original.setPixmap(QPixmap(resize_image_path))

    def start_detection(self):
        try:
            # 检查是否选择了图片和模型
            if self.MyWindow.current_image is None:
                raise ValueError("Please select an image first.")
            if self.MyWindow.current_model is None:
                raise ValueError("Please select a model first.")
        except ValueError as e:
            # 捕获特定的 ValueError 异常，并显示相关的错误信息
            print(f"Error: {e}")
            # 可以使用 QMessageBox 或其他方式在界面上提示用户
            QMessageBox.warning(self.MyWindow, "Input Error", str(e))
            return

        traget_size = [self.MyWindow.label_image_predection.width(), self.MyWindow.label_image_predection.height()]
        results = self.MyWindow.current_model(self.MyWindow.current_image)  # 读取图像并执行检测的逻辑
        result = results[0]                     # 获取检测结果
        image_array = result.plot()               # 在图像上绘制检测结果
        detection_image = image_array
        detection_image = cv2.resize(detection_image, (traget_size[0], traget_size[1]))
        save_path = "./test/tmp/tmp_image_detection.jpg"
        cv2.imwrite(save_path, detection_image)
        self.MyWindow.label_image_predection.setPixmap(QPixmap(save_path))
        QMessageBox.information(self.MyWindow, "Success", "Successfully detected the image!", QMessageBox.StandardButton.Ok)











class Video():
    def __init__(self, MyWindow):
        self.MyWindow = MyWindow
        self.cap = None
        self.timer = None

    def select_a_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.MyWindow, "Select an Image", "./", "Video File (*.mp4 *.avi)"
        )
        if file_path:
            normalized_path = os.path.normpath(file_path)
            self.MyWindow.label_video_filepath.setText(f"Current Video: {normalized_path}")

            # suffix = file_path.split(".")[-1]
            # save_path = os.path.join("./test/tmp", "tmp_video." + suffix)
            # save_dir = os.path.dirname(save_path)
            #
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)  # 如果不存在，则创建该文件夹
            # # 将图像转移到images目录下并且修改为英文的形式
            # shutil.copy(file_path, save_path)

            self.MyWindow.current_video = normalized_path

            traget_size = [self.MyWindow.label_video_original.width(), self.MyWindow.label_video_original.height()]
            cap = cv2.VideoCapture(normalized_path)
            if not cap.isOpened():
                print("Error: Cannot open video file.")
                return

            # 读取视频的第一帧
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read video frame.")
                return
            current_video_page = frame
            # 释放视频捕获对象
            cap.release()

            current_video_page = cv2.resize(current_video_page, (traget_size[0], traget_size[1]))
            resize_video_path = "./test/tmp/tmp_video_page_resize.jpg"
            # self.MyWindow.current_image = resize_image_path# 将图像进行缩放
            cv2.imwrite(resize_video_path, current_video_page)  # 给变量进行赋值方便后面实际进行读取
            # 将图像显示在界面上并将预测的文字内容进行初始化
            self.MyWindow.label_video_original.setPixmap(QPixmap(resize_video_path))

    def select_the_webcam(self):
        self.MyWindow.current_video = 0
        self.MyWindow.label_video_filepath.setText(f"Current Video: Webcam")
        self.MyWindow.label_video_original.setText("Webcam selected.")

    def start_detection(self):
        self.cap = cv2.VideoCapture(self.MyWindow.current_video)
        self.track_history = defaultdict(lambda: [])

        if self.cap is None or not self.cap.isOpened():
            print("Error: Video not selected or cannot open video.")
            return

        # 启动定时器，每隔一定时间获取一帧
        self.timer = QTimer(self.MyWindow)
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(30)  # 每30ms读取一帧

    def process_frame(self):
        # 读取下一帧
        ret, frame = self.cap.read()
        if not ret:
            print("Video ended or failed to read frame.")
            self.timer.stop()  # 停止定时器
            self.cap.release()
            return

        # 在这里执行目标检测（假设你已经加载了YOLO模型）
        # 假设我们有一个YOLO模型对象 `self.model`，并调用 `self.model.track(frame)`
        results = self.MyWindow.current_model.track(frame, persist=True)

        boxes = results[0].boxes.xywh.cpu()
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
        else:
            track_ids = []
            class_ids = []

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks for 'person' class (class_id == 0)
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            if class_id == 0:  # Only track 'person' class
                x, y, w, h = box
                track = self.track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    annotated_frame,
                    [points],
                    isClosed=False,
                    color=(0, 0, 255),
                    thickness=2,
                )

        # 转换为适合Qt显示的格式
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        target_size = [
            self.MyWindow.label_video_predection.width(),
            self.MyWindow.label_video_predection.height()
        ]
        q_image = q_image.scaled(target_size[0], target_size[1], Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


        # 更新 QLabel 显示检测结果
        pixmap = QPixmap.fromImage(q_image)
        self.MyWindow.label_video_predection.setPixmap(pixmap)


    def stop_detection(self):
        if self.timer:
            self.timer.stop()
        if self.cap:
            self.cap.release()
        self.MyWindow.label_video_predection.clear()
        print("Detection stopped.")