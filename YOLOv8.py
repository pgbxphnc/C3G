import os
import zipfile
from tqdm import tqdm
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

def download_market1501():
    """
    下载Market-1501数据集
    """
    extracted_path = "Market-1501-v15.09.15"
    
    if os.path.exists(extracted_path):
        print("数据集已存在，跳过下载")
        return
    
    # 使用Kaggle API下载数据集
    print("正在通过Kaggle下载Market-1501数据集...")
    try:
        os.system("kaggle datasets download -d pengcw1/market-1501 -p .")
        
        target_path = "Market-1501-v15.09.15.zip"
        
        print("正在解压数据集...")
        with zipfile.ZipFile(target_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # 删除压缩包
        os.remove(target_path)
        print("数据集准备完成！")
        
    except Exception as e:
        print(f"下载失败: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)
        return

def detect_person():
    """
    使用YOLOv8进行行人检测
    """
    # 加载YOLOv8模型
    model = YOLO('yolov8x.pt')  # 使用最大的模型以获得最好的性能
    
    # 读取Market-1501数据集中的图片
    dataset_path = "Market-1501-v15.09.15/bounding_box_test"
    output_path = "detection_results"
    os.makedirs(output_path, exist_ok=True)
    
    # 设置检测参数
    conf_threshold = 0.5  # 置信度阈值
    
    print("开始行人检测...")
    for img_name in tqdm(os.listdir(dataset_path)):
        if not img_name.endswith(('.jpg', '.png')):
            continue
            
        img_path = os.path.join(dataset_path, img_name)
        
        # 进行检测
        results = model(img_path, conf=conf_threshold, classes=0)  # 只检测人类(class 0)
        
        # 获取原始图片
        img = cv2.imread(img_path)
        
        # 处理检测结果
        for result in results:
            boxes = result.boxes  # 获取检测框
            for box in boxes:
                # 获取坐标和置信度
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # 绘制边界框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 添加置信度标签
                label = f'Person {conf:.2f}'
                cv2.putText(img, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存结果
        cv2.imwrite(os.path.join(output_path, f'det_{img_name}'), img)

def analyze_results(output_path):
    """
    分析检测结果
    """
    total_detections = 0
    total_images = 0
    
    for img_name in os.listdir(output_path):
        if not img_name.startswith('det_'):
            continue
            
        img = cv2.imread(os.path.join(output_path, img_name))
        # 统计绿色边界框的数量（简单方法）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_detections += len(contours)
        total_images += 1
    
    print(f"\n检测结果分析:")
    print(f"处理图片总数: {total_images}")
    print(f"检测到的行人总数: {total_detections}")
    print(f"平均每张图片检测到的行人数: {total_detections/total_images:.2f}")

if __name__ == "__main__":
    # 安装必要的包
    os.system("pip install ultralytics kaggle")
    
    # 下载数据集
    download_market1501()
    
    # 进行行人检测
    detect_person()
    
    # 分析结果
    analyze_results("detection_results")
    
    print("处理完成！结果保存在 detection_results 文件夹中")