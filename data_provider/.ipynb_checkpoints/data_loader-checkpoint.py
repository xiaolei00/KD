import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2  # 用于读取图像和进行图像处理
import glob
# from ultralytics import YOLO
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# 自定义Dataset类，从磁盘读取图像并截取出人
class Dataset_Basketball(Dataset):
    def __init__(self, root_dir, transform_train=None, transform_val=None):
        self.root_dir = root_dir
        self.transform_train = transform_train
        self.transform_val = transform_val
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            return None  # or handle the error as needed
        person_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        person_image = Image.fromarray(person_image)
        '''
        # 使用目标检测模型去检测球员
        person_results = self.person_detect_model(image, classes=[0])
        person_result = person_results[0]
        person_boxes = person_result.boxes.cpu().numpy()
        person_clses = person_boxes.cls
        person_xyxys = person_boxes.xyxy
        person_confs = person_boxes.conf
        # 存储每个人的处理结果
        person_images = []
        # 遍历检测到的每个人
        for person_conf, person_xyxy in zip(person_confs, person_xyxys):
            if person_conf >= 0.4:
                x_min, y_min, x_max, y_max = person_xyxy
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                # 截取人区域
                person_image = image[y_min:y_max, x_min:x_max]
                # 转换为PIL图像，适应后续预处理
                person_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
                person_image = Image.fromarray(person_image)
                # 归一化和尺寸调整
                if self.transform:
                    person_image = self.transform(person_image)
                # print(f'person_image.shape: {person_image.shape}')
                # 将处理好的图像存入列表
                person_images.append(person_image)
                # print(type(person_image))
        try:
            person_images_tensor = torch.stack(person_images)
        except RuntimeError as e:
            print(f'error img_path: {img_path}')
        # print(f'person_images_tensor: {person_images_tensor.shape}')
        # 返回处理后的所有人的图像，作为一个列表
        return person_images_tensor
        '''
        person_image_augment = self.transform_train(person_image)
        person_image_norm = self.transform_val(person_image)
        return person_image_augment, person_image_norm

