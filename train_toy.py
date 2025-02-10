import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
import timm
from torchvision import transforms as Transforms
import torch.nn.functional as F
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import glob
import shutil
from ultralytics import YOLO
from PIL import Image
import time
from data_provider.data_factory import data_provider
import argparse
import torch.optim as optim
import torch.multiprocessing as mp
import math
from tqdm import tqdm

def create_directory_if_not_exists(directory):
    # 检查目录是否存在
    if not os.path.exists(directory):
        # 如果目录不存在，则创建目录
        os.makedirs(directory)
        print("目录 '{}' 创建成功".format(directory))
    else:
        print("目录 '{}' 已经存在".format(directory))

# 递归删除指定目录下的.ipynb_checkpoints文件夹
def remove_ipynb_checkpoints(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for dir in dirs:
            if dir == ".ipynb_checkpoints":
                folder_path = os.path.join(root, dir)
                shutil.rmtree(folder_path)
                print(f"Deleted: {folder_path}")

class SwinTransformerTeacher(nn.Module):

    def __init__(self, num_features=512):
        super(SwinTransformerTeacher, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224')
        self.num_features = num_features
        self.feat = nn.Linear(1024, num_features) if num_features > 0 else None

    def forward(self, x):
        x = self.model.forward_features(x)
        if not self.feat is None:
            x = self.feat(x)
        return x

class ResNetStudent(nn.Module):

    def __init__(self, num_features=512):
        super(ResNetStudent, self).__init__()
        self.model = timm.create_model('resnet50', pretrained=True)  # 使用ResNet-18作为学生模型
        self.flatten_dim = 2048 * 7 * 7
        self.feat = nn.Linear(self.flatten_dim, num_features) if num_features > 0 else None

    def forward(self, x):
        x = self.model.forward_features(x)
        # 展平特征图，将其变为 [batch_size, 2048 * 7 * 7]
        x = x.view(x.size(0), -1)  # 展平
        if not self.feat is None:
            x = self.feat(x)
        return x


class Data_Processor(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.transformer = Transforms.Compose([
            Transforms.Resize((self.height, self.width)),
            Transforms.ToTensor(),
            Transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        return self.transformer(img).unsqueeze(0)

# 定义蒸馏损失
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        self.l2_loss = nn.MSELoss()

    def forward(self, student_outputs, teacher_outputs, student_features, teacher_features, labels):
        # 计算KL散度损失
        kl_loss = self.kl_loss(
            nn.functional.log_softmax(student_outputs / self.temperature, dim=1),
            nn.functional.softmax(teacher_outputs / self.temperature, dim=1)
        ) * (self.temperature ** 2)

        # 计算L2损失
        l2_loss = self.l2_loss(student_features, teacher_features)

        # 计算交叉熵损失
        ce_loss = self.ce_loss(student_outputs, labels)

        # 总蒸馏损失
        loss = self.alpha * kl_loss + (1 - self.alpha) * (ce_loss + l2_loss)
        return loss

# 自监督蒸馏损失函数
class SelfSupervisedDistillationLoss(nn.Module):
    def __init__(self, temperature=0.5, alpha=0.5):
        super(SelfSupervisedDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.l2_loss = nn.MSELoss()

    def forward(self, student_features, teacher_features):
        # 对比损失：实例对比学习，使用余弦相似度作为度量
        student_features = F.normalize(student_features, dim=1)
        teacher_features = F.normalize(teacher_features, dim=1)
        
        # 计算学生和教师特征之间的对比损失
        logits = torch.mm(student_features, teacher_features.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(logits.device)
        
        contrastive_loss = F.cross_entropy(logits, labels)
        
        # 特征对齐损失
        alignment_loss = self.l2_loss(student_features, teacher_features)
        
        # 总自监督蒸馏损失
        loss = self.alpha * contrastive_loss + (1 - self.alpha) * alignment_loss
        return loss

# 定义特征匹配损失（L2距离）
def feature_distillation_loss(student_features, teacher_features):
    """ 
    student_features: 学生模型输出的特征
    teacher_features: 教师模型输出的特征
    计算学生与教师输出特征之间的L2距离
    """
    return F.mse_loss(student_features, teacher_features)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    mp.set_start_method('spawn', force=True)  # 设置 'spawn' 方法
    
    # 手动创建一个包含参数的命名空间
    args = argparse.Namespace()
    args.root_dir_train = "/homec/xiaolei/projects/ReID/datasets/train"
    args.root_dir_valid = "/homec/xiaolei/projects/ReID/datasets/valid"
    args.train_epochs = 100
    args.batch_size = 256
    args.num_workers = 10
    args.height = 224
    args.width = 224
    stage_train = "train"
    stage_valid = "valid"
    
    train_data_loader = data_provider(args, stage=stage_train)
    valid_data_loader = data_provider(args, stage=stage_valid)
    
    # 初始化教师模型和学生模型
    teacher_model = SwinTransformerTeacher(num_features=512).cuda()
    student_model = ResNetStudent(num_features=512).cuda()
    
    # 加载教师模型的预训练权重（假设教师模型已经训练好）
    teacher_weight_path = '/homec/xiaolei/projects/ISR/weights/swin_base_patch4_window7_224.pth'
    teacher_weight = torch.load(teacher_weight_path)
    teacher_model.load_state_dict(teacher_weight['state_dict'], strict=True)
    teacher_model.eval()  # 冻结教师模型
    
    # 定义优化器和损失
    distillation_loss = SelfSupervisedDistillationLoss(temperature=0.5, alpha=0.5)
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    
    # 假设已经定义了 dataloader，并且数据无标签
    epochs = args.train_epochs  # 设置训练的 epoch 数
    best_loss = math.inf
    path = "/homec/xiaolei/projects/ReID/weights/student_model"
    
    # 训练学生模型（无监督）
    for epoch in range(epochs):
        student_model.train()
        train_loss = []
        # 可视化进度条
        with tqdm(total=len(train_data_loader), desc=f"Epoch {epoch + 1}/{args.train_epochs}") as pbar:
            # 将数据送入模型进行训练
            for person_image_bs in train_data_loader:
                optimizer.zero_grad()
                if not isinstance(person_image_bs, torch.Tensor):
                    person_image_bs = torch.stack(person_image_bs)
                person_image_bs = person_image_bs.to(device)
                # print(type(person_image_bss))
                # 通过教师模型和学生模型获取输出特征
                with torch.no_grad():  # 教师模型保持冻结状态
                    teacher_features = teacher_model(person_image_bs)
        
                student_features = student_model(person_image_bs)
        
                # 计算特征匹配损失（L2距离）
                loss = distillation_loss(student_features, teacher_features)
                train_loss.append(loss.item())
        
                # 反向传播并更新学生模型参数
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                # 更新进度条
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)
            
            train_loss_avg = np.average(train_loss)
            torch.save(student_model.state_dict(), path + '/' + f'checkpoint_{epoch}_{train_loss_avg}.pth')
        # 验证阶段
        student_model.eval()
        val_loss = []
        with tqdm(total=len(valid_data_loader), desc=f"Epoch {epoch + 1}/{args.train_epochs}") as pbar:
            with torch.no_grad():
                 for person_image_bs in valid_data_loader:
                    if not isinstance(person_image_bs, torch.Tensor):
                        person_image_bs = torch.stack(person_image_bs)
                    person_image_bs = person_image_bs.to(device)
                    teacher_features = teacher_model(person_image_bs)
                    student_features = student_model(person_image_bs)
                    loss = distillation_loss(student_features, teacher_features)
                    val_loss.append(loss.item())
    
                    # 更新进度条
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update(1)
    
            val_loss_avg = np.average(val_loss)  # 计算平均验证损失
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss}')
    
        # 保存最优的学生模型
        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            temp_best_model_path = os.path.join("/homec/xiaolei/projects/ReID/weights/student_model", "best_student_model.pth")
            torch.save(student_model.state_dict(), temp_best_model_path)
            print(f'Best model saved with Validation Loss: {best_loss}')
    
    # 保存训练好的学生模型
    best_model_path = os.path.join("/homec/xiaolei/projects/ReID/weights/student_model", "student_model.pth")
    torch.save(student_model.state_dict(), best_model_path)