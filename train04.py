import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

from config import get_config
from optimizer import build_optimizer
from lr_scheduler import build_scheduler

from models.dis_losses import KDLoss, FreqMaskingDistillLossv2

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

    def extract_feat(self, x):
        # 创建一个空列表，用于保存各层的输出特征
        features = []
        
        patch_embed = self.model.patch_embed  # Patch Embedding 层
        pos_drop = self.model.pos_drop
        layers = self.model.layers  # 基本层（包含多个 SwinBlock）
        
        x = patch_embed(x)  # Patch Embedding
        x = pos_drop(x)
        for layer in layers:  # 逐个通过 BasicLayer
            # x = layer(x)
            # features.append(x)
            for block in layer.blocks:
                x = block(x)
            features.append(x)
            if layer.downsample is not None:
                x = layer.downsample(x)
        return tuple(features)

    def forward_specific_stage(self, x, stage, down_sample=True):
        BS, L, C = x.shape

        if stage == 2:
            if down_sample:
                x = self.model.layers[-4].downsample(x)

            for block in self.model.layers[-3].blocks:
                x = block(x)

        if stage == 3:
            if down_sample:
                x = self.model.layers[-3].downsample(x)

            for block in self.model.layers[-2].blocks:
                x = block(x)

        if stage == 4:
            if down_sample:
                x = self.model.layers[-2].downsample(x)

            for block in self.model.layers[-1].blocks:
                x = block(x)

            norm_layer = self.model.norm
            x = norm_layer(x)

        return x
        
    def forward_features(self, x):
        x = self.model.forward_features(x)
        return x

    def forward(self, x):
        x = self.model.forward_features(x)
        if not self.feat is None:
            x = self.feat(x)
        return x

class ResNetStudent(nn.Module):
    def __init__(self, num_features=512):
        super(ResNetStudent, self).__init__()
        self.model = timm.create_model('resnet50', pretrained=True)  # 使用ResNet-50作为学生模型
        # 修改 layer1 和 layer2，向每个残差块中的 ReLU 前加上 InstanceNorm2d
        self._modify_layer(self.model.layer1)
        self._modify_layer(self.model.layer2)
        self._modify_layer_stride(self.model.layer4[0].conv2, self.model.layer4[0].downsample[0])
        # 输出设置
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_dim = 2048
        self.feat = nn.Linear(self.flatten_dim, num_features) if num_features > 0 else None
    
    def extract_feat(self, x):
        # 创建一个空列表，用于保存各层的输出特征
        features = []
        
        # 提取每个阶段的层
        conv1 = self.model.conv1  # 初始卷积层
        bn1 = self.model.bn1
        act1 = self.model.act1
        maxpool = self.model.maxpool
        layer1 = self.model.layer1  # 第一阶段（残差块1）
        layer2 = self.model.layer2  # 第二阶段（残差块2）
        layer3 = self.model.layer3  # 第三阶段（残差块3）
        layer4 = self.model.layer4  # 第四阶段（残差块4）
        
        x = conv1(x)
        x = bn1(x)
        x = act1(x)
        x = maxpool(x)
        stage1_out = layer1(x)  # 第一阶段的输出
        features.append(stage1_out)
        stage2_out = layer2(stage1_out)  # 第二阶段的输出
        features.append(stage2_out)
        stage3_out = layer3(stage2_out)  # 第三阶段的输出
        features.append(stage3_out)
        stage4_out = layer4(stage3_out)  # 第四阶段的输出
        features.append(stage4_out)
        return tuple(features)

    def _modify_layer(self, layer):
        """
        在每个残差块中的 ReLU 前加上 InstanceNorm2d 操作。
        """
        for block in layer:
            # 修改 conv1 和 conv2 之后的 ReLU，将 InstanceNorm2d 放在 ReLU 前面
            # 对于每个残差块，将 InstanceNorm2d 加入到 ReLU 之前
            block.act3 = nn.Sequential(
                nn.InstanceNorm2d(block.conv3.out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
            
    def _modify_layer_stride(self, last_layer, last_layer_downsample):
        # 在最后一层将stride改为1
        last_layer.stride = (1, 1)
        last_layer_downsample.stride = (1, 1)
        
    def forward_features(self, x):
        x = self.model.forward_features(x)
        return x

    def forward(self, x):
        x = self.model.forward_features(x)
        # 池化操作，[batch_size, 2048, 7, 7] -> [batch_size, 2048, 1, 1]
        x = self.gap(x)
        # 展平特征图，将其变为 [batch_size, 2048 * 1 * 1]
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

def cosine_similarity_loss(student_features, teacher_features, alpha=0.5):
    """
    计算余弦相似度损失，主要用于ReID任务中对齐特征。
    
    :param student_features: 学生模型的特征 (B, 512)
    :param teacher_features: 教师模型的特征 (B, 512)
    :param alpha: 蒸馏损失的权重，通常在 0-1 之间
    :return: 损失值
    """
    # 归一化特征向量
    student_features = F.normalize(student_features, p=2, dim=1)
    teacher_features = F.normalize(teacher_features, p=2, dim=1)
    
    # 计算余弦相似度
    cosine_similarity = F.cosine_similarity(student_features, teacher_features)
    
    # 损失为 1 - cosine_similarity，越接近1，表示相似度越高，损失越低
    loss = 1 - cosine_similarity.mean()
    
    return loss * alpha

def rbf_kernel(x, y, sigma=1.0):
    """
    计算高斯 RBF 核函数
    :param x: 输入张量 x (batch_size, feature_dim)
    :param y: 输入张量 y (batch_size, feature_dim)
    :param sigma: 核函数的宽度，控制相似度的范围
    :return: 计算得到的 RBF 核
    """
    # 计算样本之间的平方欧几里得距离
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    dist = xx + yy.t() - 2 * torch.matmul(x, y.t())
    
    # 计算 RBF 核（高斯核）
    return torch.exp(-dist / (2 * sigma ** 2))

def mmd_loss(X, Y, sigma=1.0):
    """
    计算最大均值差异（MMD）损失
    :param X: 样本集 X (batch_size_1, feature_dim)
    :param Y: 样本集 Y (batch_size_2, feature_dim)
    :param sigma: 核函数的宽度，控制相似度的范围
    :return: MMD 损失
    """
    # 计算 RBF 核
    XX = rbf_kernel(X, X, sigma)  # X 中样本对之间的核
    YY = rbf_kernel(Y, Y, sigma)  # Y 中样本对之间的核
    XY = rbf_kernel(X, Y, sigma)  # X 和 Y 中样本对之间的核

    # 计算 MMD 损失
    loss = XX.mean() + YY.mean() - 2 * XY.mean()
    
    return loss

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    mp.set_start_method('spawn', force=True)  # 设置 'spawn' 方法
    
    # 手动创建一个包含参数的命名空间
    args = argparse.Namespace()
    args.root_dir_train = "/homec/xiaolei/projects/ReID/datasets/basketball_player_fusion/train"
    args.root_dir_valid = "/homec/xiaolei/projects/ReID/datasets/basketball_player_fusion/valid"
    args.train_epochs = 200
    args.batch_size = 256
    args.num_workers = 10
    args.height = 224
    args.width = 224
    args.resume = False
    config = get_config(args)
    stage_train = "train"
    stage_valid = "valid"
    
    remove_ipynb_checkpoints(args.root_dir_train)
    remove_ipynb_checkpoints(args.root_dir_valid)
    
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
    for param in teacher_model.parameters():
        param.requires_grad = False
    if args.resume:
        student_model_weight_path = 'weights/student_model_base5_strong_reid_mmd_mse_loss_confusion/best_student_model.pth'
        student_model_weight = torch.load(student_model_weight_path)
        student_model.load_state_dict(student_model_weight, strict=True)
    
    s_loss = dict()
    # 查看模型可使用的函数
    # dir(student_model)

    # optimizer = optim.Adam(student_model.parameters(), lr=1e4)
    optimizer = build_optimizer(config, student_model)
    mse_loss = nn.MSELoss()
    # 学习率优化器
    lr_scheduler = build_scheduler(config, optimizer, len(train_data_loader))
    scaler = torch.cuda.amp.GradScaler()
    
    # 假设已经定义了 dataloader，并且数据无标签
    epochs = args.train_epochs  # 设置训练的 epoch 数
    best_loss = math.inf
    path = "/homec/xiaolei/projects/ReID/weights/student_model_base5_strong_reid_mmd_mse_loss_confusion"
    create_directory_if_not_exists(path)
    
    # 训练学生模型（无监督）
    for epoch in range(0, epochs):
        student_model.train() 
        optimizer.zero_grad()
        num_steps = len(train_data_loader)
        train_loss = []
        
        # 可视化进度条
        with tqdm(total=len(train_data_loader), desc=f"Epoch {epoch + 1}/{args.train_epochs}") as pbar:
            # 将数据送入模型进行训练
            for idx, person_image_bs in enumerate(train_data_loader):
                optimizer.zero_grad()
                # person_image_bs = torch.cat((person_image_bs[0], person_image_bs[1]), dim=0)
                if not isinstance(person_image_bs, torch.Tensor):
                    person_image_bs = torch.stack(person_image_bs)
                # print(f'person_image_bs: {person_image_bs.shape}')
                # continue
                person_image_bs = person_image_bs.to(device)
                # print(type(person_image_bss))
                # 通过教师print(f'x: {x.shape}')模型和学生模型获取输出特征
                with torch.cuda.amp.autocast():
                    with torch.no_grad():  # 教师模型保持冻结状态
                        teacher_output = teacher_model(person_image_bs)
    
                    student_output = student_model(person_image_bs)
                    
                    s_loss['ori_loss'] = cosine_similarity_loss(student_output, teacher_output, 1.0)
                    s_loss['mmd_loss'] = mmd_loss(student_output, teacher_output)
                    s_loss['mse_loss'] = mse_loss(student_output, teacher_output)
    
                    loss = s_loss['ori_loss'] + s_loss['mmd_loss'] + s_loss['mse_loss']
                    train_loss.append(loss.item())
        
                # 反向传播并更新学生模型参数
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step_update(epoch * num_steps + idx)
    
                # 更新进度条
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)
            
            train_loss_avg = np.average(train_loss)
            # 每10个epoch保存模型的权重
            if epoch % 10 == 0:
                torch.save(student_model.state_dict(), path + '/' + f'checkpoint_{epoch}_{train_loss_avg}.pth')
        
        # 验证阶段
        student_model.eval()
        val_output_loss = []
        val_feature_loss = []
        with tqdm(total=len(valid_data_loader), desc=f"Epoch {epoch + 1}/{args.train_epochs}") as pbar:
            with torch.no_grad():
                 for person_image_bs in valid_data_loader:
                    # person_image_bs = torch.cat((person_image_bs[0], person_image_bs[1]), dim=0)
                    if not isinstance(person_image_bs, torch.Tensor):
                        person_image_bs = torch.stack(person_image_bs)
                    # print(f'person_image_bs: {person_image_bs.shape}')
                    # continue
                    person_image_bs = person_image_bs.to(device)
                    # 教师
                    teacher_output = teacher_model(person_image_bs)
                    # 学生
                    student_output = student_model(person_image_bs)
                    
                    s_loss['ori_loss'] = cosine_similarity_loss(student_output, teacher_output, 1.0)
                    s_loss['mmd_loss'] = mmd_loss(student_output, teacher_output)
                    s_loss['mse_loss'] = mse_loss(student_output, teacher_output)
    
                    output_loss = s_loss['ori_loss'] + s_loss['mmd_loss'] + s_loss['mse_loss']
                    val_output_loss.append(output_loss.item())
    
                    # 更新进度条
                    pbar.set_postfix({'val_output_loss': output_loss.item()})
                    pbar.update(1)
    
            val_output_loss_avg = np.average(val_output_loss)  # 计算平均验证损失
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {train_loss_avg}, Validation Output Loss: {val_output_loss_avg}')
    
        # 保存最优的学生模型
        if val_output_loss_avg < best_loss:
            best_loss = val_output_loss_avg
            temp_best_model_path = os.path.join(path, f"best_student_model.pth")
            torch.save(student_model.state_dict(), temp_best_model_path)
            print(f'Best model saved with Validation Loss: {best_loss}')
    
    # 保存训练好的学生模型
    best_model_path = os.path.join(path, "student_model.pth")
    torch.save(student_model.state_dict(), best_model_path)