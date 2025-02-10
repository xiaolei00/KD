import torch
import torch.nn as nn
import timm

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