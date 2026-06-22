"""
VPN (Variational Positive-incentive Noise) 核心模块
包含多尺度掩码编码器和VPN生成器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ResNet组件（从VPN项目移植）
def Conv1(channel_in, channel_out, stride=2):
    """ResNet的初始卷积层"""
    return nn.Sequential(
        nn.Conv2d(
            channel_in,
            channel_out,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False
        ),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
    )


class BasicBlock(nn.Module):
    """ResNet的BasicBlock"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# class MultiScaleMaskEncoder(nn.Module):
#     """
#     多尺度掩码编码器
#     保留空间位置和形状信息，输出多尺度特征图
#     """
#     def __init__(self, in_channels=1, base_channels=32):
#         super(MultiScaleMaskEncoder, self).__init__()
        
#         # 初始特征提取
#         self.init_conv = nn.Sequential(
#             nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(base_channels),
#             nn.ReLU(inplace=True)
#         )
        
#         # 1/4尺度分支
#         self.scale_4x = nn.Sequential(
#             nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(base_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(base_channels),
#             nn.ReLU(inplace=True)
#         )
        
#         # 1/8尺度分支
#         self.scale_8x = nn.Sequential(
#             nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(base_channels * 2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(base_channels * 2),
#             nn.ReLU(inplace=True)
#         )
        
#         # 1/16尺度分支
#         self.scale_16x = nn.Sequential(
#             nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(base_channels * 4),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
#             nn.BatchNorm2d(base_channels * 4),
#             nn.ReLU(inplace=True)
#         )
    
#     def forward(self, mask):
#         """
#         Args:
#             mask: 掩码张量 (B, 1, H, W)
#         Returns:
#             features: 多尺度特征图列表
#                 - features[0]: 1/4尺度特征 (B, 32, H/4, W/4)
#                 - features[1]: 1/8尺度特征 (B, 64, H/8, W/8)
#                 - features[2]: 1/16尺度特征 (B, 128, H/16, W/16)
#         """
#         # 初始特征提取
#         x = self.init_conv(mask)
        
#         # 多尺度特征提取
#         feat_4x = self.scale_4x(x)
#         feat_8x = self.scale_8x(feat_4x)
#         feat_16x = self.scale_16x(feat_8x)
        
#         return [feat_4x, feat_8x, feat_16x]


class VPNGenerator(nn.Module):
    """
    VPN生成器：基于ResNet18架构生成正向激励噪声
    参考VPN项目的GaussianNoiseGeneratorResnet18实现
    
    支持两种模式：
    1. 输入模式：在输入图像上加噪声（原有功能）
    2. 特征模式：在多层特征图上加噪声（新功能）
    
    支持共享backbone：
    如果提供了shared_backbone，将使用共享的backbone提取特征，而不是独立的ResNet18编码器
    """
    def __init__(self, 
                 n_channel=1,
                 input_size=(256, 256),
                 coeff_category=0.01,
                 noise_mode='input',
                 shared_backbone=None):
        super(VPNGenerator, self).__init__()
        
        self.n_channel = n_channel
        self.input_size = input_size
        self.coeff_category = coeff_category
        self.noise_mode = noise_mode  # 'input' 或 'feature'
        self.shared_backbone = shared_backbone  # 共享的backbone（如CSPNeXt）
        self.use_shared_backbone = shared_backbone is not None
        
        self.block = BasicBlock
        self.num_blocks = [2, 2, 2, 2]
        self.upper_bound = 0.1
        
        self._build_up()
    
    def _build_up(self):
        """构建网络结构"""
        if self.use_shared_backbone:
            # 使用共享backbone模式：只需要输入融合层和输出头
            # 输入融合层：将图像+掩码组合转换为单通道（用于输入到共享backbone）
            self.input_fusion = nn.Sequential(
                nn.Conv2d(self.n_channel * 2, self.n_channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.n_channel),
                nn.ReLU(inplace=True)
            )
            
            # 特征融合层：融合共享backbone的多层特征
            # CSPNeXt输出4层特征，我们需要融合它们
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(64//2 + 128//2 + 256//2 + 512//2, 64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        else:
            # 独立ResNet18编码器模式（原有实现）
            # 初始卷积层：输入是图像+掩码的组合（2通道）
            self.conv1 = Conv1(self.n_channel * 2, 64, 2)
            
            # ResNet层
            self.in_planes = 64
            self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
            self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
            self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
            self.layer4 = self._make_layer(self.block, 64, self.num_blocks[3], stride=2)
            
            # 特征模式：为多层特征图生成噪声的共享编码器
            # 使用一个共享的编码器提取特征，然后为每一层生成对应的噪声
            self.feature_encoder = nn.Sequential(
                self.conv1,
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4
            )
        
        # 输出头：均值和方差（用于输入模式）
        self.fc_variance = nn.Conv2d(64, self.n_channel, kernel_size=3,
                                     stride=1, padding=1, bias=False)
        self.fc_mean = nn.Conv2d(64, self.n_channel, kernel_size=3,
                                  stride=1, padding=1, bias=False)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        """构建ResNet层"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x, feature_layers=None):
        """
        前向传播
        
        Args:
            x: 图像+掩码的组合 (B, 2*C, H, W)
                - 通道0: 图像
                - 通道1: 掩码（经过coeff_category缩放）
            feature_layers: 特征层配置列表（用于特征模式）
                每个元素为字典：{'channels': int, 'size': (H, W)}
                如果为None，则使用输入模式
        
        Returns:
            输入模式：
                mu: 均值矩阵 (B, C, H, W)
                variance: 方差矩阵 (B, C, H, W)
            特征模式：
                noise_dict: 字典，键为层名，值为噪声张量 (B, num, C, H, W)
        """
        if feature_layers is not None:
            # 特征模式：为多层特征图生成噪声
            return self._forward_feature_mode(x, feature_layers)
        else:
            # 输入模式：在输入图像上加噪声
            return self._forward_input_mode(x)
    
    def _forward_input_mode(self, x):
        """输入模式：在输入图像上加噪声"""
        B, _, H, W = x.shape  # 获取实际输入尺寸
        
        if self.use_shared_backbone:
            # 使用共享backbone：融合输入，通过backbone提取特征，然后融合多层特征
            # 1. 输入融合：将图像+掩码组合转换为单通道
            fused_input = self.input_fusion(x)  # (B, 1, H, W)
            
            # 2. 通过共享backbone提取特征
            backbone_features = self.shared_backbone(fused_input)  # 返回4层特征
            
            # 3. 融合多层特征：将4层特征上采样到相同尺寸并拼接
            # backbone_features[0]: (B, 32, H/4, W/4)
            # backbone_features[1]: (B, 64, H/8, W/8)
            # backbone_features[2]: (B, 128, H/16, W/16)
            # backbone_features[3]: (B, 256, H/32, W/32)
            
            # 上采样所有特征到第一层的尺寸
            feat0 = backbone_features[0]  # (B, 32, H/4, W/4)
            feat1 = F.interpolate(backbone_features[1], size=feat0.shape[2:], mode='bilinear', align_corners=False)  # (B, 64, H/4, W/4)
            feat2 = F.interpolate(backbone_features[2], size=feat0.shape[2:], mode='bilinear', align_corners=False)  # (B, 128, H/4, W/4)
            feat3 = F.interpolate(backbone_features[3], size=feat0.shape[2:], mode='bilinear', align_corners=False)  # (B, 256, H/4, W/4)
            
            # 拼接并融合
            fused_feat = torch.cat([feat0, feat1, feat2, feat3], dim=1)  # (B, 32+64+128+256, H/4, W/4)
            out = self.feature_fusion(fused_feat)  # (B, 64, H/4, W/4)
            
            # 上采样到输入尺寸
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        else:
            # 使用独立ResNet18编码器（原有实现）
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            
            # 动态上采样到输入尺寸
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        # 输出方差（确保为正）
        variance = self.fc_variance(out).abs()
        
        # 输出均值（当前实现中设为0，类似VPN项目）
        mu = torch.zeros(B, self.n_channel, H, W).to(x.device)
        
        return mu, variance
    
    def _forward_feature_mode(self, x, feature_layers):
        """特征模式：为多层特征图生成噪声"""
        B = x.shape[0]
        device = x.device
        
        if self.use_shared_backbone:
            # 使用共享backbone：融合输入，通过backbone提取特征
            # 1. 输入融合：将图像+掩码组合转换为单通道
            fused_input = self.input_fusion(x)  # (B, 1, H, W)
            
            # 2. 通过共享backbone提取特征
            backbone_features = self.shared_backbone(fused_input)  # 返回4层特征
            
            # 3. 融合多层特征
            feat0 = backbone_features[0]  # (B, 32, H/4, W/4)
            feat1 = F.interpolate(backbone_features[1], size=feat0.shape[2:], mode='bilinear', align_corners=False)
            feat2 = F.interpolate(backbone_features[2], size=feat0.shape[2:], mode='bilinear', align_corners=False)
            feat3 = F.interpolate(backbone_features[3], size=feat0.shape[2:], mode='bilinear', align_corners=False)
            
            fused_feat = torch.cat([feat0, feat1, feat2, feat3], dim=1)
            encoded = self.feature_fusion(fused_feat)  # (B, 64, H/4, W/4)
        else:
            # 使用独立ResNet18编码器（原有实现）
            encoded = self.feature_encoder(x)  # (B, 64, H_encoded, W_encoded)
        
        # 为每一层生成噪声
        noise_dict = {}
        for layer_name, layer_config in feature_layers.items():
            channels = layer_config['channels']
            size = layer_config['size']  # (H, W)
            
            # 上采样编码特征到目标尺寸
            encoded_resized = F.interpolate(
                encoded, size=size, mode='bilinear', align_corners=False
            )
            
            # 为每一层生成对应的方差和均值
            # 使用1x1卷积将编码特征映射到目标通道数
            if not hasattr(self, f'fc_variance_{layer_name}'):
                # 动态创建输出头（如果不存在）
                setattr(self, f'fc_variance_{layer_name}',
                       nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1, bias=False).to(device))
                setattr(self, f'fc_mean_{layer_name}',
                       nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1, bias=False).to(device))
            
            variance_layer = getattr(self, f'fc_variance_{layer_name}')(encoded_resized).abs()
            mu_layer = torch.zeros(B, channels, size[0], size[1]).to(device)
            
            # 存储mu和variance（用于后续采样）
            noise_dict[layer_name] = {
                'mu': mu_layer,
                'variance': variance_layer
            }
        
        return noise_dict
    
    def sampling(self, mu, variance, num):
        """
        使用重参数化技巧生成多个噪声样本
        
        Args:
            mu: 均值矩阵 (B, C, H, W) 或 字典（特征模式）
            variance: 方差矩阵 (B, C, H, W) 或 字典（特征模式）
            num: 每个样本采样的噪声数量
        
        Returns:
            输入模式：
                noise: 生成的噪声 (B, num, C, H, W)
            特征模式：
                noise_dict: 字典，键为层名，值为噪声张量 (B, num, C, H, W)
        """
        if isinstance(mu, dict):
            # 特征模式：为多层生成噪声
            noise_dict = {}
            for layer_name in mu.keys():
                mu_layer = mu[layer_name]
                var_layer = variance[layer_name]
                noise_dict[layer_name] = self._sampling_single(mu_layer, var_layer, num)
            return noise_dict
        else:
            # 输入模式：单层噪声
            return self._sampling_single(mu, variance, num)
    
    def _sampling_single(self, mu, variance, num):
        """为单层生成噪声样本"""
        batch_size = mu.shape[0]
        channel = mu.shape[1]
        height = mu.shape[2]
        width = mu.shape[3]
        
        # 采样标准高斯噪声
        noise = torch.randn(batch_size, num, channel, height, width).to(variance.device)
        
        # 扩展variance和mu以匹配采样数量
        var = variance.expand(num, -1, -1, -1, -1).permute(1, 0, 2, 3, 4)
        m = mu.expand(num, -1, -1, -1, -1).permute(1, 0, 2, 3, 4)
        
        # 重参数化：noise = variance * epsilon + mu
        noise = var * noise + m
        
        return noise

