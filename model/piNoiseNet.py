import torch
import torch.nn as nn
import torch.nn.functional as F
from .Backbone import CSPNeXt
from mmdet.models.necks.fpn import FPN
from .vpn_modules import  VPNGenerator

class Competition2(nn.Module):
    """竞争机制模块，通过像素重排和softmax实现特征增强"""
    
    def __init__(self):
        super(Competition2, self).__init__()
        self.down = nn.PixelUnshuffle(downscale_factor=8)
        self.up = nn.PixelShuffle(upscale_factor=8)

    def forward(self, x):
        x = self.down(x) ** 2
        B, C, H, W = x.shape
        x = x.reshape([B, C, -1]).softmax(dim=-1).reshape(B, C, H, W) * x
        x = self.up(x)
        return x


class piNoiseNet(nn.Module):
    """piNoiseNet模型：用于红外小目标检测的U-Net架构，集成VPN生成器"""
    
    def __init__(self, 
                 enable_vpn=False,
                 noise_mode='input',
                 noise_size_m=1,
                 gamma=0.01):
        super(piNoiseNet, self).__init__()
        
        self.enable_vpn = enable_vpn
        self.noise_mode = noise_mode
        self.noise_size_m = noise_size_m
        
        # 超参数配置
        self.backbone_channels = [64//2, 128//2, 256//2, 512//2]
        self.backbone_channels_cat = [64//2 + 0, 128//2 + 64, 256//2 + 64*4, 512//2 + 64*16]

        self.backbone = CSPNeXt()

        self.neck = FPN(
            in_channels=self.backbone_channels_cat[-3:],
            out_channels=self.backbone_channels[-3],
            act_cfg=dict(type='ReLU'),
            num_outs=len(self.backbone_channels[-3:])
        )

        self.channel_shut1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.backbone_channels[-3] * 2,
                out_channels=self.backbone_channels[-3],
                kernel_size=1,
                stride=1
            ),
            nn.BatchNorm2d(self.backbone_channels[-3]),
            nn.ReLU(inplace=True)
        )
        self.channel_shut2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.backbone_channels[-3] + self.backbone_channels[-4],
                out_channels=self.backbone_channels[-4],
                kernel_size=1,
                stride=1
            ),
            nn.BatchNorm2d(self.backbone_channels[-4]),
            nn.ReLU(inplace=True)
        )
        self.sparse_head = nn.ModuleList([
            nn.Conv2d(self.backbone_channels[-4], self.backbone_channels[-4], 3, 1, 1),
        ])

        self.upsample = nn.Sequential(
            nn.Conv2d(
                self.backbone_channels[-4],
                8,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(8),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.sparse_head.append(self.upsample)

        self.last = nn.Sequential(
            nn.Conv2d(
                8,
                4,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.last2 = nn.Sequential(
            nn.Conv2d(
                4,
                1,
                kernel_size=5,
                stride=1,
                padding=5//2
            ),
        )

        self.up2x = nn.Upsample(scale_factor=2, mode='bilinear')
        self.competition = Competition2()
        
        # VPN模块
        if self.enable_vpn:
            # 注意：新的VPNGenerator不需要mask_encoder，直接使用图像+掩码组合
            # 但为了兼容性，保留mask_encoder（可选）
            # self.mask_encoder = MultiScaleMaskEncoder(in_channels=1, base_channels=32)
            # VPNGenerator的新接口：n_channel, input_size, coeff_category, noise_mode, shared_backbone
            # 使用默认尺寸初始化，forward中会动态适应实际输入尺寸
            # 注意：VPNGenerator现在支持动态输入尺寸，所以可以使用任意默认值
            # 共享backbone架构：将分割网络的backbone传递给VPNGenerator，实现权重共享
            self.vpn_coeff_category = gamma  # 使用gamma作为coeff_category
            self.vpn_generator = VPNGenerator(
                n_channel=1,  # 图像通道数
                input_size=(256, 256),  # 默认尺寸，实际会动态适应
                coeff_category=self.vpn_coeff_category,
                noise_mode=noise_mode,  # 'input' 或 'feature'
                shared_backbone=self.backbone  # 共享backbone（CSPNeXt）
            )





    def forward(self, input, mask=None, input_original=None, mask_original=None, return_noise=False):
        """前向传播
        
        Args:
            input: 输入图像张量，形状为 (B, 1, H, W) - 增强后的图像（用于检测网络）
            mask: 掩码张量，形状为 (B, 1, H, W)，用于VPN生成器（推理时使用预测掩码）
            input_original: 原始图像张量，形状为 (B, 1, H, W) - 未增强的图像（用于VPN generator生成噪声）
            mask_original: 原始掩码张量，形状为 (B, 1, H, W) - 未增强的掩码（用于VPN generator）
            return_noise: 是否返回生成的噪声（用于可视化）
            
        Returns:
            输出预测掩码，形状为 (B, 1, H, W) 或 (B*num, 1, H, W)（如果noise_size_m > 1）
            如果return_noise=True，还返回生成的噪声
            
        Note:
            - 训练时：如果提供input_original和mask_original，VPN generator使用原始图像生成噪声，噪声加到增强图像上
            - 推理时：如果只提供input和mask，VPN generator使用增强图像（向后兼容）
        """
        # 确定用于VPN generator的图像和mask
        if input_original is not None and mask_original is not None:
            # 训练模式：使用原始图像和mask生成噪声
            vpn_input = input_original
            vpn_mask = mask_original
        else:
            # 推理模式或向后兼容：使用增强后的图像
            vpn_input = input
            vpn_mask = mask
        
        # 确定用于检测网络的输入（始终使用增强后的图像）
        detection_input = input
        
        B, C, H, W = detection_input.shape
        noise = None
        original_batch_size = B
        batch_data_labels = None  # 用于特征模式
        
        # VPN噪声生成
        if self.enable_vpn and vpn_mask is not None:
            # 确保vpn_generator在正确的设备上
            if next(self.vpn_generator.parameters()).device != vpn_input.device:
                self.vpn_generator = self.vpn_generator.to(vpn_input.device)
            
            # 类似VPN项目：将图像和掩码组合
            # batch_data_labels = batch_data + mask_coeff * mask
            # 这里mask已经是归一化的，直接使用coeff_category缩放
            mask_scaled = vpn_mask * self.vpn_coeff_category
            batch_data_labels = torch.cat([vpn_input, mask_scaled], dim=1)  # (B, 2, H, W)
            
            if self.noise_mode == 'input':
                # 输入模式：在输入图像上加噪声
                mu, variance = self.vpn_generator(batch_data_labels)
                noises = self.vpn_generator.sampling(mu, variance, self.noise_size_m)
                
                # 扩展增强后的输入batch，对每个噪声样本分别处理
                expanded_input = detection_input.expand(self.noise_size_m, -1, -1, -1, -1).permute(1, 0, 2, 3, 4)
                input_with_noise = expanded_input + noises  # (B, noise_size_m, C, H, W)
                detection_input = input_with_noise.reshape(B * self.noise_size_m, C, H, W)
                
                if return_noise:
                    noise = noises.reshape(B * self.noise_size_m, C, H, W)
            # 特征模式：将在后续特征层上加噪声，这里先不处理
        
        # 使用detection_input进行后续处理
        input = detection_input
        
        # 像素重排下采样
        ps8x = self.pixel_shuffle_down(self.pixel_shuffle_down(self.pixel_shuffle_down(input)))
        ps16x = self.pixel_shuffle_down(ps8x)
        ps32x = self.pixel_shuffle_down(ps16x)

        # 骨干网络特征提取
        layers = self.backbone(input)
        feats4x = layers[0]
        
        # VPN特征模式：为特征层生成噪声
        if self.enable_vpn and self.noise_mode == 'feature' and batch_data_labels is not None:
            # 准备特征层配置
            feature_layers = {
                'layers_0': {'channels': layers[0].shape[1], 'size': (layers[0].shape[2], layers[0].shape[3])},
                'layers_1': {'channels': layers[1].shape[1], 'size': (layers[1].shape[2], layers[1].shape[3])},
                'layers_2': {'channels': layers[2].shape[1], 'size': (layers[2].shape[2], layers[2].shape[3])},
                'layers_3': {'channels': layers[3].shape[1], 'size': (layers[3].shape[2], layers[3].shape[3])},
            }
            
            # 生成噪声
            noise_dict = self.vpn_generator(batch_data_labels, feature_layers=feature_layers)
            noises_dict = self.vpn_generator.sampling(
                {k: v['mu'] for k, v in noise_dict.items()},
                {k: v['variance'] for k, v in noise_dict.items()},
                self.noise_size_m
            )
            
            # 在每一层特征图上应用噪声
            # 注意：需要扩展batch维度以匹配noise_size_m
            for i, layer_name in enumerate(['layers_0', 'layers_1', 'layers_2', 'layers_3']):
                if layer_name in noises_dict:
                    noise_layer = noises_dict[layer_name]  # (B, noise_size_m, C, H, W)
                    # 扩展特征图
                    expanded_layer = layers[i].expand(self.noise_size_m, -1, -1, -1, -1).permute(1, 0, 2, 3, 4)
                    # 添加噪声
                    layers[i] = (expanded_layer + noise_layer).reshape(B * self.noise_size_m, layers[i].shape[1], layers[i].shape[2], layers[i].shape[3])
            
            # 更新feats4x（从layers[0]提取）
            feats4x = layers[0]
            # 更新B以反映扩展后的batch size
            B = B * self.noise_size_m
        
        # 特征融合
        layers[-3] = torch.concat([layers[-3], ps8x], dim=1)
        layers[-2] = torch.concat([layers[-2], ps16x], dim=1)
        layers[-1] = torch.concat([layers[-1], ps32x], dim=1)

        # FPN特征金字塔
        feats = self.neck(layers[-3:])
        feats8x, feats16x, feats32x = feats

        # VPN特征模式：在FPN特征层上加噪声
        if self.enable_vpn and self.noise_mode == 'feature' and batch_data_labels is not None:
            # 为FPN特征层生成噪声
            # 注意：如果之前已经扩展了batch，FPN特征的batch已经是扩展后的
            # 需要基于原始batch size生成噪声，然后扩展到当前batch
            original_B = B // self.noise_size_m if B > original_batch_size else original_batch_size
            
            fpn_feature_layers = {
                'feats8x': {'channels': feats8x.shape[1], 'size': (feats8x.shape[2], feats8x.shape[3])},
                'feats16x': {'channels': feats16x.shape[1], 'size': (feats16x.shape[2], feats16x.shape[3])},
                'feats32x': {'channels': feats32x.shape[1], 'size': (feats32x.shape[2], feats32x.shape[3])},
            }
            
            fpn_noise_dict = self.vpn_generator(batch_data_labels, feature_layers=fpn_feature_layers)
            fpn_noises_dict = self.vpn_generator.sampling(
                {k: v['mu'] for k, v in fpn_noise_dict.items()},
                {k: v['variance'] for k, v in fpn_noise_dict.items()},
                self.noise_size_m
            )
            
            # 应用噪声到FPN特征
            # 如果batch已经扩展（B > original_batch_size），说明之前已经扩展了
            # 此时需要将噪声也扩展到匹配的batch size
            if 'feats8x' in fpn_noises_dict:
                if B == original_batch_size:
                    # batch未扩展，需要扩展
                    expanded_feats8x = feats8x.expand(self.noise_size_m, -1, -1, -1, -1).permute(1, 0, 2, 3, 4)
                    feats8x = (expanded_feats8x + fpn_noises_dict['feats8x']).reshape(original_B * self.noise_size_m, feats8x.shape[1], feats8x.shape[2], feats8x.shape[3])
                else:
                    # batch已扩展，直接应用噪声（噪声已经是(B, noise_size_m, C, H, W)格式）
                    expanded_feats8x = feats8x.unsqueeze(1).expand(-1, self.noise_size_m, -1, -1, -1)
                    feats8x = (expanded_feats8x + fpn_noises_dict['feats8x']).reshape(B, feats8x.shape[1], feats8x.shape[2], feats8x.shape[3])
            if 'feats16x' in fpn_noises_dict:
                if B == original_batch_size:
                    expanded_feats16x = feats16x.expand(self.noise_size_m, -1, -1, -1, -1).permute(1, 0, 2, 3, 4)
                    feats16x = (expanded_feats16x + fpn_noises_dict['feats16x']).reshape(original_B * self.noise_size_m, feats16x.shape[1], feats16x.shape[2], feats16x.shape[3])
                else:
                    expanded_feats16x = feats16x.unsqueeze(1).expand(-1, self.noise_size_m, -1, -1, -1)
                    feats16x = (expanded_feats16x + fpn_noises_dict['feats16x']).reshape(B, feats16x.shape[1], feats16x.shape[2], feats16x.shape[3])
            if 'feats32x' in fpn_noises_dict:
                if B == original_batch_size:
                    expanded_feats32x = feats32x.expand(self.noise_size_m, -1, -1, -1, -1).permute(1, 0, 2, 3, 4)
                    feats32x = (expanded_feats32x + fpn_noises_dict['feats32x']).reshape(original_B * self.noise_size_m, feats32x.shape[1], feats32x.shape[2], feats32x.shape[3])
                else:
                    expanded_feats32x = feats32x.unsqueeze(1).expand(-1, self.noise_size_m, -1, -1, -1)
                    feats32x = (expanded_feats32x + fpn_noises_dict['feats32x']).reshape(B, feats32x.shape[1], feats32x.shape[2], feats32x.shape[3])

        # 特征融合和上采样
        feats8x = self.channel_shut1(torch.concat([feats8x, self.up2x(feats16x)], dim=1))
        feats4x = self.channel_shut2(torch.concat([feats4x, self.up2x(feats8x)], dim=1))
        out = feats4x

        # 稀疏头处理
        for m in self.sparse_head:
            out = m(out)

        # 最终输出
        out = self.last2(self.last(out))
        output = out.sigmoid()
        
        if return_noise:
            return output, noise
        return output

    def pixel_shuffle_down(self, x):
        """像素重排下采样
        
        Args:
            x: 输入特征图
            
        Returns:
            下采样后的特征图
        """
        return torch.pixel_unshuffle(x, 2)

