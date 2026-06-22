"""工具模块：包含卷积块组件"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """Linear Attention模块：使用线性复杂度的注意力机制
    
    Args:
        dim: 输入特征维度（通道数）
    """
    
    def __init__(self, dim):
        super(LinearAttention, self).__init__()
        self.dim = dim
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
            
        Returns:
            处理后的特征图 (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 生成Q, K, V
        qkv = self.to_qkv(x)  # (B, 3*C, H, W)
        q, k, v = qkv.chunk(3, dim=1)  # 每个都是 (B, C, H, W)
        
        # 转换为 (B, H*W, C) 格式
        q = q.view(B, C, -1).permute(0, 2, 1).contiguous()  # (B, H*W, C)
        k = k.view(B, C, -1).permute(0, 2, 1).contiguous()  # (B, H*W, C)
        v = v.view(B, C, -1).permute(0, 2, 1).contiguous()  # (B, H*W, C)
        
        # 归一化
        q = self.norm(q)  # (B, H*W, C)
        k = self.norm(k)  # (B, H*W, C)
        
        # Linear attention: Q * (K^T * V) 而不是 (Q * K^T) * V
        # 先计算 K^T * V，复杂度为 O(HWC^2) 而不是 O(H^2W^2C)
        # K^T: (B, C, H*W), V: (B, H*W, C) -> K^T * V: (B, C, C)
        k = F.softmax(k, dim=-1)  # 对K进行softmax归一化
        kv = torch.bmm(k.transpose(1, 2), v)  # (B, C, C)
        
        # 计算 Q * (K^T * V)
        # Q: (B, H*W, C), kv: (B, C, C) -> out: (B, H*W, C)
        out = torch.bmm(q, kv)  # (B, H*W, C)
        
        # 转换回 (B, C, H, W) 格式
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)  # (B, C, H, W)
        
        # 投影层
        out = self.proj(out)
        
        # 残差连接
        out = out + x
        
        return out


class conv_block(nn.Module):
    """基础卷积块：包含两个连续的卷积层
    
    Args:
        in_ch: 输入通道数
        out_ch: 输出通道数
    """
    
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.main_branch2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=3 // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入特征图
            
        Returns:
            处理后的特征图
        """
        x1 = self.main_branch(x)
        x2 = self.main_branch2(x1)
        return x2
