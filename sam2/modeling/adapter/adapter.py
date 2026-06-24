# In training/model/adapter.py
import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        """
        一个简单的Adapter模块.
        :param input_dim: 输入特征的维度 (例如SAM2中的256)
        :param bottleneck_dim: 瓶颈层的维度 (一个远小于input_dim的数，例如64)
        """
        super().__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.relu = nn.ReLU()
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        
        # 初始化up_project的权重为0，确保在训练开始时，Adapter是一个恒等变换
        # 这是一个关键的技巧，能让训练过程更稳定
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        # 核心路径：降维 -> 激活 -> 升维
        bottleneck = self.down_project(x)
        activated = self.relu(bottleneck)
        up_projected = self.up_project(activated)
        
        # 残差连接：将原始输入x与Adapter的输出相加
        # 这是一个“旁路”，让原始信息可以直接流过
        return x + up_projected