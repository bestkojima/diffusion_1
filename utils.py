class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

import yaml

# 从 YAML 文件加载配置
def load_config_from_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


import torch
def svd_batch(matrix,k=None):
    U, S, V = torch.svd(matrix)
    
    
    "soft make"
    # a=torch.diag_embed(S)
    # temp=torch.zeros_like(a)
    # temp[:,:,:2,:2]=a[:,:,:2,:2]
    # a.shape
    reconstructed_matrix = torch.matmul(torch.matmul(U,torch.diag_embed(S)) , V.transpose(2,3))
    
    return reconstructed_matrix


import torch
from torch import nn

class ModelParamsCounter:
    """
    用于计算模型参数量和模型大小的 Wrapper 类。
    """
    def __init__(self, model: nn.Module):
        """
        初始化模型参数量统计器。

        :param model: 需要统计参数量的 PyTorch 模型。
        """
        self.model = model


        self.print_parameter_info(trainable_only=True)  # 只统计可训练的参数
        self.print_parameter_info(trainable_only=False)
    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        计算模型的参数量。

        :param trainable_only: 是否只统计可训练的参数。
        :return: 模型的参数量（以整数表示）。
        """
        if trainable_only:
            # 只统计可训练的参数
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            # 统计所有参数
            return sum(p.numel() for p in self.model.parameters())

    def calculate_model_size(self, trainable_only: bool = True) -> int:
        """
        计算模型的大小（以字节为单位）。

        :param trainable_only: 是否只统计可训练的参数。
        :return: 模型的大小（以字节为单位）。
        """
        total_size = 0
        for param in self.model.parameters():
            if trainable_only and not param.requires_grad:
                continue
            total_size += param.numel() * param.element_size()
        return total_size

    def print_parameter_info(self, trainable_only: bool = True):
        """
        打印模型的参数量和模型大小信息。

        :param trainable_only: 是否只统计可训练的参数。
        """
        total_params = self.count_parameters(trainable_only)
        total_size = self.calculate_model_size(trainable_only)

        print(f"{'Trainable' if trainable_only else 'Total'} Parameters: {total_params}")
       
        print(f"{'Trainable' if trainable_only else 'Total'} Model Size: {total_size / 1024 / 1024:.2f} MB")
        print(f"{'Trainable' if trainable_only else 'Total'} Model Size: {total_size / 1024 / 1024 / 1024:.2f} GB")
       

# 示例模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 主函数

def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


if __name__ == '__main__':

     # 创建模型
    model = SimpleModel()

    # 创建 Wrapper 实例
    params_counter = ModelParamsCounter(model)

  

    # matrix = torch.randn(2,3,3,3)
    # c=svd_batch(matrix)
    # from torch.nn import functional as F
    # z=F.mse_loss(c,matrix)
    # print(z)