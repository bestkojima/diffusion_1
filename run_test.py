from main_structure.model import Unet,secondUnet
from main_structure.schedule import GaussianDiffusion
from main_structure.train import Trainer

import torchvision
import os
import errno
import shutil
import argparse


import torch
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device={device}")

# 从 utils 导入加载 YAML 配置的函数
from utils import load_config_from_yaml

# 加载 YAML 配置文件
config = load_config_from_yaml("./config.yaml")


# 提取 DCA_Sefusion_mnist_32 配置
config = config["test"] #选取对应配置

"换对应模型"
model = Unet(
    dim=config["dim"],
    image_size=config["img_size"],
    dim_mults=config["dim_mults"],
    channels=config["channels"]
).to(device)


print("model initlize")


diffusion = GaussianDiffusion(
    model,
    image_size=config["img_size"],
    channels=config["channels"],
    timesteps=config["numsteps"],  # 使用 YAML 中的 numsteps
    loss_type=config["loss_type"],
    train_routine=config["train_routine"],
    sampling_routine=config["sampling_routine"],
).to(device)


import torch
diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

trainer = Trainer(
    diffusion,
    "./mnist", # data path
    image_size=config["img_size"],
    train_batch_size=config["train_batch_size"],
    train_lr=config["train_lr"],
    train_num_steps=config["train_num_steps"],  # 使用 YAML 中的 train_num_steps
    gradient_accumulate_every=config["gradient_accumulate_every"],
    ema_decay=config["ema_decay"],
    fp16=config["fp16"],
    results_folder=config["name"],
    load_path=config["load_path"],
    save_and_sample_every=config["save_and_sample_every"],
    dataset=config["dataset"]
)

trainer.train(config)