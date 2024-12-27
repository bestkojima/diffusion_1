from comet_ml import Experiment
import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image

import numpy as np
from tqdm import tqdm
from einops import rearrange

import torchgeometry as tgm
import glob
import os
from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from torch import linalg as LA
from sklearn.mixture import GaussianMixture
try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


from dataset import Dataset,Dataset_Aug1
from utils import EMA
import os
import errno
from datetime import datetime
def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass



def cycle(dl):
    while True:
        for data in dl:
            yield data

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        load_path = None,
        dataset = None,
        shuffle=True
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.lr=train_lr
        
        if dataset == 'train':
            print(dataset, "DA used")
            self.ds = Dataset_Aug1(folder, image_size)
        else:
            print(dataset)
            self.ds = Dataset(folder, image_size)

        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=shuffle, pin_memory=True, num_workers=16, drop_last=True))

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.fp16 = fp16

        self.reset_parameters()

        if load_path != None:
            self.load(load_path)


    def reset_parameters(self):
        """
        model_parms -> ema_model
        """
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        """
        ema_model update
        """
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, itrs=None):
        """
        保存模型数据
        TODO: add dataset name
        """
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        if itrs is None:
            torch.save(data, str(self.results_folder / f'model.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model_{itrs}.pt'))
    
    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])


    def add_title(self, path, title):
        """
        
        none
        """
        import cv2
        import numpy as np

        img1 = cv2.imread(path)

        # --- Here I am creating the border---
        black = [0, 0, 0]  # ---Color of the border---
        constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
        height = 20
        violet = np.zeros((height, constant.shape[1], 3), np.uint8)
        violet[:] = (255, 0, 180)

        vcat = cv2.vconcat((violet, constant))

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(vcat, str(title), (violet.shape[1] // 2, height-2), font, 0.5, (0, 0, 0), 1, 0)
        cv2.imwrite(path, vcat)



    def train(self,config):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        backwards = partial(loss_backwards, self.fp16)

        self.step = 0
        acc_loss = 0
        
        #######################!!!wandb!!!###################
        #TODO
        import wandb
        os.environ["WANDB_MODE"] = "offline" #  online comment
        wandb.init(
    # set the wandb project where this run will be logged
        project="Diffusion_skip_connection",
        name=config["name"],
        notes="test_run",
        tags=["test","test2"],
    
        # track hyperparameters and run metadata
        config={"learning_rate": self.lr,
                "architecture": "DCA+sefusion",
                "train_num_steps": self.train_num_steps,
                "result_folder": self.results_folder,
                "image_size": self.image_size,
                "train_batch_size": self.batch_size,
                "save_and_sample_every":self.save_and_sample_every,
                

        }
        )
        wandb.config.update(config)






        #######################!!!wandb!!!###################

# 使用 tqdm 包装 while 循环
        with tqdm(total=self.train_num_steps, desc="Training", unit="step") as pbar:
            while self.step < self.train_num_steps:
                u_loss = 0
                for i in range(self.gradient_accumulate_every):
                    data_1 = next(self.dl)
                    data_2 = torch.randn_like(data_1)
                    
                    data_1, data_2 = data_1.to(device), data_2.to(device)
                    
                    loss = torch.mean(self.model(data_1, data_2))
                    if self.step % 100 == 0:
                        print(f'{self.step}: {loss.item()}')
                    u_loss += loss.item()
                    
                    backwards(loss / self.gradient_accumulate_every, self.opt)

                acc_loss = acc_loss + (u_loss / self.gradient_accumulate_every)
                ######
                
                wandb.log({"loss": u_loss / self.gradient_accumulate_every})
                
               
                # from utils import grad_norm
                # try:
                #     wandb.log({"grad_norm": grad_norm(self.ema_model.module.denoise_fn)}) # 梯度震荡
                #     print(grad_norm(self.ema_model.module.denoise_fn))
                # except Exception as e:
                #     print("梯度震荡:"+str(e))
                ##

                # hidden_norm=0.0
                # for i in self.ema_model.module.denoise_fn.skip_output:
                #     hidden_norm += torch.norm(i,p=2).item()
                # wandb.log({"hidden_norm": hidden_norm}) # 隐藏震荡
                # print(hidden_norm)
                
                ###
                self.opt.step()
                self.opt.zero_grad()

                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    milestone = self.step // self.save_and_sample_every
                    batches = self.batch_size

                    data_1 = next(self.dl) # 原数据
                    data_2 = torch.randn_like(data_1)
                    og_img = data_2.to(device) #修改原图
    
                    xt, direct_recons, all_images = self.ema_model.module.sample(batch_size=batches, img=og_img)

                    #生成图像
                    all_images = (all_images + 1) * 0.5
                    utils.save_image(all_images, str(self.results_folder / f'sample-recon-{milestone}.png'), nrow=6)

                    # direct_recons = (direct_recons + 1) * 0.5
                    # utils.save_image(direct_recons, str(self.results_folder / f'sample-direct_recons-{milestone}.png'), nrow=6)

                    # xt = (xt + 1) * 0.5
                    # utils.save_image(xt, str(self.results_folder / f'sample-xt-{milestone}.png'), nrow=6)

                    acc_loss = acc_loss / (self.save_and_sample_every + 1)
                    print(f'Mean of last {self.step}: {acc_loss}')
                    acc_loss = 0
                    ####
                    print(data_1.shape)
                    print(all_images.shape)
                    #fid ssim rmse
                    from pytorch_msssim import ssim
                    rmse_score = torch.sqrt(torch.mean((data_1.to(device) - all_images) ** 2))
                    wandb.log({"rmse_score": rmse_score.item()})
                    ssim_score = ssim(data_1.to(device), all_images, data_range=1, size_average=True)
                    wandb.log({"ssim_score": ssim_score.item()})
                    from criteria.fid_score import calculate_fid_given_samples
                    fid_score = calculate_fid_given_samples(samples=[data_1.to("cpu"), all_images.to("cpu")],batch_size=batches)
                    wandb.log({"fid_score": fid_score})

                    ####
                    self.save()
                    if self.step % (self.save_and_sample_every * 100) == 0:
                        self.save(self.step)

                self.step += 1
        
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item(),"Current Time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

        print('training completed')

    