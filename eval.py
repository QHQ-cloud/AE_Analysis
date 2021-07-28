"""完成测试"""
import torch
import config
from model.C3D_denseNeXt_withSEModule import DenseNet
import numpy as np
from process.dataloader_classify import test_loader_val
import os
import pickle
import torch.nn.functional as F
from tqdm import tqdm

model = DenseNet(n_input_channels=1, num_init_features=64,
                       growth_rate=32,
                       block_config=(3, 6, 12, 8), num_classes=4).to(config.device)
model.load_state_dict(torch.load('D:\modelling/checkpoint_classify/model29.pkl'))


# model.load_state_dict(torch.load(os.path.join(config.model_path, 'xxx.pkl')))
# optimizer.load_state_dict(torch.load(os.path.join(config.model_path,'xxx.pkl')))
def eval():

    for idx,(batch_x, batch_y) in enumerate(tqdm(test_loader_val,total=len(test_loader_val),ascii = True)):
        batch_x = batch_x.to(config.device)
        batch_y = batch_y.to(config.device)
        y_hat_test = model(batch_x)  # [10,1]

        break
eval()