# 3D卷积参数配置
import torch
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 首先做数据预处理 需要定义好目录结构
#./dataset/AE/train or test/label1 or label 2/picture.jpg
nEpochs = 101  # 训练的epoch个数?
snapshot = 1 # 保存模型的频率
lr = 0.005# Learning rate
dataset = 'AE' # acoustic emission
interval = 30
milestones = [150, 225]
current_file = os.path.dirname(os.path.abspath(__file__))
# 模型保存
model_path = r'./checkpoint'
optimizer_path = r'./checkpoint'


model_path_cla = r'./checkpoint'
optimizer_path_cla = r'./checkpoint'
