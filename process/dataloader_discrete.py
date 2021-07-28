import torch
import os
import numpy as np
import cv2
count = 16
from torch.utils.data import Dataset,DataLoader
class PicDataset(Dataset):
    def __init__(self,dataset='AE',split='train',clip_len=16):
        assert dataset == 'AE'
        self.root_dir = r'D:\about_AE_mission'
        self.output_dir = r'D:\modelling\dataset'
        folder = os.path.join(self.output_dir, split) # r'D:\modelling\dataset\train'
        self.clip_len = clip_len  # 16张图片代表一个状态
        self.split = split
        # self.crop_size = 572
        # 数据还需要进行截取  目前是不能截取 只能是resize
        label_list = os.listdir(folder)
        label_list = sorted(label_list,key = lambda x:int(x.split('_')[-1]),reverse=False)
        # sorted(a,key = lambda x:int(x.split('_')[-1]),reverse=True)
        # ['','','','']
        self.fnames, self.labels = [], []
        for label in label_list: # every label --> string
            for fname in os.listdir(os.path.join(folder, label)):
                #r'D:\modelling\dataset\train\label_0'
                # 命名时候不能出现相同
                # 如果文件夹里面不存在文件夹 直接略过
                self.fnames.append(os.path.join(folder, label, fname))
                self.labels.append(np.float32(int(label.split('_')[-1]))) # 现在目标值是一个连续值 退而求其次 可以有分类
                # 现在的label 本来需要有一个字典的map 但我们现在是回归模型 重复但需要保证数量一致
# 现在的label里面包含的东西必须是数字 因为是回归 [0,0,0,0,1,1,1,1,...]

        assert len(self.labels) == len(self.fnames)
    def __getitem__(self, index):
# 在这个get_item阶段 还需要对数据进行预处理
        buffer = self.load_frames(self.fnames[index])  # [8,673,1118,3]
        # r'D:\modelling\dataset\train\label_0\AEPic-185_141' 每一个这样的文件夹
        buffer = self.crop(buffer, self.clip_len) # [8,572,572,3]
        # 最好使用224 这样的图片大小
        label = np.array(self.labels[index])  # 0 -> int
        buffer = self.normalize(buffer)  # 去均值并去噪 [8,572,572,3]
        buffer = self.resize(buffer)
        buffer = self.to_tensor(buffer)  # 转换channel_first
        return torch.from_numpy(buffer), torch.LongTensor(label)

    def __len__(self):
        return len(self.fnames)


    def load_frames(self, file_dir):  # 目前来看 因为都是四位数 所以不会出现顺序不对的情况
        # frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)],
                        key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # ['D:\\modelling\\dataset\\train\\label_0\\AEPic-185_141\\0178-0001.jpg',
        #  'D:\\modelling\\dataset\\train\\label_0\\AEPic-185_141\\0179-0001.jpg',
        #  'D:\\modelling\\dataset\\train\\label_0\\AEPic-185_141\\0180-0001.jpg',
        #  'D:\\modelling\\dataset\\train\\label_0\\AEPic-185_141\\0181-0001.jpg',
        #  'D:\\modelling\\dataset\\train\\label_0\\AEPic-185_141\\0182-0001.jpg',
        #  'D:\\modelling\\dataset\\train\\label_0\\AEPic-185_141\\0183-0001.jpg',
        #  'D:\\modelling\\dataset\\train\\label_0\\AEPic-185_141\\0184-0001.jpg',
        #  'D:\\modelling\\dataset\\train\\label_0\\AEPic-185_141\\0185-0001.jpg']
        frame_count = len(frames)
        # 缓存 pytorch 中是channels first
        buffer = np.empty((frame_count, 1000, 500, 3), np.dtype('float32'))
        # 1000 和 500 是 原始图片的大小
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64) # [1000, 500, 3]
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len):
        # randomly select time index for temporal jittering
        time_index = 0  # 已经在之前处理好了16张图片 所以此处time_stamp 为0
        buffer = buffer[:,:,:,:1]

        return buffer
    def normalize(self, buffer):  # [16,1000,500,1]
        for i, frame in enumerate(buffer):  # [None,572,572,3]
            frame -= np.array([[[248.1185]]])
            frame /= np.array([[[37.7234]]])
            frame = cv2.GaussianBlur(frame, (7, 7), 1)
            frame = np.expand_dims(frame, axis=-1)
            buffer[i] = frame
        # [175.66002324732824，175.66270362571802，176.3923076070816]
        # 算出来的均值很大  发现去均值后 产生噪音 采用高斯滤波
        # [10.107902943727074，10.10220117886538，9.057330856906324]
        return buffer

    def resize(self,buffer):
        buffer_resize = np.empty((count, 448, 224, 1), np.dtype('float32'))
        for i,frame in enumerate(buffer):  # [572,572,3]
            frame = cv2.resize(frame,(224,448))  # numpy.ndarray [224,224,3]
            frame = np.expand_dims(frame, axis=-1)
            buffer_resize[i] = frame
        return buffer_resize

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))  # channel_first


train_data = PicDataset(dataset='AE', split='train', clip_len=16)
train_loader = DataLoader(train_data, batch_size=3, shuffle=True)
test_data = PicDataset(dataset='AE', split='test', clip_len=16)
test_loader = DataLoader(test_data, batch_size=3)
if __name__ == '__main__':
    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)
        # 现在的测试集合太少了
        if i == 2:
            break
# torch.Size([3, 1, 16, 448, 224])
# tensor([ 45.,  33., 194.])
# torch.Size([3, 1, 16, 448, 224])
# tensor([191., 108.,   4.])
# torch.Size([3, 1, 16, 448, 224])
# tensor([247., 179., 148.])
#todo 做迁移学习的步骤
# 之前我们选择的十分类模型其参数已经基本上训练的很好了，如何才能迁移到新的分类模型上呢？
# 在这里保持我们的卷积层参数不变 其线性层发生了较大的变化 希望模型训练起来比较容易一点
# model_dict = model.state_dict()  # 这里的 model 是我们新训练的模型，其参数仅仅是何凯明参数
# pretrained_dict = torch.load('D://modelling//checkpoint_classify//previeus//model32.pkl')
# # 这是导入我们之前10分类模型的参数 这里的参数跟新模型有所不同 我们只保留卷积参数
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias', 'conv4a.weight', 'conv4a.bias', 'conv4b.weight', 'conv4b.bias', 'conv5a.weight', 'conv5a.bias', 'conv5b.weight', 'conv5b.bias', 'conv6a.weight', 'conv6a.bias', 'conv6b.weight', 'conv6b.bias']}
# # 这一步是筛选
# model_dict.update(pretrained_dict)  # 更新新模型卷积层的参数
# model.load_state_dict(model_dict)
# 运行一次以后 就消掉 因为有更好的保存下来