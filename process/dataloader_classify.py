# 建立 分类模型的 data loader
import torch
import os
import numpy as np
import cv2
count = 16
from torch.utils.data import Dataset,DataLoader
class PicDatasetClassify(Dataset):
    def __init__(self,dataset='AE',split='train',clip_len=16):
        assert dataset == 'AE'
        self.root_dir = r'D:\about_AE_mission'
        self.output_dir = r'D:\modelling\dataset_classify'
        folder = os.path.join(self.output_dir, split) # r'D:\modelling\dataset_classify\train'
        self.clip_len = clip_len  # 16张图片代表一个状态
        self.split = split
        # 数据还需要进行截取
        label_list = os.listdir(folder)
        label_list = sorted(label_list,key = lambda x:int(x.split('_')[-1]),reverse=False)
        #['label_1', 'label_2', 'label_3', 'label_4']
        self.fnames, self.labels = [], []
        for label in label_list: # every label --> string   'label_1'
            for fname in os.listdir(os.path.join(folder, label)):
                #r'D:\modelling\dataset_classify\train\label_1'  # 没有直接跳过
                # 命名时候不能出现相同
                # 如果文件夹里面不存在文件夹 直接略过
                self.fnames.append(os.path.join(folder, label, fname))
                self.labels.append(np.float32(int(label.split('_')[-1]) -1)) # 现在目标值是一个连续值 退而求其次 可以有分类
                # 现在的label [0,0,0,0,...,1,1,1,1,1,1,1....,2,2,2,2...,3,3,3....]

        assert len(self.labels) == len(self.fnames)
        # 现在要考虑的一点是 因为出现了越位的情况 遍历是否会出问题
        self.label2index = {label: index for index, label in enumerate(label_list)}
        # {'label_1': 0, 'label_2': 1, 'label_3': 2, 'label_4': 3}
        # self.label_array = np.array([self.label2index[label] for label in self.labels]).astype(np.float32)
        # array([2,2,2,2,2,2,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,...])
    def __getitem__(self, index):
# 在这个get_item阶段 还需要对数据进行预处理
        buffer = self.load_frames(self.fnames[index])  # [16,1000,500,3]
        # r'D:\modelling\dataset\train\label_0\AEPic-185_141' 每一个这样的文件夹
        buffer = self.crop(buffer, self.clip_len) # [8,572,572,3]
        # 最好使用224 这样的图片大小
        label = np.array(self.labels[index])  # 0 -> int
        buffer = self.normalize(buffer)  # 去均值并去噪 []
        buffer = self.resize(buffer)
        buffer = self.to_tensor(buffer)  # 转换channel_first
        return torch.from_numpy(buffer), torch.LongTensor(label)  # 两个np.float32 为何报错？

    def __len__(self):
        return len(self.fnames)


    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)],key = lambda x:int(x.split('_')[-1].split('.')[0]))
        # ['D:\\modelling\\dataset\\train\\label_0\\AEPic-185_141\\0178-0001.jpg',
        #  'D:\\modelling\\dataset\\train\\label_0\\AEPic-185_141\\0179-0001.jpg',
        #  'D:\\modelling\\dataset\\train\\label_0\\AEPic-185_141\\0180-0001.jpg',
        #  'D:\\modelling\\dataset\\train\\label_0\\AEPic-185_141\\0181-0001.jpg',
        #  'D:\\modelling\\dataset\\train\\label_0\\AEPic-185_141\\0182-0001.jpg',
        #  'D:\\modelling\\dataset\\train\\label_0\\AEPic-185_141\\0183-0001.jpg',
        #  'D:\\modelling\\dataset\\train\\label_0\\AEPic-185_141\\0184-0001.jpg',
        #  'D:\\modelling\\dataset\\train\\label_0\\AEPic-185_141\\0185-0001.jpg']
        frame_count = len(frames)  #16
        # 需要对图片进行crop 左边274 右边272 留下 572 上边50 下边51
        # 这是之前的图片需要进行crop 但是目前不需要
        buffer = np.empty((frame_count, 1000, 500, 3), np.dtype('float32'))
        # 1118 和 673 是 原始图片的大小
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64) # [1000, 500, 3]
            buffer[i] = frame
        # [16,1000,500,3]
        return buffer

    def crop(self, buffer, clip_len):
        # randomly select time index for temporal jittering

        buffer = buffer[:,:,:,:1]
        # [16,1000,500,1]
        return buffer
    def normalize(self, buffer):
        for i, frame in enumerate(buffer):  # [None,572,572,3]
            frame -= np.array([[[248.1185]]])
            frame /= np.array([[[37.7234]]])
            frame = cv2.GaussianBlur(frame, (7, 7), 1)
            frame = np.expand_dims(frame, axis=-1)
            buffer[i] = frame
        # 数据集的BGR平均值为
        # 248.11852324710082
        # 数据集的BGR标准差为
        # 37.723387243403195
        return buffer
    # [16,1000,500,1]
    # 应该是不需要二值化的 本来就是二值化图像
    def resize(self,buffer):
        buffer_resize = np.empty((count, 448, 224, 1), np.dtype('float32'))
        for i,frame in enumerate(buffer):  # [1000,500,1]
            frame = cv2.resize(frame,(224,448))  # numpy.ndarray [224,224,3]
            frame = np.expand_dims(frame, axis=-1)
            buffer_resize[i] = frame
        return buffer_resize
    # [16,448,224,1]  -> [1,16,448,224]
    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))  # channel_first


train_data = PicDatasetClassify(dataset='AE', split='train', clip_len=16)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_data = PicDatasetClassify(dataset='AE', split='test', clip_len=16)
test_loader = DataLoader(test_data,batch_size=4)
test_loader_val = DataLoader(test_data, shuffle=True,batch_size=1)
# 由于使用了新的网络结构，这个batch_size还要修改
# for i, sample in enumerate(train_loader):
#     inputs = sample[0]
#     labels = sample[1]
#     print(inputs.size())
#     print(labels)
#
#     if i == 6:
#         break
# print(len(train_data))
# print(train_data.labels)
# 如何处理样本不均衡问题？
# TODO 不要忘了 label 从0开始 否则会报cuda错误

# 由于电脑的配置问题 所以还是要把关键的东西拿回来