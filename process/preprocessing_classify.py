# 专门用来预处理数据
import os
import cv2
# import re
import random
import config
# 直接通过random划分训练集、测试集
# 在这里需要仔细考虑重复问题
count = 16 # 一个文件夹里面存16张图片
class PreprocessClassify(object):
    def __init__(self):
        self.root_dir = r'D:\about_AE_mission'
        self.output_dir = r'D:\modelling\dataset_classify'
    @staticmethod
    def make_dir():# 避免后续麻烦 首先制作好几个标签文件夹
        if not os.path.exists(r'D:\modelling\dataset_classify\train'):
            os.mkdir(r'D:\modelling\dataset_classify\train')
        if not os.path.exists(r'D:\modelling\dataset_classify\test'):
            os.mkdir(r'D:\modelling\dataset_classify\test')
        for i in range(4):  # 在这里先简单一点 以30秒为1级别
            if not os.path.exists(r'D:\modelling\dataset_classify\train\label_{}'.format(i + 1)):
                os.mkdir(r'D:\modelling\dataset_classify\train\label_{}'.format(i + 1))
        for i in range(4):
            if not os.path.exists(r'D:\modelling\dataset_classify\test\label_{}'.format(i + 1)):
                os.mkdir(r'D:\modelling\dataset_classify\test\label_{}'.format(i + 1))
    # 0-20s 21-40s
    # 在每一个文件夹中 都还需要很多的文件夹
    def preprocessing(self,dataset):
        assert dataset == 'AE'
        for dir_ in os.listdir(self.root_dir):  # 目前都是千以上 不会出现乱序问题
            destory_time = int(dir_.split('_')[1])
            pic_files_full = [name for name in os.listdir(self.root_dir + r'\\' + dir_)]
            pic_files_full = sorted(pic_files_full, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            pic_files_1_leave = pic_files_full[1:]  # 缺一个
            pic_files_2_leave = pic_files_full[2:]
            pic_files_3_leave = pic_files_full[3:]
            pic_files_4_leave = pic_files_full[4:]
            pic_files_5_leave = pic_files_full[5:]
            pic_files_6_leave = pic_files_full[6:]
            pic_files_7_leave = pic_files_full[7:]
            pic_files_8_leave = pic_files_full[8:]
            pic_files_9_leave = pic_files_full[9:]
            pic_files_10_leave = pic_files_full[10:]
            pic_files_11_leave = pic_files_full[11:]
            pic_files_12_leave = pic_files_full[12:]
            pic_files_13_leave = pic_files_full[13:]
            pic_files_14_leave = pic_files_full[14:]
            pic_files_15_leave = pic_files_full[15:]

            concat_list = [
                *zip(pic_files_full, pic_files_1_leave, pic_files_2_leave, pic_files_3_leave, pic_files_4_leave,
                     pic_files_5_leave, pic_files_6_leave, pic_files_7_leave, pic_files_8_leave, pic_files_9_leave,
                     pic_files_10_leave, pic_files_11_leave, pic_files_12_leave, pic_files_13_leave, pic_files_14_leave,
                     pic_files_15_leave)]  # list 当中的每一个元组
            # ('B8_01054.jpg','B8_01055.jpg','B8_01056.jpg','B8_01057.jpg',
            #   'B8_01058.jpg','B8_01059.jpg','B8_01060.jpg','B8_01061.jpg','B8_01062.jpg',
            #   'B8_01063.jpg','B8_01064.jpg','B8_01065.jpg','B8_01066.jpg','B8_01067.jpg',
            #   'B8_01068.jpg','B8_01069.jpg')
            for i,obj in enumerate(concat_list):
                # obj 是一个元组 包含16张图片的字符串
                if random.random() < 0.8:
                    access_time = int(obj[-1].split('_')[-1].split('.')[0])
                    # 1354  1069
                    if (destory_time - access_time) > (config.interval * 5):
                        intensity = 1
                    elif (destory_time - access_time) > (config.interval * 3):
                        intensity = 2
                    elif (destory_time - access_time) > (config.interval * 1):
                        intensity = 3
                    else:
                        intensity = 4

                    if not os.path.exists(self.output_dir + r'\\' + 'train' + r'\\' + 'label_{}'.format(intensity) + r'\\' + dir_ + '_' + str(hash(i))):
                        os.mkdir(self.output_dir + r'\\' + 'train' + r'\\' + 'label_{}'.format(intensity) + r'\\' + dir_ + '_' + str(hash(i)))
                        # 这个方法没有返回值
                    for j in obj: # j就是每一张图片的名字
                        img = cv2.imread(os.path.join(self.root_dir,dir_,j))
                        cv2.imwrite(filename=os.path.join(self.output_dir + r'\\' + 'train' + r'\\' + 'label_{}'.format(intensity) + r'\\' + dir_ + '_' + str(hash(i)),j), img=img)
                else:
                    access_time = int(obj[-1].split('_')[-1].split('.')[0])
                    # 1354  1069
                    if (destory_time - access_time) > (config.interval * 5):
                        intensity = 1
                    elif (destory_time - access_time) > (config.interval * 3):
                        intensity = 2
                    elif (destory_time - access_time) > (config.interval * 1):
                        intensity = 3
                    else:
                        intensity = 4
                    if not os.path.exists(self.output_dir + r'\\' + 'test' + r'\\' + 'label_{}'.format(
                            intensity) + r'\\' + dir_ + '_' + str(hash(i))):
                        os.mkdir(self.output_dir + r'\\' + 'test' + r'\\' + 'label_{}'.format(
                            intensity) + r'\\' + dir_ + '_' + str(hash(i)))
                        # 这个方法没有返回值
                    for j in obj:  # j就是每一张图片的名字
                        img = cv2.imread(os.path.join(self.root_dir, dir_, j))
                        cv2.imwrite(filename=os.path.join(self.output_dir + r'\\' + 'test' + r'\\' + 'label_{}'.format(
                            intensity) + r'\\' + dir_ + '_' + str(hash(i)), j), img=img)



# 这个函数是否有泛化能力 是否会与别的产生冲突？
# 还有一个地方没有完成 就是训练集跟测试集的划分 应该在之前就定义好一个train/test的东西 所以代码还需要修改
# 最好定义为一个类 需要我们数据多一点
pre = PreprocessClassify()
pre.make_dir()
pre.preprocessing(config.dataset)
# 这是做回归模型的处理结果
#TODO 需要一个map字典 需要dump loss_list 这是在之后绘图有用