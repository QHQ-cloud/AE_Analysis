# 计算自己图片数据集的均值 方便 pytorch 中去均值 处理
import os
import numpy as np
import cv2  # opencv的图像处理格式是BGR
img_list = []
# 一般情况下 我们需要对训练集做处理 求均值 带入测试集中
ims_root_path = r'D:\modelling\dataset_classify\train'  # 图像数据集的路径 必须用训练集带入测试集
for ims_path in os.listdir(ims_root_path):  # label_1
    ims_list = os.listdir(os.path.join(ims_root_path,ims_path))
    for img_list_16 in ims_list:  # 'AEPic_1354_10z_280'
        imgx_list = os.listdir(os.path.join(ims_root_path,ims_path,img_list_16))
        for img in imgx_list:
            img_list.append(os.path.join(ims_root_path,ims_path,img_list_16,img))


# 在这里得到的均值是训练集所有图片的均值
# 用来计算均值的，一定要用训练集，否则违背了深度学习的原则（即模型训练仅能从训练数据中获取信息）。对于得到的mean值，训练集、验证集和测试集都要分别减去
# B_means = []
# G_means = []
_means = []
# B_std = []
# G_std = []
_std = []
# 还没有对图片进行crop操作   不能crop  只能resize
for img in img_list:
    im = cv2.imread(img)  # [1000,500,3]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # [1000,500,1]
    # extract value of diffient channel
    # count mean for every channel
    im__mean = np.mean(im)

    im__std = np.std(im)

    # save single mean value to a set of means
    _means.append(im__mean)

    _std.append(im__std)


print('数据集的BGR平均值为\n{}'.format(np.mean(_means)))
print('数据集的BGR标准差为\n{}'.format(np.mean(_std)))
# 数据集的BGR平均值为(目前两个样本的情况)
# [247.81885295275427，247.81885295275427，247.81885295275427]
# 数据集的BGR标准差为
# [38.79730064571446，38.79730064571446，38.79730064571446]



# 247.82854836263738
# 数据集的BGR标准差为
# 38.773676267855215