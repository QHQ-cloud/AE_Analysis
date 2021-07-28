# 计算自己图片数据集的均值 方便 pytorch 中去均值 处理
import os
import numpy as np
import cv2  # opencv的图像处理格式是BGR
img_list = []
# 一般情况下 我们需要对训练集做处理 求均值 带入测试集中
ims_root_path = r'D:\modelling\dataset\train'  # 图像数据集的路径
for ims_path in os.listdir(ims_root_path):  # label_0
    ims_list = os.listdir(os.path.join(ims_root_path,ims_path))  #AEPic_1354_8x_285
    for img_list_16 in ims_list:  # 'AEPic_1354_10z_280'
        imgx_list = os.listdir(os.path.join(ims_root_path,ims_path,img_list_16))
        for img in imgx_list:
            img_list.append(os.path.join(ims_root_path,ims_path,img_list_16,img))
# ['F:\\capture\\pic\\AEPic-185\\0037-0001.jpg',
#  'F:\\capture\\pic\\AEPic-185\\0038-0001.jpg',
#  'F:\\capture\\pic\\AEPic-185\\0039-0001.jpg',
#  'F:\\capture\\pic\\AEPic-185\\0040-0001.jpg',
#  'F:\\capture\\pic\\AEPic-185\\0041-0001.jpg',
#  'F:\\capture\\pic\\AEPic-185\\0042-0001.jpg',
#  'F:\\capture\\pic\\AEPic-185\\0043-0001.jpg',

# 在这里得到的均值是训练集所有图片的均值
# 用来计算均值的，一定要用训练集，否则违背了深度学习的原则（即模型训练仅能从训练数据中获取信息）。对于得到的mean值，训练集、验证集和测试集都要分别减去
B_means = []
G_means = []
R_means = []
B_std = []
G_std = []
R_std = []
# 还没有对图片进行crop操作   不能crop  只能resize
for img in img_list:
    im = cv2.imread(img)  # [673,1118,3]
    # extract value of diffient channel
    # im = im[50:50 + 572,274: 274 + 572,:]  # [572, 572, 3]
    im_B = im[:, :, 0]  # [572,572]
    im_G = im[:, :, 1]
    im_R = im[:, :, 2]
    # count mean for every channel
    im_B_mean = np.mean(im_B)
    im_G_mean = np.mean(im_G)
    im_R_mean = np.mean(im_R)
    im_B_std = np.std(im_B)
    im_G_std = np.std(im_G)
    im_R_std = np.std(im_R)
    # save single mean value to a set of means
    B_means.append(im_B_mean)
    G_means.append(im_G_mean)
    R_means.append(im_R_mean)
    B_std.append(im_B_std)
    G_std.append(im_G_std)
    R_std.append(im_R_std)
# three sets  into a large set
mean_pre = [B_means, G_means, R_means]
std_pre = [B_std,G_std,R_std]
mean = [0, 0, 0] # 创建一个缓存 保存最终输出
std = [0, 0, 0]
# count the sum of different channel means
mean[0] = np.mean(mean_pre[0])
mean[1] = np.mean(mean_pre[1])
mean[2] = np.mean(mean_pre[2])
std[0] = np.mean(std_pre[0])
std[1] = np.mean(std_pre[1])
std[2] = np.mean(std_pre[2])
print('数据集的BGR平均值为\n[{}，{}，{}]'.format(mean[0], mean[1], mean[2]))
print('数据集的BGR标准差为\n[{}，{}，{}]'.format(std[0], std[1], std[2]))
# 数据集的BGR平均值为(目前两个样本的情况)
# [247.81885295275427，247.81885295275427，247.81885295275427]
# 数据集的BGR标准差为
# [38.79730064571446，38.79730064571446，38.79730064571446]
# 灰度图也