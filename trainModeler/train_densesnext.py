# 训练
from datetime import datetime
import os
import pickle
import numpy as np
import config
import torch
from torch import optim,nn
from process.dataloader_classify import train_loader,train_data
from process.dataloader_classify import test_loader,test_data
from model.C3D_denseNeXt import DenseNet
from tools.common_tools import show_confMat,plot_line,ModelTrainer
# 由于训练不能一次性完成，所以选择加载模型文件的方式分步训练

if __name__ == '__main__':
    # ===========================
    mat_list_ = []
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    with open(r'D:\modelling/files/acc_classify_.json', 'rb') as file1:
        acc_rec = pickle.load(file1)
    with open(r'D:\modelling/files/loss_classify_.json', 'rb') as file2:
        loss_rec = pickle.load(file2)
    with open(r'D:\modelling/files/mat_classify_.json', 'rb') as file3:
        mat_list_ = pickle.load(file3)


    # ============================
    model = DenseNet(n_input_channels=1, num_init_features=64,
                       growth_rate=32,
                       block_config=(3, 6, 12, 8), num_classes=4).to(config.device)
    # adam优化器加上动量吃显存 densenet 本身就吃显存
    # 在前几个训练过程中观察到好像有振荡的行为
    optimizer = optim.SGD(params=model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)  # 选择优化器
    # optimizer = optim.Adam(model.parameters(), lr=config.lr,weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=config.milestones)

    model.load_state_dict(torch.load('D:\modelling/checkpoint/model23.pkl'))

    optimizer.load_state_dict(torch.load('D:\modelling/checkpoint/optimizer23.pkl'))

    # todo 训练的时候不一定加载最好的，但是测试要加载最好的
    # todo 我们加载进入模型的那几个（字典）列表可能有重复
    # todo 中间略过了个回合（训练时，先一个回合，中午6个回合，下午两个回合，这两个回合相当于中午那三个567回合白干了）对应于列表里的456到时候删除即可
    # todo 发现准确率在之前总是忽高忽低，这可能是由于我们学习率设置太大的缘故
    # todo 一般batch_size大的话，学习率才设置大一些
    # 1702 777 1150 855 这个是平衡比例后的权重设计（0.38 + 0.17 + 0.26 + 0.19）
    # 为了增大红色预警召回率，增大后面。
    loss_function = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.19,0.17,0.26,0.19 * 2])).float())# 在这里加上weight
    loss_function.to(config.device)

    modelName = 'DenseNet' # Options: C3D or R2Plus1D or R3D

    # print("Device being used:", config.device)
    # TODO 需要修改的地方就是加载的模型 每运行一次都要进行修改 还有train/epoch

    best_acc =  max(acc_rec["valid"])
    best_epoch = 0
    # 配置超参数
    num_classes = 4
    MAX_EPOCH = 300    #  参考论文中 4.2 Training
    BATCH_SIZE = 64    #  参考论文中 4.2 Training
    log_interval = 1
    val_interval = 1
    start_epoch = -1
    class_names = ("none","yellow","orange","red")
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(config.current_file, "results", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # print(log_dir)
    for epoch in range(24, 30):
        # 走过多少个epoch,就从几开始，接下来就是7
        # 训练(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch)
        loss_train, acc_train, mat_train = ModelTrainer.train(train_loader, model, loss_function, optimizer, epoch, config.device, MAX_EPOCH)
        loss_valid, acc_valid, mat_valid = ModelTrainer.valid(test_loader, model, loss_function, config.device)
        # 统计
        mat_list_.append((mat_train,mat_valid))
        # 统计用于绘图
        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}".format(
            epoch + 1, MAX_EPOCH, acc_train, acc_valid, loss_train, loss_valid, optimizer.param_groups[0]["lr"]))
        # 看一下学习率是多少
        # optimizer.param_groups[0]：长度为6的字典，包括[‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]这6个参数
        scheduler.step()  # 更新学习率

        # 绘图
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)

        # show_confMat(mat_train, class_names, "train", log_dir, verbose=epoch == MAX_EPOCH-1)
        # show_confMat(mat_valid, class_names, "valid", log_dir, verbose=epoch == MAX_EPOCH-1)
        #
        # plt_x = np.arange(1, epoch+2)
        # plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        # plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)

        if (epoch % 1 == 0) or (best_acc < max(acc_rec["valid"])):
            best_acc = max(acc_rec["valid"])
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_acc": best_acc}
            path_checkpoint = os.path.join(log_dir, "checkpoint_best{}.pkl".format(epoch))
            torch.save(checkpoint, path_checkpoint)
            torch.save(model.state_dict(), "D:\modelling/checkpoint/model{}.pkl".format(epoch))
            torch.save(optimizer.state_dict(),"D:\modelling/checkpoint/optimizer{}.pkl".format(epoch))

    print(" done ~~~~ {}, best acc: {} in :{} epochs. ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                                                      best_acc, best_epoch))
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)
# 考虑如何执行这段代码
#     1：首先不要设置过多的epoch,训练几个回合
#     2：训练完成以后，保存loss和acc
#     3：在下次训练回合之前，读入之前保存的loss,acc，并且读入pkl文件
    with open(r'D:\modelling/files/acc_classify_.json','wb') as file1:
        pickle.dump(acc_rec,file=file1)  # wb 覆盖原先文件
    with open(r'D:\modelling/files/loss_classify_.json', 'wb') as file2:
        pickle.dump(loss_rec, file=file2)
    with open(r'D:\modelling/files/mat_classify_.json', 'wb') as file3:
        pickle.dump(mat_list_, file=file3)
# todo  建立一个map字典 以表示对应关系
# {'label_1': 0, 'label_2': 1, 'label_3': 2, 'label_4': 3}


#  ##########################################################
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
#  ##########################################################