# 训练
import os
import warnings
from tools.common_tools import ModelTrainer
from datetime import datetime
import numpy as np
from tqdm import tqdm
import config
import torch
from torch import optim
from process.dataloader_discrete import train_loader,test_loader
from model.model_all_with_regression import DenseNet
import pickle
from sklearn.preprocessing import StandardScaler as ss
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    model = DenseNet(n_input_channels=1, num_init_features=64,
                           growth_rate=32,
                           block_config=(3, 6, 12, 8), num_classes=286).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # ###########################
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load('D://modelling//checkpoint_classify//model29.pkl')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in config.weights_copy_key}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    model.load_state_dict(torch.load("D://modelling//checkpoint_discrete//model63.pkl"))
    # ###########################
    mat_list_ = []
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    with open(r'D:\modelling/files_discrete/acc_classify_.json', 'rb') as file1:
        acc_rec = pickle.load(file1)
    with open(r'D:\modelling/files_discrete/loss_classify_.json', 'rb') as file2:
        loss_rec = pickle.load(file2)
    with open(r'D:\modelling/files_discrete/mat_classify_.json', 'rb') as file3:
        mat_list_ = pickle.load(file3)
    # ============================
    loss_function = torch.nn.CrossEntropyLoss() #
    loss_function.to(config.device)

    modelName = 'DenseNet'  # Options: C3D or R2Plus1D or R3D

    # print("Device being used:", config.device)
    # TODO 需要修改的地方就是加载的模型 每运行一次都要进行修改 还有train/epoch

    best_acc = 0
    best_epoch = 0
    # 配置超参数
    num_classes = 286
    MAX_EPOCH = 300  # 参考论文中 4.2 Training
    BATCH_SIZE = 64  # 参考论文中 4.2 Training
    log_interval = 1
    val_interval = 1
    start_epoch = -1
    # class_names = ("none", "yellow", "orange", "red")
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(config.current_file, "results_discrete", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # print(log_dir)

    for epoch in range(64, 80):
        # 走过多少个epoch,就从几开始，接下来就是7
        # 训练(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch)
        loss_train, acc_train, mat_train = ModelTrainer.train(train_loader, model, loss_function, optimizer, epoch,config.device, MAX_EPOCH)
        loss_valid, acc_valid, mat_valid = ModelTrainer.valid(test_loader, model, loss_function, config.device)
        # 统计
        mat_list_.append((mat_train, mat_valid))
        # 统计用于绘图
        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}".format(epoch + 1, MAX_EPOCH, acc_train, acc_valid, loss_train, loss_valid, optimizer.param_groups[0]["lr"]))
        # 看一下学习率是多少
        # optimizer.param_groups[0]：长度为6的字典，包括[‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]这6个参数

        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)
        if (epoch % 1 == 0) or (best_acc < max(acc_rec["valid"])):
            best_acc = max(acc_rec["valid"])
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_acc": best_acc}
            path_checkpoint = os.path.join(log_dir, "checkpoint_best{}.pkl".format(epoch))
            torch.save(checkpoint, path_checkpoint)
            # 注意 checkpoint 是给不加SE的，checkpoint_classify给了SE
            torch.save(model.state_dict(), "D:\modelling/checkpoint_discrete/model{}.pkl".format(epoch))
            torch.save(optimizer.state_dict(), "D:\modelling/checkpoint_discrete/optimizer{}.pkl".format(epoch))

    print(" done ~~~~ {}, best acc: {} in :{} epochs. ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M'),best_acc, best_epoch))
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)
    # 考虑如何执行这段代码
    #     1：首先不要设置过多的epoch,训练几个回合
    #     2：训练完成以后，保存loss和acc
    #     3：在下次训练回合之前，读入之前保存的loss,acc，并且读入pkl文件
    with open(r'D:\modelling/files_discrete/acc_classify_.json', 'wb') as file1:
        pickle.dump(acc_rec, file=file1)  # wb 覆盖原先文件
    with open(r'D:\modelling/files_discrete/loss_classify_.json', 'wb') as file2:
        pickle.dump(loss_rec, file=file2)
    with open(r'D:\modelling/files_discrete/mat_classify_.json', 'wb') as file3:
        pickle.dump(mat_list_, file=file3)


