# 在这里 退而求其次 做分类模型 在这里 需要对时间进行分箱 在这里需要考虑文件夹命名是否出现重复

# 因为hash加文件名的原因 不会出现重复情况
# hash保证一个文件夹里面的记录到相同的的label_的时候不会出现重复 文件名保证即使有相同的hash也是不同的文件夹
import torch
import torch.nn as nn
import config
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
# 我们图片的格式是448 * 224 * 1
# 这是一个糟糕的网络架构
class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # 第一层卷积先不要折减图片数量  [b,1,16,448,224] -> [b,32,16,224,112]
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # [b,32,16,224,112] -> [b,64,8,112,56]
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # [b,64,8,112,56] -> [b,128,4,56,28]
        self.conv4a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # # [b,128,4,56,28] -> [b,256,2,28,14]
        self.conv5a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # [b,256,2,28,14] -> [b,512,1,14,7]
        self.conv6a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv6b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool6 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # [b,512,1,14,7] -> [b,512,1,7,3]

        self.fc6 = nn.Linear(512 * 3 * 7, 2048)
        self.fc7 = nn.Linear(2048, 256)
        self.fc8 = nn.Linear(256, 32)
        self.fc9 = nn.Linear(32, 4)

        # 加上batchnormlization 之后 发现要计算的太多了  不太好
        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x):
        # print ('1:',x.size())
        x = self.relu(self.conv1(x))
        # print ('2:',x.size())
        x = self.pool1(x)
        # print ('3:',x.size())

        x = self.relu(self.conv2(x))
        # print ('4:',x.size())
        x = self.pool2(x)
        # print ('5:',x.size())
        x = self.relu(self.conv3(x))
        # print('4_:', x.size())
        x = self.pool3(x)
        # print('5_:', x.size())
        x = self.relu(self.conv4a(x))
        # print ('6:',x.size())
        x = self.relu(self.conv4b(x))
        # print ('7:',x.size())
        x = self.pool4(x)
        # print ('8:',x.size())

        x = self.relu(self.conv5a(x))
        # print ('9:',x.size())
        x = self.relu(self.conv5b(x))
        # print ('10:',x.size())
        x = self.pool5(x)
        # print ('11:',x.size())
        x = self.relu(self.conv6a(x))
        # print('9_:', x.size())
        x = self.relu(self.conv6b(x))
        # print('10_:', x.size())
        x = self.pool6(x)
        # print('11_:', x.size())

        x = x.view(-1, 512 * 3 * 7)
        # print ('15:',x.size())
        x = self.relu(self.fc6(x))
        # print ('16:',x.size())
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        # print('17:', x.size())
        x = self.dropout(x)

        x = self.relu(self.fc8(x))
        x = self.dropout(x)
        logits = self.fc9(x)
        # print ('18:',logits.size())
        return logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# def get_1x_lr_params(model):
#     """
#     This generator returns all the parameters for conv and two fc layers of the net.
#     """
#     b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
#          model.fc6, model.fc7,model.fc8]
#     for i in range(len(b)):
#         for k in b[i].parameters():
#             if k.requires_grad:
#                 yield k
#
# def get_10x_lr_params(model):
#     """
#     This generator returns all the parameters for the last fc layer of the net.
#     """
#     b = [model.fc9]
#     for j in range(len(b)):
#         for k in b[j].parameters():
#             if k.requires_grad:
#                 yield k

# if __name__ == "__main__":
#     # inputs = torch.rand(4, 1, 16, 448, 224).to(config.device)
#     net = C3D().to(config.device)
#     # outputs = net(inputs)
#     # # with SummaryWriter('runs/exp') as w:
#     # #     w.add_graph(net,(inputs,))
#     # print(outputs.size())
#     summary(model=net,input_size=(1,16,448,224))




    # 神经自架构为网络减负