import torch
import torch.nn as nn
from torchsummary import summary
import config
from torch.utils.tensorboard import SummaryWriter
# 我们图片的格式是448 * 224 * 1
class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self):
        super(C3D, self).__init__()
        # torch.Size([b, 1, 16, 448, 224])
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

        self.fc6 = nn.Linear(512 * 3 * 7, 4096)
        self.fc7 = nn.Linear(4096, 256)
        self.fc8 = nn.Linear(256, 32)
        self.fc9 = nn.Linear(32, 1)

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
#     inputs = torch.rand(4, 1, 16, 448, 224).to(config.device)
#     net = C3D().to(config.device)
#     outputs = net(inputs)
#     print(outputs.size())
#     summary(model = net,input_size = (1,16,448,224))


# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv3d-1     [-1, 32, 16, 448, 224]             896
#               ReLU-2     [-1, 32, 16, 448, 224]               0
#          MaxPool3d-3     [-1, 32, 16, 224, 112]               0
#             Conv3d-4     [-1, 64, 16, 224, 112]          55,360
#               ReLU-5     [-1, 64, 16, 224, 112]               0
#          MaxPool3d-6       [-1, 64, 8, 112, 56]               0
#             Conv3d-7      [-1, 128, 8, 112, 56]         221,312
#               ReLU-8      [-1, 128, 8, 112, 56]               0
#          MaxPool3d-9       [-1, 128, 4, 56, 28]               0
#            Conv3d-10       [-1, 256, 4, 56, 28]         884,992
#              ReLU-11       [-1, 256, 4, 56, 28]               0
#            Conv3d-12       [-1, 256, 4, 56, 28]       1,769,728
#              ReLU-13       [-1, 256, 4, 56, 28]               0
#         MaxPool3d-14       [-1, 256, 2, 28, 14]               0
#            Conv3d-15       [-1, 512, 2, 28, 14]       3,539,456
#              ReLU-16       [-1, 512, 2, 28, 14]               0
#            Conv3d-17       [-1, 512, 2, 28, 14]       7,078,400
#              ReLU-18       [-1, 512, 2, 28, 14]               0
#         MaxPool3d-19        [-1, 512, 1, 14, 7]               0
#            Conv3d-20        [-1, 512, 1, 14, 7]       7,078,400
#              ReLU-21        [-1, 512, 1, 14, 7]               0
#            Conv3d-22        [-1, 512, 1, 14, 7]       7,078,400
#              ReLU-23        [-1, 512, 1, 14, 7]               0
#         MaxPool3d-24         [-1, 512, 1, 7, 3]               0
#            Linear-25                 [-1, 8192]      88,088,576
#              ReLU-26                 [-1, 8192]               0
#           Dropout-27                 [-1, 8192]               0
#            Linear-28                 [-1, 4096]      33,558,528
#              ReLU-29                 [-1, 4096]               0
#           Dropout-30                 [-1, 4096]               0
#            Linear-31                  [-1, 256]       1,048,832
#              ReLU-32                  [-1, 256]               0
#           Dropout-33                  [-1, 256]               0
#            Linear-34                    [-1, 1]             257
# ================================================================