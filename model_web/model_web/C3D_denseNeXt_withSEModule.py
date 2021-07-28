import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pickle
 # 1.====================导库
# densenet的网络架构：分为三个部分，1：头部迅速降低分辨率；2：多个denseblock的堆叠
# （每个denseblock中堆叠denselayer,每个denselayer（bottlenecklayer）都是由1*1卷积后先生成4k最终生成k个特征图再与前面堆叠）
# 体现了shortPath with 特征复用 3:经过池化操作最终输出
# （transition组件降低特征图通道数以及池化降低分辨率）block内部不允许分辨率下降
fen = [] # 用于可视化分组卷积的东西 如果不需要 注释掉forward_super中的append

class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      groups=16,
                      padding=1,
                      bias=False))
        self.add_module('norm3', nn.BatchNorm3d(growth_rate))
        self.add_module('relu3', nn.ReLU(inplace=True))
        self.add_module(
            'conv3',
            nn.Conv3d(growth_rate,
                      growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('seModule', SELayer(32, 8))
        self.drop_rate = drop_rate

    def forward_super(self, i):
        for module in self._modules.values():
            i = module(i)
            if isinstance(module, torch.nn.modules.conv.Conv3d):

                fen.append(i)
                # print(module)
        return i
    # 这里有一个需要思考的地方 就是我们能不能把空间分离的那种卷积也加入进来？
    def forward(self, x):
        new_features = self.forward_super(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)

# 或许我们遇到了更好的网络 再考虑轻量化的在每一个分组卷积那里都加入长方形的卷积单元
# 加入一个卷积层 不要忘了对它做bn和relu
# 我们可以做纵向和横向的模型融合

 # 2.====================DenseLayer模块 加入分组卷积

class _DenseBlock(nn.Sequential):
    # 这些都是module的容器 不必写forward 方法
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer{}'.format(i + 1), layer)
# 2.====================DenseBlock模块
class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
#         channel=32 (b,32,t,h,w) => (b,32,1,1,1)
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        """
        We found empirically that on ResNet
        architectures, removing the biases of the FC layers in the
        excitation operation facilitates the modelling of channel
        dependencies
        """

    def forward(self, x):
        b, c, _, _,_ = x.size()
        y = self.avg_pool(x).view(b, c)  #  batchsize input_num

        y = self.fc(y)
        # print(y)
        y = y.view(b, c, 1, 1, 1)
        return x * y.expand_as(x)
# 3.====================Transition模块以及SE模块
#DenseNet(num_init_features=64,
#                          growth_rate=32,
#                          block_config=(6, 12, 24, 16),
#                          **kwargs)
class DenseNet(nn.Module):
    def __init__(self,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=4):

        super().__init__()

        # First convolution
        self.features = [('conv1',
                          nn.Conv3d(n_input_channels,
                                    num_init_features,
                                    kernel_size=(conv1_t_size, 7, 7),
                                    stride=(conv1_t_stride, 2, 2),
                                    padding=(conv1_t_size // 2, 3, 3),
                                    bias=False)),
                         ('norm1', nn.BatchNorm3d(num_init_features)),
                         ('relu1', nn.ReLU(inplace=True))]
        if not no_max_pool:
            self.features.append(
                ('pool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)))
        self.features = nn.Sequential(OrderedDict(self.features))
# 使用OrderedDict会根据放入元素的先后顺序进行排序
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        #自适应的池化操作
        out = F.adaptive_avg_pool3d(out,
                                    output_size=(1, 1,
                                                 1)).view(features.size(0), -1)
        out = self.classifier(out)
        # with open(r'D:\modelling/fenzu/fen12.json', 'wb') as file1:
        #     pickle.dump(fen,file1)
        return out
# =========================== denseNet 模块
# 记录一次：5.70MB(三者都有)

# densenet 17.02MB

# densenet 加上分组卷积 5.67MB

# 纯的只加上se17.05MB
