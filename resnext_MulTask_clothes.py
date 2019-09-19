# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo # model_zoo是和导入预训练模型相关的包
from torch.utils.checkpoint import *
import torch.nn.functional as F
from torch.nn import init
import numpy as np


model_urls={
     'resnext50': 'http://ai2-vision.s3.amazonaws.com/elastic/imagenet_models/resnext50.pth.tar',
     'resnext101': 'https://dl.fbaipublicfiles.com/resnext/imagenet_models/resnext_101_32x4d.t7',
     'resnext50_elastic': 'http://ai2-vision.s3.amazonaws.com/elastic/imagenet_models/resnext50_elastic.pth.tar'
}

# ASPP模块
class ASPP(nn.Module): 
     # ASPP原论文里面，在给定的Input Feature Map上以r=(6,12,18,24)的3×3空洞卷积并行采样
    def __init__(self, C, depth, num_classes, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        super(ASPP, self).__init__()
        self._C = C # C/基数 
        self._depth = depth # 通道的维数
        self._num_classes = num_classes # 类别数
        self._norm = norm # nn.BatchNorm2d
        # 二维自适应平均池
        self.global_pooling = nn.AdaptiveAvgPool2d(1) # 返回 1x1 的池化结果
        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = nn.Conv2d(C, depth, kernel_size=1, stride=1, bias=False) # 普通1x1卷积;起到降维的作用
        self.aspp2 = nn.Conv2d(C, depth, kernel_size=3, stride=1,
                               dilation=int(6*mult), padding=int(6*mult),   # 空洞卷积的扩张率为6
                               bias=False)
        self.aspp3 = nn.Conv2d(C, depth, kernel_size=3, stride=1,
                               dilation=int(12*mult), padding=int(12*mult), # 空洞卷积的扩张率为12
                               bias=False)
        self.aspp4 = nn.Conv2d(C, depth, kernel_size=3, stride=1,
                               dilation=int(18*mult), padding=int(18*mult), # 空洞卷积的扩张率为18
                               bias=False)
        self.aspp5 = nn.Conv2d(C, depth, kernel_size=1, stride=1, bias=False) # 普通1x1卷积;起到降维的作用
        self.aspp1_bn = self._norm(depth, momentum) # BN操作
        self.aspp2_bn = self._norm(depth, momentum)
        self.aspp3_bn = self._norm(depth, momentum)
        self.aspp4_bn = self._norm(depth, momentum)
        self.aspp5_bn = self._norm(depth, momentum)
        self.conv2 = nn.Conv2d(depth * 5, depth, kernel_size=1, stride=1, bias=False) # 普通1x1卷积
        self.bn2 = self._norm(depth, momentum) # BN操作
        self.conv3 = nn.Conv2d(depth, num_classes, kernel_size=1, stride=1) # 普通1x1卷积

    def forward(self, x):
        x1 = self.aspp1(x)        # 普通分支
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)

        x2 = self.aspp2(x)        # 空洞卷积的扩张率为6
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)

        x3 = self.aspp3(x)        # 空洞卷积的扩张率为12
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)

        x4 = self.aspp4(x)        # 空洞卷积的扩张率为18
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)

        x5 = self.global_pooling(x) # 全局上下文的分支
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        # # 上采样
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)(x5) 

        x = torch.cat((x1, x2, x3, x4, x5), 1) # 沿着通道的维数拼接tensor
        x = self.conv2(x) # depth * 5 ==> depth，起到降维的作用
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x) # depth ==> num_classes，得到目标类别

        return x


# 分割模块
class Selayer(nn.Module):

    def __init__(self, inplanes):
        super(Selayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.global_avgpool(x)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out


# ResNext模块
class Block_ResNext(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2
    # 注意：在ResNext论文里面，cardinality(C)的取值为 32
    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False) # 1x1卷积
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False) # 3x3卷积
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False) # 1x1卷积
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

        # 一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数
        self.shortcut = nn.Sequential() # 旁路
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential( # self.shortcut操作
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False), # 1x1 卷积操作
                nn.BatchNorm2d(self.expansion*group_width)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # 1x1卷积
        out = F.relu(self.bn2(self.conv2(out))) # 3x3卷积
        out = self.bn3(self.conv3(out)) # 1x1卷积
        out += self.shortcut(x) # + 旁路
        out = F.relu(out)
        return out


# ResNext+Elastic模块
class BottleneckX(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None, dilation=1, norm=None, elastic=False, se=False):
        super(BottleneckX, self).__init__()
        self.se = se
        self.elastic = elastic and stride == 1 and planes < 512
        if self.elastic:
            self.down = nn.AvgPool2d(2, stride=2) # 下采样
            self.ups = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # 双线性上采样
        # half resolution
        # nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))
        self.conv1_d = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1_d = norm(planes)
        self.conv2_d = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, groups=cardinality // 2,  # 3x3卷积，分组卷积
                                 dilation=dilation, padding=dilation, bias=False)
        self.bn2_d = norm(planes)
        self.conv3_d = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)

        # full resolution
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, groups=cardinality // 2,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)

        # after merging
        self.bn3 = norm(planes * self.expansion)
        if self.se:
            self.selayer = Selayer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.__flops__ = 0

    def forward(self, x):
        residual = x
        out_d = x
        if self.elastic:
            if x.size(2) % 2 > 0 or x.size(3) % 2 > 0:
                out_d = F.pad(out_d, (0, x.size(3) % 2, 0, x.size(2) % 2), mode='replicate') # 填充
            out_d = self.down(out_d) # 下采样

        out_d = self.conv1_d(out_d) # 1x1卷积
        out_d = self.bn1_d(out_d)
        out_d = self.relu(out_d)

        out_d = self.conv2_d(out_d) # 3x3卷积
        out_d = self.bn2_d(out_d)
        out_d = self.relu(out_d)

        out_d = self.conv3_d(out_d) # 1x1卷积

        if self.elastic:
            out_d = self.ups(out_d) # 上采样
            self.__flops__ += np.prod(out_d[0].shape) * 8
            if out_d.size(2) > x.size(2) or out_d.size(3) > x.size(3):
                out_d = out_d[:, :, :x.size(2), :x.size(3)]

        # 普通卷积部分
        out = self.conv1(x)   # 1x1卷积
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # 3x3卷积
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) # 1x1卷积

        out = out + out_d # 普通卷积部分 + Elastic卷积部分
        out = self.bn3(out)

        if self.se:
            out = self.selayer(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual # 旁路
        out = self.relu(out)

        return out


# 函数入口
class ResNext(nn.Module):

    def __init__(self, block, layers, num_classes=1000, seg=False, elastic=False, se=False):
        self.inplanes = 64
        self.cardinality = 32 # C/基数
        self.seg = seg # 分割
        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #    num_features: 一般输入参数为batch_size*num_features*height*width，即为其中特征的数量
        #    eps：分母中添加的一个值，目的是为了计算的稳定性，默认为：1e-5
        #    momentum：一个用于运行过程中均值和方差的一个估计参数（我的理解是一个稳定系数，类似于SGD中的momentum的系数）
        #    affine：当设为true时，会给定可以学习的系数矩阵gamma和beta
        self._norm = lambda planes, momentum=0.05 if seg else 0.1: torch.nn.BatchNorm2d(planes, momentum=momentum)
        super(ResNext, self).__init__()
        """ 网络参数的定义部分 """
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # 7x7x64的卷积
        self.bn1 = self._norm(64) # 正则化
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 3x3的池化
        # 例如: layers=[6, 8, 5, 3]; elastic=True
        self.layer1 = self._make_layer(block, 64, layers[0], elastic=elastic, se=se) # 加入elastic结构
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, elastic=elastic, se=se) # 加入elastic结构
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, elastic=elastic, se=se) # 加入elastic结构
        """ 原程序
        if seg:
            self.layer4 = self._make_mg(block, 512, se=se)
            self.aspp = ASPP(512 * block.expansion, 256, num_classes, self._norm)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, torch.nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        else: 
        """
        # 非分割部分
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, elastic=False, se=se) # 最后一部分没有使用elastic结构
        self.avgpool = nn.AdaptiveAvgPool2d(1) # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(1) # 全局平均池化
#        self.fc = nn.Linear(512 * block.expansion, num_classes) # 全连接层
#        init.normal_(self.fc.weight, std=0.01) # 全连接层weight初始化
        # 全连接层
        self.fc1 = nn.Linear(512 * block.expansion, num_classes[0])
        init.normal_(self.fc1.weight, std=0.01)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes[1])
        init.normal_(self.fc2.weight, std=0.01)

#        self.fc = nn.Linear(512 * block.expansion, num_classes) # 全连接层
#        init.normal_(self.fc.weight, std=0.01) # 可能有问题

#        self.linear = nn.ModuleList([nn.Sequential(
#            nn.Linear(512 * block.expansion, num_classes[0]),
#            nn.Softmax(dim=1))])


#            for n, p in self.named_parameters():
#                if n.split('.')[-1] == 'weight':
#                    if 'conv' in n:
                        # kaiming高斯初始化===> 目的是使得每一卷积层的输出的方差都为1
                        # a为Relu函数的负半轴斜率，mode表示是让前向传播还是反向传播的输出的方差为1，nonlinearity可以选择是relu还是leaky_relu
#                        init.kaiming_normal_(p, mode='fan_in', nonlinearity='relu')
#                    if 'bn' in n:
#                        p.data.fill_(1)
#                    if 'bn3' in n:
#                        p.data.fill_(0)
#                elif n.split('.')[-1] == 'bias':
#                    p.data.fill_(0)

##############################################################################################
###多任务 构成的是一个 2x5x3 的 Attention Module
        filter = [64, 256, 512, 1024, 2048]
        # 注意力模块
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])]) # g函数——>h函数 
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])]) # f函数    # conv_layer([数组])

        # j表示第j个任务，i表示共享网络的第i个块的共享特征
        for j in range(2): # 0-1 即2个任务
            if j < 1: # 上面已经有一个，即 0
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
#                self.fc.append(nn.Linear(512 * block.expansion, num_classes[j + 1])) # 之后的全连接层  nn.Linear(filter[3], num_classes[j + 1])
            for i in range(4): # 3 # 0-1-2--每个任务有4个 Attention Module,即后面接3个注意力模块
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))

        for i in range(4): # 0-1-2-3
            if i < 3: # 0-1-2
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
            else: # 3
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

        #参数化
        for n, p in self.named_parameters():
            if n.split('.')[-1] == 'weight':
               if 'conv' in n:
                   # kaiming高斯初始化===> 目的是使得每一卷积层的输出的方差都为1
                   # a为Relu函数的负半轴斜率，mode表示是让前向传播还是反向传播的输出的方差为1，nonlinearity可以选择是relu还是leaky_relu
                   init.kaiming_normal_(p, mode='fan_in', nonlinearity='relu')
               if 'bn' in n:
                   p.data.fill_(1)
               if 'bn3' in n:
                   p.data.fill_(0)
            elif n.split('.')[-1] == 'bias':
               p.data.fill_(0)


    # 注意力部分函数--f函数
    def conv_layer(self, channel):
        conv_block = nn.Sequential(
            # f函数
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0), # 3x3卷积 # 3  1
            nn.BatchNorm2d(num_features=channel[1]), # BN操作
            nn.ReLU(inplace=True), # ReLU
        )
        return conv_block

    # 注意力部分函数--g函数--h函数
    def att_layer(self, channel):
        att_block = nn.Sequential(
            # g函数
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0), # 1x1卷积
            nn.BatchNorm2d(channel[1]), # BN操作
            nn.ReLU(inplace=True), # ReLU
            # g函数
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0), # 1x1卷积
            nn.BatchNorm2d(channel[2]), # BN操作
            nn.Sigmoid(),  # Sigmoid
        )
        return att_block

##############################################################################################

    def _make_layer(self, block, planes, blocks, stride=1, elastic=False, se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self._norm(planes * block.expansion),
            )

        layers = list()
        # 该部分是将每个blocks的第一个residual结构保存在layers列表中
        layers.append(block(self.inplanes, planes, self.cardinality, stride, downsample=downsample, norm=self._norm, elastic=elastic, se=se))
        self.inplanes = planes * block.expansion
        # 该部分是将每个blocks的剩下residual 结构保存在layers列表中，这样就完成了一个blocks的构造
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, norm=self._norm, elastic=elastic, se=se))
        return nn.Sequential(*layers)

    # 分割部分的函数
    def _make_mg(self, block, planes, dilation=2, multi_grid=(1, 2, 4), se=False):
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=1, dilation=1, bias=False),
            self._norm(planes * block.expansion),
        )

        layers = list()
        layers.append(block(self.inplanes, planes, self.cardinality, downsample=downsample, dilation=dilation*multi_grid[0], norm=self._norm, se=se))
        self.inplanes = planes * block.expansion
        layers.append(block(self.inplanes, planes, self.cardinality, dilation=dilation*multi_grid[1], norm=self._norm, se=se))
        layers.append(block(self.inplanes, planes, self.cardinality, dilation=dilation*multi_grid[2], norm=self._norm, se=se))
        return nn.Sequential(*layers)


    def forward(self, x):
        size = (x.shape[2], x.shape[3]) # 图像的尺寸
        """ con1部分 ---原论文
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.seg:
            for module in self.layer1._modules.values():
                x = checkpoint(module, x)
            for module in self.layer2._modules.values():
                x = checkpoint(module, x)
            for module in self.layer3._modules.values():
                x = checkpoint(module, x)
            for module in self.layer4._modules.values():
                x = checkpoint(module, x)
            x = self.aspp(x)
            x = nn.Upsample(size, mode='bilinear', align_corners=True)(x)
        else: # 非分割部分
            x = self.layer1(x) # 有elastic结构
            x = self.layer2(x) # 有elastic结构
            x = self.layer3(x) # 有elastic结构
            x = self.layer4(x) # 没有elastic结构
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x) # 全连接部分
        return x
        """
        # 共享特征
        g_encoder = [0] * 5  # [0, 0, 0, 0]
        # 注意力编码器
        atten_encoder = [0] * 2 # [0, 0]
        for i in range(2):
            atten_encoder[i] = [0] * 5
        for i in range(2):
            for j in range(5):
                atten_encoder[i][j] = [0] * 3
#        print("atten_encoder: ", np.shape(atten_encoder))
        # 最终atten_encoder的结构: (2, 5, 3)

        x = self.conv1(x) # conv7x7
#        print("x:", np.shape(x)) # [1, 64, 112, 112]
        g_encoder[0]=x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
#        print("x:", np.shape(x)) # x: torch.Size([1, 64, 56, 56])
#        g_encoder[0]=x

        g_encoder[1] = self.layer1(x) # 有elastic结构
#        print("g_encoder[1]:", np.shape(g_encoder[1])) # [1, 256, 56, 56]
        g_encoder[2] = self.layer2(g_encoder[1]) # 有elastic结构
#        print("g_encoder[2]:", np.shape(g_encoder[2])) # [1, 512, 28, 28]
        g_encoder[3] = self.layer3(g_encoder[2]) # 有elastic结构
#        print("g_encoder[3]:", np.shape(g_encoder[3])) # [1, 1024, 14, 14]
        g_encoder[4] = self.layer4(g_encoder[3]) # 没有elastic结构
#        print("g_encoder[4]:", np.shape(g_encoder[4])) # [1, 2048, 7, 7]

#        x = self.avgpool(x)
#        x = x.view(x.size(0), -1)
#        x = self.fc(x) # 全连接部分


#        g_encoder[1] = self.layer1(g_encoder[0])
#        g_encoder[2] = self.layer2(g_encoder[1])
#        g_encoder[3] = F.relu(self.bn1(self.layer3(g_encoder[2]))) # BN+ReLU操作

        # 应用注意力模块：(2, 5, 3)
        pred=[]
        for k in range(2): # k=2
          for j in range(5):
            if j == 0: # k表示第k个任务，j表示第k个任务中第j个Attention Module
                atten_encoder[k][j][0] = self.encoder_att[k][j](g_encoder[0]) # g函数——>h函数
                atten_encoder[k][j][1] = (atten_encoder[k][j][0]) * g_encoder[0] # P操作---元素相乘
                atten_encoder[k][j][2] = self.encoder_block_att[j](atten_encoder[k][j][1]) # f函数
                atten_encoder[k][j][2] = F.max_pool2d(atten_encoder[k][j][2], kernel_size=2, stride=2) # Attention模块的pool操作
            else:  # 添加 u操作
                atten_encoder[k][j][0] = self.encoder_att[k][j](torch.cat((g_encoder[j], atten_encoder[k][j - 1][2]), dim=1)) # u操作——>g函数——>h函数
                atten_encoder[k][j][1] = (atten_encoder[k][j][0]) * g_encoder[j] # P操作---元素相乘
                atten_encoder[k][j][2] = self.encoder_block_att[j](atten_encoder[k][j][1]) # f函数
                if j < 4: # j=1,2,3
                    atten_encoder[k][j][2] = F.max_pool2d(atten_encoder[k][j][2], kernel_size=2, stride=2) # Attention模块的pool操作
          # 池化
          temp= self.avgpool(atten_encoder[k][-1][-1])
          pred.append(temp)

        # 本体
        pred[0] = pred[0].view(pred[0].size(0), -1)
#        print("pred[0]: ", np.shape(pred[0])) # torch.Size([1, 2048])
        out1 = self.fc1(pred[0]) # 全连接层
        # 隐义
        pred[1] = pred[1].view(pred[1].size(0), -1)
#        print("pred[1]: ", np.shape(pred[1])) # torch.Size([1, 2048])
        out2 = self.fc2(pred[1]) # 全连接层
#       out[k] = self.linear[k](pred[k])

#        x = self.avgpool(atten_encoder[k][-1][-1])
#        x = x.view(x.size(0), -1)
#        x = self.fc(x) # 全连接部分


#        pred = F.avg_pool2d(atten_encoder[k][-1][-1], 8) # 平均池化操作
#        pred0 = self.avgpool(atten_encoder[0][-1][-1]) # atten_encoder[k][-1][-1]
#        pred0 = pred0.view(pred0.size(0), -1) # 为了将前面多维度的tensor展平成一维
#        out0 = self.fc(pred0) # 全连接层

#        pre1 = self.avgpool(atten_encoder[1][-1][-1]) # atten_encoder[k][-1][-1]
#        pred1 = pred1.view(pred1.size(0), -1) # 为了将前面多维度的tensor展平成一维
#        out1 = self.linear[1](pred1) # 全连接层

        return out1, out2

def resnext50(seg=False, **kwargs): # 普通的resnext50结构
    print("使用resnext50结果...")
    model = ResNext(BottleneckX, [3, 4, 6, 3], seg=seg, elastic=False, **kwargs)
    return model


def resnext50_elastic(seg=False, **kwargs):  # 实验所使用的结构 resnext50_elastic
    print("使用resnext50_elastic结果...")
    model = ResNext(BottleneckX, [6, 8, 5, 3], seg=seg, elastic=True, **kwargs)
    return model


def se_resnext50(seg=False, **kwargs):
    model = ResNext(BottleneckX, [3, 4, 6, 3], seg=seg, elastic=False, se=True, **kwargs)
    return model


def se_resnext50_elastic(seg=False, **kwargs):
    model = ResNext(BottleneckX, [6, 8, 5, 3], seg=seg, elastic=True, se=True, **kwargs)
    return model


def resnext101(seg=False, **kwargs):
    model = ResNext(BottleneckX, [3, 4, 23, 3], seg=seg, elastic=False, **kwargs)
    return model


def resnext101_elastic(seg=False, **kwargs):
    model = ResNext(BottleneckX, [12, 14, 20, 3], seg=seg, elastic=True, **kwargs)
    return model






