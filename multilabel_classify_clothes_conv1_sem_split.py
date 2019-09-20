# -*- coding: utf-8 -*-
import argparse
import time
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
import models # 文件夹里面有__init__.py文件可以导入相应的模型
import os
from PIL import Image
from utils import add_flops_counting_methods, save_checkpoint, AverageMeter
from torch.utils.data import Dataset

# 可选模型
model_names = ['resnext50', 'resnext50_elastic', 'resnext101', 'resnext101_elastic',
               'dla60x', 'dla60x_elastic', 'dla102x', 'dla102x_elastic',
               'se_resnext50', 'se_resnext50_elastic', 'densenet201', 'densenet201_elastic']

"""
1.torch一个基础的抽象类torch.utils.data.Dataset类
2.DatasetFolder类(2继承1)
3.ImageFolder类(3继承2)

class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError
 
    def __len__(self):
        raise NotImplementedError
 
    def __add__(self, other):
        return ConcatDataset([self, other])


自定义Datasets的关键就是重载 "__len__"和"__getitem__"两个函数！
而 "__add__"函数的作用是使得类定义对象拥有"object1 + object2"的功能，一般情况不需要重载该函数。
1: __len__函数：使得类对象拥有 "len(object)"功能，返回dataset的size。
2: __getitem__函数：使得类对象拥有"object[index]"功能，可以用索引i去获得第i+1个样本


def __len__(self):
    # 因为图片的id是unique的，所以self.ids的长度就等于总图片数
    return len(self.ids)


"""

# 数据集大小：3000
# 训练集大小：2397
# 测试集大小：603
data_class=[8, 7]

class MyDataset(Dataset):
    # 路径名，数据变换
    def __init__(self, txt_path, txt_sem_path, transform=None, target_transform=None, sd=None): 
        imgs = [] # 图像路径数组
        ont = []  # 本体标签数组
        sem = []  # 隐义标签数组
        image_label=[] # 三合一的元组
        # 处理本体信息
        fh = open(txt_path, 'r')
        num_lines=fh.readlines() # 读取所有行
        num_lines=len(num_lines) # 得到文件总行数
        fh.close() # 关闭文件
        fh = open(txt_path, 'r')
        print(sd+": ", num_lines)
        for line in fh:
            # rstrip() 删除 string 字符串末尾的指定字符（默认为空格）
            line = line.rstrip()
            # split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串
            # str -- 分隔符，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等
            words = line.split()
            label_list=words[1:] # ['0', '1', '3']
            label_list = list(map(int,label_list)) # 将字符串数组转为整型数组
            classnum_ont=data_class[0] # 本体类别
            cap_ont=np.array([0]*classnum_ont,dtype=np.int) # 创建0数组
            cap_ont[label_list] = True # 对应有标签的值，赋值为1
#            imgs.append((words[0], cap_ont)) # 添加的是元组
            ont.append(cap_ont)
            imgs.append(words[0]) # 添加图片路径  
        fh.close() # 关闭文件
#        print("---------------------")
        # 处理隐义信息
        fs = open(txt_sem_path, 'r')
        for line in fs:
            # rstrip() 删除 string 字符串末尾的指定字符（默认为空格）
            line = line.rstrip()
            # split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串
            # str -- 分隔符，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等
            words = line.split()
            label_list=words[1:] # ['0', '1', '3']
            label_list = list(map(int,label_list)) # 将字符串数组转为整型数组
            classnum_sem=data_class[1] # 隐义类别
            cap_sem=np.array([0]*classnum_sem,dtype=np.int) # 创建0数组
            cap_sem[label_list] = True # 对应有标签的值，赋值为1
#            imgs.append((words[0], cap_sem)) # 添加的是元组
            sem.append(cap_sem)
        fs.close() # 关闭文件

        for i in range(num_lines):
#            print(i)
            image_label.append((imgs[i],ont[i],sem[i])) # 三合一的元组

        imgs=image_label # 转换
        self.imgs = imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
#        fn, label = self.imgs[index]  
        fn, ont, sem = self.imgs[index]  
        img = Image.open(fn)
        img=img.convert('RGB')   # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

#        return img, label
        return img, ont, sem

    def __len__(self):
        return len(self.imgs)




parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', metavar='ARCH', default='resnext50_elastic', choices=model_names,   # 模型的名字  
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext50_elastic)')  
parser.add_argument('--workers', '--j', default=4, type=int, metavar='N',  # 16
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',   # 36
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '-b', default=16, type=int,   # 96 # batch_size大小-------------
                    metavar='N', help='mini-batch size (default: 96)')
parser.add_argument('--num_gpus', '--g', default=1, type=int,      # GPU个数
                    metavar='N', help='number of GPUs to match (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, # 学习率
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print_freq', '-p', default=150, type=int,  # 117  # 刷新频率
                    metavar='N', help='print frequency (default: 117)')
parser.add_argument('--resume', default='2-ImageNet_pre-trained-model/', type=str, metavar='PATH', # 模型保存的ckpt文件
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', # 在验证集上评估模型
                    help='evaluate model on validation set')
# 分布式处理部分
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')  # 分布式进程数
parser.add_argument('--dist-url', default='', type=str,  # tcp://224.66.41.62:23456
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='', type=str,  # gloo
                    help='distributed backend')  # 分布式后端


def main():
    global args
    args = parser.parse_args()
    print('config: wd', args.weight_decay, 'lr', args.lr, 'batch_size', args.batch_size, 'num_gpus', args.num_gpus)
    # torch.cuda.device_count(): 返回可得到的GPU数量
    iteration_size = args.num_gpus // torch.cuda.device_count()  # do multiple iterations
    assert iteration_size >= 1
    args.weight_decay = args.weight_decay * iteration_size  # will cancel out with lr
    args.lr = args.lr / iteration_size
    args.batch_size = args.batch_size // iteration_size
    # 对于只有一块GPU来说，参数没有改变
    print('real: wd', args.weight_decay, 'lr', args.lr, 'batch_size', args.batch_size, 'iteration_size', iteration_size)
    # 分布式处理部分
    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size)

    # create model
    print("=> creating model '{}'".format(args.arch))
#    from resnext import resnext50_elastic 
#    from resnext_MulTask_11 import resnext50_elastic
#    from resnext_MulTask_12 import resnext50_elastic
    # 没有elastic结构
#    from resnext_MulTask_clothes_conv1_split import resnext50
#    model = resnext50(num_classes=data_class)  # 注意类别: 两个任务，对应两个不同的类别数

    # 有elastic结构
    from resnext_MulTask_clothes_conv1_split import resnext50_elastic
    model = resnext50_elastic(num_classes=data_class[1 ])  # 注意类别: 两个任务，对应两个不同的类别数

#    from resnext_MulTask_11 import resnext50
#    from resnext import resnext50
#    model = resnext50(num_classes=80)
#    model = models.__dict__[args.arch](num_classes=80) # 加载模型

    # count number of parameters
    count = 0
    params = list()
    for n, p in model.named_parameters():
        if '.ups.' not in n:
            params.append(p)
            count += np.prod(p.size())
    print('Parameters:', count/1000000.0, "( 百万)") # 参数的数量

    # count flops
    model = add_flops_counting_methods(model)
    model.eval()
    image = torch.randn(1, 3, 224, 224) # 图像归一化大小

    model.start_flops_count()
    model(image)[0].sum() # 有改动
    model.stop_flops_count()
    print("GFLOPs", model.compute_average_flops_cost() / 1000000000.0, '( 十亿)') # FLOP的个数

    # normal code
    model = torch.nn.DataParallel(model).cuda()
    # BCE损失函数
    criterion = nn.BCEWithLogitsLoss().cuda() 
    # SGD优化策略
    optimizer = torch.optim.SGD([{'params': iter(params), 'lr': args.lr},
                                 ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume: 
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume) # 加载ckpt文件

#            resume = ('module.fc.bias' in checkpoint['state_dict'] and
#                      checkpoint['state_dict']['module.fc.bias'].size() == model.module.fc.bias.size()) or \
#                     ('module.classifier.bias' in checkpoint['state_dict'] and
#                      checkpoint['state_dict']['module.classifier.bias'].size() == model.module.classifier.bias.size())
            resume=False
            if resume:
                # True resume: resume training on MS-COCO  # 在MS-COCO上 评估？
                print()
                print("resume training on MS-COCO...")
                print("在MS-COCO上 评估...")
                print()
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else print('no optimizer found')
                args.start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else args.start_epoch
            else:
                # Fake resume: transfer from ImageNet # 从ImageNet——>MS-COCO 训练？
                print()
                print("transfer from ImageNet...")
#                print("从ImageNet——>MS-COCO 训练...")
                print("从ImageNet——>服饰数据集 训练...")
                print()

                pretrained_dict = checkpoint['state_dict']
                model_dict = model.state_dict() # 字典对象
                pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict) #  对模型的参数进行更新
                model.load_state_dict(model_dict)

#                for n, p in list(checkpoint['state_dict'].items()):
#                    if 'classifier' in n or 'fc' in n:
#                        print(n, 'deleted from state_dict')
#                        del checkpoint['state_dict'][n]
#                model.load_state_dict(checkpoint['state_dict'], strict=False)

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch'] if 'epoch' in checkpoint else 'unknown'))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
 
    # 提升一点训练速度，没什么额外开销，一般都会加
    # 仅限于非多尺度训练！否则效果更差！
    cudnn.benchmark = True


#################################################################################################
    # 本体标签
    train_txt_path = os.path.join("train.txt")
    val_txt_path = os.path.join("test.txt")

    # 隐义标签
    train_sem_txt_path = os.path.join("train-sem.txt")
    val_sem_txt_path = os.path.join("test-sem.txt")

    # 服饰数据集正则化部分
    normTransform = transforms.Normalize(mean=[0.56391764, 0.43714827, 0.4107524], std=[0.22986116, 0.21178758, 0.20076773])

    # 训练集的数据变换
    trainTransform = transforms.Compose([
         transforms.RandomResizedCrop(224), # 随机裁剪,
         transforms.RandomHorizontalFlip(), # 随机水平翻转
         transforms.ToTensor(),
         normTransform # 正则化
    ])
    # 测试集的数据变换
    valTransform = transforms.Compose([
        transforms.Resize((224, 224)), # 调整图像大小
        transforms.ToTensor(),
        normTransform # 正则化
    ])

    # 构建MyDataset实例
    train_data = MyDataset(txt_path=train_txt_path, txt_sem_path=train_sem_txt_path, transform=trainTransform, sd='训练') # 路径名，数据变换
    val_data = MyDataset(txt_path=val_txt_path, txt_sem_path=val_sem_txt_path, transform=valTransform, sd='测试')
    print("---------------------")
    train_sampler = torch.utils.data.sampler.RandomSampler(train_data) # 随机采样器
    # 构建DataLoder
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,  shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,shuffle=False,
            num_workers=args.workers, pin_memory=True)


#################################################################################################
### 继续调用程序
    if args.evaluate:
        validate_multi(val_loader, model, criterion) # 在验证集上测试数据，return返回
        return

    for epoch in range(args.start_epoch, args.epochs):
        # 学习率调整
        coco_adjust_learning_rate(optimizer, epoch) 
        # 模型训练
        train_multi(train_loader, model, criterion, optimizer, epoch, iteration_size)
        print("***********************************************")
        print("模型训练完第 "+str(epoch+1)+" 轮，下面进行验证集实验...")
        print("***********************************************")
        # evaluate on validation set
        # 模型验证
        validate_multi(val_loader, model, criterion)
        # 模型保存的位置
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, False, filename='5-clothes-result/' + args.arch + '_checkpoint.pth.tar') 
        # filename='3-MSCOCO--model-train-demo/'+ 'coco_' + args.arch + '_checkpoint.pth.tar'


def train_multi(train_loader, model, criterion, optimizer, epoch, iteration_size):
    batch_time =  AverageMeter()
    data_time = AverageMeter()
    # 本体
#    losses1 = AverageMeter() # 训练损失
#    prec1 = AverageMeter() # 准确率
#    rec1 = AverageMeter() # 召回率
    # 隐义
    losses2 = AverageMeter() # 训练损失
    prec2 = AverageMeter() # 准确率
    rec2 = AverageMeter() # 召回率

    """
    优化器基本使用方法
       1.建立优化器实例
       2.循环：
          1.清空梯度
          2.向前传播
          3.计算Loss
          4.反向传播
          5.更新参数
	
      from torch import optim
      建立优化器实例
      optimizer = optim.SGD(params=net.parameters(), lr=1)
      1.清空梯度
      optimizer.zero_grad()  # net.zero_grad()
      2.向前传播
      input_ = t.autograd.Variable(t.randn(1, 3, 32, 32))
      output = net(input_)
      3.计算Loss
      loss = criterion(output, target.float())
      4.反向传播
      loss.backward()
      5.更新参数
      optimizer.step()

    """

    # switch to train mode
    # 模型训练准备
    model.train()  # 启用 BatchNormalization 和 Dropout
    optimizer.zero_grad() # 梯度清零
    end = time.time()
    # 本体
#    tp1, fp1, fn1, tn1, count1 = 0, 0, 0, 0, 0
    # 隐义
    tp2, fp2, fn2, tn2, count2 = 0, 0, 0, 0, 0
    for i, (input, target1, target2) in enumerate(train_loader):
#        print("input:", np.shape(input))  # input: torch.Size([16, 3, 224, 224])
#        print("target:", np.shape(target)) # target: torch.Size([16, 3, 80])
        # measure data loading time
#        print("target1: ", np.shape(target1))
#        print("target1: ", target1)
        data_time.update(time.time() - end) # 数据加载时间
        # 非阻塞(Non-Blocking):非阻塞允许多个线程同时进入临界区
#        target1 = target1.cuda(non_blocking=True)
        target2 = target2.cuda(non_blocking=True)
#        print("target:", np.shape(target)) # target: torch.Size([16, 3, 80])
        # 假设A是一个 m 行 n 列的矩阵；
        #     A.max(1)：返回A每一行最大值组成的一维数组；
        # torch.max()[0]， 只返回最大值的每个数
        # troch.max()[1]， 只返回最大值的每个索引
#        target1 = target1.max(dim=1)[0]   # target: torch.Size([16, 80])
#        target2 = target2.max(dim=1)[0]   # target: torch.Size([16, 80])
        # compute output 
        output2 = model(input) # 计算模型输出
#        print("output1: ", output1)
#        print("target1: ", target1)
#        print("output:", np.shape(output)) # torch.Size([16, 80])
#        print("-----------------------")
#        print()
        # BCE损失函数
        # 本体
#        print("output1: ", np.shape(output1))
#        print("target1: ", np.shape(target1))
#        print("target1:", target1)
#        loss1 = criterion(output1, target1.float()) * 80.0 #  损失 扩大80倍
        # 隐义
        loss2 = criterion(output2, target2.float()) * 80.0 #  损失 扩大80倍

        # measure accuracy and record loss
        # x.data:  .data返回和 x 的相同数据 tensor, 但不会加入到x的计算历史里
        # greater than（大于）
#        pred1 = output1.data.gt(0.0).long()  #  相当于是数据类型的转换
        pred2 = output2.data.gt(0.0).long()  #  相当于是数据类型的转换
        # 本体
#        tp1, fp1, fn1, tn1, this_prec1, this_rec1 = evaluating_train(pred1, target1, input)
        # 隐义
#        tp2, fp2, fn2, tn2, this_prec2, this_rec2 = evaluating_train(pred2, target2, input)

        """ 原程序
        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)
        count += input.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()
        this_acc = (this_tp + this_tn).float() / (this_tp + this_tn + this_fp + this_fn).float()

        this_prec = this_tp.float() / (this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0
        """
        # 本体
#        tp1 += (pred1 + target1).eq(2).sum(dim=0)
#        fp1 += (pred1 - target1).eq(1).sum(dim=0)
#        fn1 += (pred1 - target1).eq(-1).sum(dim=0)
#        tn1 += (pred1 + target1).eq(0).sum(dim=0)
#        count1 += input.size(0)

#        this_tp1 = (pred1 + target1).eq(2).sum()
#        this_fp1 = (pred1 - target1).eq(1).sum()
#        this_fn1 = (pred1 - target1).eq(-1).sum()
#        this_tn1 = (pred1 + target1).eq(0).sum()
#        this_acc1 = (this_tp1 + this_tn1).float() / (this_tp1 + this_tn1 + this_fp1 + this_fn1).float()
#        mAP1 = this_acc1.data.cpu().numpy()

#        this_prec1 = this_tp1.float() / (this_tp1 + this_fp1).float() * 100.0 if this_tp1 + this_fp1 != 0 else 0.0
#        this_rec1 = this_tp1.float() / (this_tp1 + this_fn1).float() * 100.0 if this_tp1 + this_fn1 != 0 else 0.0

        # 隐义
        tp2 += (pred2 + target2).eq(2).sum(dim=0)
        fp2 += (pred2 - target2).eq(1).sum(dim=0)
        fn2 += (pred2 - target2).eq(-1).sum(dim=0)
        tn2 += (pred2 + target2).eq(0).sum(dim=0)
        count2 += input.size(0)

        this_tp2 = (pred2 + target2).eq(2).sum()
        this_fp2 = (pred2 - target2).eq(1).sum()
        this_fn2 = (pred2 - target2).eq(-1).sum()
        this_tn2 = (pred2 + target2).eq(0).sum()
        this_acc2 = (this_tp2 + this_tn2).float() / (this_tp2 + this_tn2 + this_fp2 + this_fn2).float()
        mAP2 = this_acc2.data.cpu().numpy()

        this_prec2 = this_tp2.float() / (this_tp2 + this_fp2).float() * 100.0 if this_tp2 + this_fp2 != 0 else 0.0
        this_rec2 = this_tp2.float() / (this_tp2 + this_fn2).float() * 100.0 if this_tp2 + this_fn2 != 0 else 0.0

        # 本体
#        losses1.update(float(loss1), input.size(0))
#        prec1.update(float(this_prec1), input.size(0))
#        rec1.update(float(this_rec1), input.size(0))
        # compute gradient and do SGD step
        # 反向传播
#        loss1.backward()

        # 隐义
        losses2.update(float(loss2), input.size(0))
        prec2.update(float(this_prec2), input.size(0))
        rec2.update(float(this_rec2), input.size(0))
        # compute gradient and do SGD step
        # 反向传播
#        loss2.backward()

        # 权值为1，进行反向传播
        loss=loss2
        loss.backward()

        if i % iteration_size == iteration_size - 1: # 在GPU=1的情况下，恒满足
            optimizer.step() # 更新参数
            optimizer.zero_grad() # 清空梯度
        # measure elapsed time
        batch_time.update(time.time() - end) # 耗时
        end = time.time()

        """ 原程序
        # C-P C-R C-F1 
        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for i in range(len(tp))]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)
        # O-P O-R O-F1
        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)
        """
        # 本体
#        mean_p_c1, mean_r_c1, mean_f_c1, p_o1, r_o1, f_o1 = evaluating_pre(tp1, fp1, fn1)
        # 隐义
        mean_p_c2, mean_r_c2, mean_f_c2, p_o2, r_o2, f_o2 = evaluating_pre(tp2, fp2, fn2)

        # 本体
#        if (i+1) % args.print_freq == 0: # 频率刷新
#            print("本体...")
#            print('Epoch: [{0}][{1}/{2}]\t'
#                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                  'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
#                  'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
#                   epoch+1, i+1, len(train_loader), batch_time=batch_time,
#                   data_time=data_time, loss=losses1, prec=prec1, rec=rec1))
#            print('P_C {:.2f} R_C {:.2f} F_C {:.2f} \t P_O {:.2f} R_O {:.2f} F_O {:.2f} \t mAP {:.2f}'
#                  .format(mean_p_c1, mean_r_c1, mean_f_c1, p_o1, r_o1, f_o1, mAP1))
#            print()

        # 隐义
        if (i+1) % args.print_freq == 0: # 频率刷新
            print("隐义...")
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                  'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
                   epoch+1, i+1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses2, prec=prec2, rec=rec2))
            print('P_C {:.2f} R_C {:.2f} F_C {:.2f} \t P_O {:.2f} R_O {:.2f} F_O {:.2f} \t mAP {:.2f}'
                  .format(mean_p_c2, mean_r_c2, mean_f_c2, p_o2, r_o2, f_o2, mAP2))
            print()
            print("------------------------------------------------------------------")
        


def validate_multi(val_loader, model, criterion):
    batch_time = AverageMeter()
    # 本体
#    losses1 = AverageMeter() # 训练损失
#    prec1 = AverageMeter() # 准确率
#    rec1 = AverageMeter() # 召回率
    # 隐义
    losses2 = AverageMeter() # 训练损失
    prec2 = AverageMeter() # 准确率
    rec2 = AverageMeter() # 召回率

    # switch to evaluate mode
    # eval()时，pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值
    model.eval() # 不启用 BatchNormalization 和 Dropout

    end = time.time()
    # 本体
#    tp1, fp1, fn1, tn1, count1 = 0, 0, 0, 0, 0
#    tp_size1, fn_size1 = 0, 0
   # 隐义
    tp2, fp2, fn2, tn2, count2 = 0, 0, 0, 0, 0
#    tp_size2, fn_size2 = 0, 0
    for i, (input, target1, target2) in enumerate(val_loader):  # target1: 本体； target2: 隐义
        # 本体
#        target1 = target1.cuda(non_blocking=True)
#        original_target1 = target1
#        target1 = target1.max(dim=1)[0]
        # 隐义
        target2 = target2.cuda(non_blocking=True)
        original_target2 = target2
#        target2 = target2.max(dim=1)[0]
        # compute output
        # requires_grad=True: 要求计算梯度 # requires_grad=False: 不要求计算梯度
        # with torch.no_grad(): 不需要计算梯度，也不会进行反向传播  （torch.no_grad()是新版本pytorch中volatile的替代）
        with torch.no_grad():
            output2 = model(input) # 前向传播
#            loss1 = criterion(output1, target1.float()) # 计算本体Loss
            loss2 = criterion(output2, target2.float()) # 计算隐义Loss

        # measure accuracy and record loss
#        pred1 = output1.data.gt(0.0).long() # 类型转换
        pred2 = output2.data.gt(0.0).long() # 类型转换
        # 调用评价指标的函数
        # 本体
#        tp1, fp1, fn1, tn1, this_prec1, this_rec1 = evaluating_val(pred1, target1, original_target1, input)
#        count += input.size(0)
        # 隐义
#        tp2, fp2, fn2, tn2, this_prec2, this_rec2 = evaluating_val(pred2, target2, original_target2, input)


        """ 原程序
        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)
        three_pred = pred.unsqueeze(1).expand(-1, 3, -1)  # n, 3, 80
        tp_size += (three_pred + original_target).eq(2).sum(dim=0)
        fn_size += (three_pred - original_target).eq(-1).sum(dim=0)
        count += input.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()
        this_acc = (this_tp + this_tn).float() / (this_tp + this_tn + this_fp + this_fn).float()

        this_prec = this_tp.float() / (this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0
        """
        
        # 本体
#        tp1 += (pred1 + target1).eq(2).sum(dim=0)
#        fp1 += (pred1 - target1).eq(1).sum(dim=0)
#        fn1 += (pred1 - target1).eq(-1).sum(dim=0)
#        tn1 += (pred1 + target1).eq(0).sum(dim=0)
#        three_pred1 = pred1.unsqueeze(1).expand(-1, 3, -1)  # n, 3, 80
#        tp_size1 += (three_pred1 + original_target1).eq(2).sum(dim=0)
#        fn_size1 += (three_pred1 - original_target1).eq(-1).sum(dim=0)
#        count1 += input.size(0)

#        this_tp1 = (pred1 + target1).eq(2).sum()
#        this_fp1 = (pred1 - target1).eq(1).sum()
#        this_fn1 = (pred1 - target1).eq(-1).sum()
#        this_tn1 = (pred1 + target1).eq(0).sum()
#        this_acc1 = (this_tp1 + this_tn1).float() / (this_tp1 + this_tn1 + this_fp1 + this_fn1).float()
#        mAP1 = this_acc1.data.cpu().numpy()

#        this_prec1 = this_tp1.float() / (this_tp1 + this_fp1).float() * 100.0 if this_tp1 + this_fp1 != 0 else 0.0
#        this_rec1 = this_tp1.float() / (this_tp1 + this_fn1).float() * 100.0 if this_tp1 + this_fn1 != 0 else 0.0


        # 隐义
        tp2 += (pred2 + target2).eq(2).sum(dim=0)
        fp2 += (pred2 - target2).eq(1).sum(dim=0)
        fn2 += (pred2 - target2).eq(-1).sum(dim=0)
        tn2 += (pred2 + target2).eq(0).sum(dim=0)
#        three_pred2 = pred2.unsqueeze(1).expand(-1, 3, -1)  # n, 3, 80
#        tp_size2 += (three_pred2 + original_target2).eq(2).sum(dim=0)
#        fn_size2 += (three_pred2 - original_target2).eq(-1).sum(dim=0)
        count2 += input.size(0)

        this_tp2 = (pred2 + target2).eq(2).sum()
        this_fp2 = (pred2 - target2).eq(1).sum()
        this_fn2 = (pred2 - target2).eq(-1).sum()
        this_tn2 = (pred2 + target2).eq(0).sum()
        this_acc2 = (this_tp2 + this_tn2).float() / (this_tp2 + this_tn2 + this_fp2 + this_fn2).float()
        mAP2 = this_acc2.data.cpu().numpy()

        this_prec2 = this_tp2.float() / (this_tp2 + this_fp2).float() * 100.0 if this_tp2 + this_fp2 != 0 else 0.0
        this_rec2 = this_tp2.float() / (this_tp2 + this_fn2).float() * 100.0 if this_tp2 + this_fn2 != 0 else 0.0


        # 本体
#        losses1.update(float(loss1), input.size(0))
#        prec1.update(float(this_prec1), input.size(0))
#        rec1.update(float(this_rec1), input.size(0))

        # 隐义
        losses2.update(float(loss2), input.size(0))
        prec2.update(float(this_prec2), input.size(0))
        rec2.update(float(this_rec2), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end) # 耗时
        end = time.time()

        # 调用预测指标的函数
        # 本体        
#        mean_p_c1, mean_r_c1, mean_f_c1, p_o1, r_o1, f_o1 = evaluating_pre(tp1, fp1, fn1)
        # 隐义
        mean_p_c2, mean_r_c2, mean_f_c2, p_o2, r_o2, f_o2 = evaluating_pre(tp2, fp2, fn2)

        """ 原程序
        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for i in range(len(tp))]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)
        """
        # 本体
#        if (i+1) % args.print_freq == 0:
#            print("本体...")
#            print('Test: [{0}/{1}]\t'
#                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                  'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
#                  'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
#                   i+1, len(val_loader), batch_time=batch_time, loss=losses1, prec=prec1, rec=rec1))
#            print('P_C {:.2f} R_C {:.2f} F_C {:.2f} \t P_O {:.2f} R_O {:.2f} F_O {:.2f} \t mAP {:.2f}'
#                  .format(mean_p_c1, mean_r_c1, mean_f_c1, p_o1, r_o1, f_o1, mAP1))
#            print()
        
        # 隐义 
        if (i+1) % args.print_freq == 0:
            print("隐义...")
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                  'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
                   i+1, len(val_loader), batch_time=batch_time, loss=losses2, prec=prec2, rec=rec2))
            print('P_C {:.2f} R_C {:.2f} F_C {:.2f} \t P_O {:.2f} R_O {:.2f} F_O {:.2f} \t mAP {:.2f}'
                  .format(mean_p_c2, mean_r_c2, mean_f_c2, p_o2, r_o2, f_o2, mAP2))
            print()
            print("------------------------------------------------------------------")


    print()
    print("***验证集的最终结果")
#    print("本体验证集最终结果...")
#    print(' * P_C {:.2f} R_C {:.2f} F_C {:.2f} \t P_O {:.2f} R_O {:.2f} F_O {:.2f} \t mAP {:.2f}'
#          .format(mean_p_c1, mean_r_c1, mean_f_c1, p_o1, r_o1, f_o1, mAP1))
#    print()
    print("隐义验证集最终结果...")
    print(' * P_C {:.2f} R_C {:.2f} F_C {:.2f} \t P_O {:.2f} R_O {:.2f} F_O {:.2f} \t mAP {:.2f}'
          .format(mean_p_c2, mean_r_c2, mean_f_c2, p_o2, r_o2, f_o2, mAP2))
    print()
    return


#  学习率调整
def coco_adjust_learning_rate(optimizer, epoch):
    # isinstance(object, classinfo)
    # # 如果参数1的object类型 与参数2的classinfo类型 相同则返回 True，否则返回 False
    if isinstance(optimizer, torch.optim.Adam):  # Adam优化不用调整学习率
        return
    lr = args.lr # 学习率
    if epoch >= 15:
        lr *= 0.1
    if epoch >= 25:
        lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 计算CP、CR、 CF1、 OP、 OR、 OF1
def evaluating_pre(tp, fp, fn):
    p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
    r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
    f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for i in range(len(tp))]

    mean_p_c = sum(p_c) / len(p_c)
    mean_r_c = sum(r_c) / len(r_c)
    mean_f_c = sum(f_c) / len(f_c)

    p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
    r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
    f_o = 2 * p_o * r_o / (p_o + r_o)
 
    return mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o



if __name__ == '__main__':
    main()







