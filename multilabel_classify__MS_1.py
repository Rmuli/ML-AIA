# -*- coding: utf-8 -*-
import argparse
import time
import shutil
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

from utils2 import AverageMeter, add_flops_counting_methods
# 评价指标-top3
from top3_metrics_Back import overall_topk, calculate_mAP


# from utils import add_flops_counting_methods, AverageMeter
# 评价指标-top3
# from metrics_top3 import calculate_metrics 

# 可选模型
model_names = ['resnext50', 'resnext50_elastic', 'resnext101', 'resnext101_elastic']


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='mscoco--images/', help='path to dataset') # metavar:设置help输出信息
parser.add_argument('--arch', metavar='ARCH', default='resnext50_elastic', choices=model_names,   # 模型的名字  
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext50_elastic)')  
parser.add_argument('--workers', '--j', default=2, type=int, metavar='N',  # 16
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=36, type=int, metavar='N',   # 36
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '-b', default=16, type=int,   # 96 # batch_size大小-------------
                    metavar='N', help='mini-batch size (default: 96)')
parser.add_argument('--num_gpus', '--g', default=1, type=int,      # GPU个数
                    metavar='N', help='number of GPUs to match (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, # 学习率
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[24, 30],  # 在线降低 学习率  --已无用
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print_freq', '-p', default=200, type=int,  # 117  # 刷新频率
                    metavar='N', help='print frequency (default: 117)')
# Checkpoints
parser.add_argument('--save_to_checkpoint', default='03-MSCOCO-train_result/resnext50', type=str, metavar='PATH', # 模型训练结果保存的地方
                    help='path to save checkpoint (default: checkpoint)')
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


# 参数解析
global args
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


# MS-COCO数据集处理函数
class CocoDetection(datasets.coco.CocoDetection):
    # 图像，注释，变换
    def __init__(self, root, annFile, transform=None, target_transform=None): 
        # root: COCO形式的数据集的根目录地址
        # annFile: COCO形式的数据集中.json文件的目录地址
        # transform: 原始图像是否需要进行变换（数据增强，默认是None不做增强）
        # target_transform: 标签是否需要进行变换（标签变换需要和原始图像变换相对应，默认是None不做增强

        # 从cocoapi导入pycocotools下的COCO类
        from pycocotools.coco import COCO
        self.root = root
        # 初始化一个COCO对象
        self.coco = COCO(annFile) # 初始化COCO对象时，将.json文件解析为字典形式导入内存，并创建调用createIndex()创建索引
        # 将每张图unique的id属性转化为list存储在self.ids中
        self.ids = list(self.coco.imgs.keys()) # self.coco.imgs是以每张图unique的id作为key，json文件images下每一image信息作为value的一个字典
        # 如: coco.imgs显示如下：
        #     { 397133:{'filename':'0000393133.jpg','coco_url':'http...','height':427,...,'wieght':640,'id':397133},
        #       377777:{'filename':'0000377777.jpg','coco_url':'http...','height':230,...,'wieght':352,'id':377777},
        #     }
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)


    def __getitem__(self, index):
        coco = self.coco
        # 通过索引获得图片的id
        img_id = self.ids[index]
        # 再通过getAnnIds方法利用img_id找到对应的anno_id
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # 根据anno_id和标签之间的映射关系，解析出标签target
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        # 由图片id加载相应的图片
        path = coco.loadImgs(img_id)[0]['file_name']
        # 根据每张图的file_name结合之前传入的图片放置的根目录读取图片信息
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        # 判断是否需要进行数据增强
        if self.transform is not None:
            img = self.transform(img)
        # 判断标签是否需要进行变换
        if self.target_transform is not None:
            target = self.target_transform(target)
        # 最终返回值形式可以根据自己需要进行设计。此处为一个tuple,包含一张图片以及对应的标签。
        return img, target


# 最好的测试结果
best_cf, best_of = 0, 0  # best test result
best_epoch=1
def main():
    global best_cf # 全局变量
    global best_of # 全局变量
    global best_epoch # 全局变量

    # 创建文件
    if not os.path.isdir(args.save_to_checkpoint):
        mkdir_p(args.save_to_checkpoint)

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

#    from resnext import resnext50
#    model = resnext50(num_classes=80)

    from resnext2 import resnext50_elastic 
    model = resnext50_elastic(num_classes=80)
 
#    from resnext_MulTask_12 import resnext50_elastic
#    model = resnext50_elastic(num_classes=80)  # 模型 经过预训练   # opts["num_labels"] = 14

    # count number of parameters
    count = 0
    params = list()
    for n, p in model.named_parameters():
        if '.ups.' not in n:
            params.append(p)
            count += np.prod(p.size())
    print('Parameters:', count/1000000.0) # 参数的数量

    # count flops
    model = add_flops_counting_methods(model)
    model.eval()
    image = torch.randn(1, 3, 224, 224)

    model.start_flops_count()
    model(image).sum()
    model.stop_flops_count()
    print("GFLOPs", model.compute_average_flops_cost() / 1000000000.0) # FLOP的个数

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

            resume = ('module.fc.bias' in checkpoint['state_dict'] and
                      checkpoint['state_dict']['module.fc.bias'].size() == model.module.fc.bias.size()) or \
                     ('module.classifier.bias' in checkpoint['state_dict'] and
                      checkpoint['state_dict']['module.classifier.bias'].size() == model.module.classifier.bias.size())
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
                print("从ImageNet——>MS-COCO 训练...")
                print()
                for n, p in list(checkpoint['state_dict'].items()):
                    if 'classifier' in n or 'fc' in n:
                        print(n, 'deleted from state_dict')
                        del checkpoint['state_dict'][n]
                model.load_state_dict(checkpoint['state_dict'], strict=False)

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch'] if 'epoch' in checkpoint else 'unknown'))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    # 提升一点训练速度，没什么额外开销，一般都会加
    # 仅限于非多尺度训练！否则效果更差！
    cudnn.benchmark = True

    # Data loading code
    # ms-coco正则化部分
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 训练集处理
    train_dataset = CocoDetection(os.path.join(args.data, 'train2014'),
                                  os.path.join(args.data, 'annotations/instances_train2014.json'),
                                  transforms.Compose([
                                      transforms.RandomResizedCrop(224), # 随机裁剪
                                      transforms.RandomHorizontalFlip(), # 随机水平翻转
                                      transforms.ToTensor(),
                                      normalize, # 正则化
                                  ]))
#    print("train_dataset: ", train_dataset)
    # 验证集处理
    val_dataset = CocoDetection(os.path.join(args.data, 'val2014'),
                                os.path.join(args.data, 'annotations/instances_val2014.json'),
                                transforms.Compose([
                                    transforms.Resize((224, 224)), # 调整图像大小
                                    transforms.ToTensor(),
                                    normalize, # 正则化
                                ])) 

    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset) # 随机采样器

    # torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=default_collate,
    #                             pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
    # 1.dataset(Dataset): (image, label)形式，数据读取接口（比如torchvision.datasets.ImageFolder）或者自定义的数据接口的输出，
    #                     该输出是torch.utils.data.Dataset类的对象(或者继承自该类的自定义类的对象)。
    # 2.batch_size: 批训练数据量的大小，根据具体情况设置即可。（默认：1）
    # 3.shuffle: 打乱数据，一般在训练数据中会采用。（默认：False）
    # 4.sampler(Sampler, optional): 从数据集中提取样本的策略。如果指定，“shuffle”必须为false。一般默认即可。
    # 5.num_workers，这个参数必须大于等于0，其他大于0的数表示通过多个进程来导入数据，可以加快数据导入速度。（默认：0）
    # 6.pin_memory (bool, optional)：数据加载器将把张量复制到CUDA内存中，然后返回它们。也就是一个数据拷贝的问题。
    # 7.drop_last (bool, optional): 如果数据集大小不能被批大小整除，则设置为“true”以除去最后一个未完成的批。如果“false”那么最后一批将更小。（默认：false）
    # 8.timeout(numeric, optional)：设置数据读取超时，但超过这个时间还没读取到数据的话就会报错。（默认：0）
    # 训练集
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    # 验证集
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate_multi(val_loader, model, criterion) # 在验证集上测试数据，return返回
        return

    for epoch in range(args.start_epoch, args.epochs):
        # 学习率调整
#        coco_adjust_learning_rate(optimizer, epoch) 
        adjust_learning_rate(optimizer, epoch) 
        print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("---------------------------------------------------")
        print('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        # 模型训练
#        train_multi(train_loader, model, criterion, optimizer, epoch, iteration_size)
        print("***********************************************")
        print("模型训练完第 "+str(epoch+1)+" 轮，下面进行验证集实验...")
        print("***********************************************")
        # evaluate on validation set
        # 模型验证
        test_CF, test_OF = validate_multi(val_loader, model, criterion)

        # save model  best_cf, best_of
        is_best = test_CF > best_cf and test_OF > best_of # 两个同时大于，才为True
        if is_best:
           best_cf, best_of= max(test_CF, best_cf), max(test_OF, best_of) # 保存最好的模型
           best_epoch=epoch + 1
           print("需要对CF/OF进行更新...")
        print()

        # 模型保存的位置
#        save_checkpoint({
#            'epoch': epoch + 1,
#            'arch': args.arch,
#            'state_dict': model.state_dict(),
#            'optimizer': optimizer.state_dict(),
#        }, False, filename='3-MSCOCO--model-train-demo/'+ 'coco_' + args.arch + '_checkpoint_'+str(epoch+1)+'.pth.tar') 

        print("正在保存模型...")
        # 模型保存的位置
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, save_to_checkpoint=args.save_to_checkpoint, filename=str(epoch + 1)) 

    print("***best_epoch: ", best_epoch)
    print('***best_cf: {:.2f} \t best_of: {:.2f}'.format(best_cf, best_of))
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))



def train_multi(train_loader, model, criterion, optimizer, epoch, iteration_size):
    batch_time =  AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter() # 训练损失
    prec = AverageMeter() # 准确率
    rec = AverageMeter() # 召回率

    # switch to train mode
    # 模型训练准备
    model.train()  # 启用 BatchNormalization 和 Dropout
    optimizer.zero_grad() # 梯度清零
    end = time.time()
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    for i, (inputs, target) in enumerate(train_loader):
        data_time.update(time.time() - end) # 数据加载时间
        target = target.cuda(non_blocking=True)
        target = target.max(dim=1)[0]   # target: torch.Size([16, 80])
        output = model(inputs) # 计算模型输出
        # BCE损失函数
        loss = criterion(output, target.float()) * 80.0 #  损失 扩大80倍

        # 对top-3指标进行处理
        output_top3 = output
        target_top3 = target

        if i == 0:
           labels_test = target_top3   # 真实标签
           outputs_test = output_top3  # 模型预测值
        else:
           labels_test = torch.cat((labels_test, target_top3), 0)
           outputs_test = torch.cat((outputs_test, output_top3), 0)

        pred = output.data.gt(0.0).long()  #  list里面，大于0的值为1,小于0的为0

        y_true = target.cpu().detach().numpy() # 数据类型转换
        y_pred = output.cpu().detach().numpy() # 数据类型转换

        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)
        count += inputs.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()
        this_acc = (this_tp + this_tn).float() / (this_tp + this_tn + this_fp + this_fn).float()

        this_prec = this_tp.float() / (this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        losses.update(float(loss), inputs.size(0))
        prec.update(float(this_prec), inputs.size(0))
        rec.update(float(this_rec), inputs.size(0))
        # compute gradient and do SGD step
        # 反向传播
        loss.backward()

        if i % iteration_size == iteration_size - 1: # 在GPU=1的情况下，恒满足
            optimizer.step() # 更新参数
            optimizer.zero_grad() # 清空梯度
        # measure elapsed time
        batch_time.update(time.time() - end) # 耗时
        end = time.time()

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

        if (i+1) % args.print_freq == 0: # 频率刷新
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                  'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
                   epoch+1, i+1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, prec=prec, rec=rec))
            print('*  all: P_C {:.2f} R_C {:.2f} F_C {:.2f} \t P_O {:.2f} R_O {:.2f} F_O {:.2f}'
                  .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))
            print()

    print("-------------------------------top-3----------------------------------")
    # top-3指标 计算
    print("训练集正在计算top-3指标....")
    mAP = calculate_mAP(labels_test, outputs_test)
    pc_top3, rc_top3, f1c_top3, po_top3, ro_top3, f1o_top3 = overall_topk(labels_test, outputs_test, 3) 
    print(' * top3: P_C {:.2f} R_C {:.2f} F_C {:.2f} \t P_O {:.2f} R_O {:.2f} F_O {:.2f} \t mAP {:.2f}'
          .format(pc_top3, rc_top3, f1c_top3, po_top3, ro_top3, f1o_top3, mAP))
    print()



def validate_multi(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter() # 训练损失
    prec = AverageMeter() # 准确率
    rec = AverageMeter() # 召回率

    model.eval() # 不启用 BatchNormalization 和 Dropout

    end = time.time()
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    tp_size, fn_size = 0, 0
    for i, (inputs, target) in enumerate(val_loader): # 每次一个batch_size的大小
        target = target.cuda(non_blocking=True)
        original_target = target
        target = target.max(dim=1)[0]
        # compute output
        # requires_grad=True: 要求计算梯度 # requires_grad=False: 不要求计算梯度
        # with torch.no_grad(): 不需要计算梯度，也不会进行反向传播  （torch.no_grad()是新版本pytorch中volatile的替代）
        with torch.no_grad():
            output = model(inputs) # 前向传播
            loss = criterion(output, target.float()) # 计算Loss

#        output=torch.sigmoid(output)  #  sigmoid激活

        # 对top-3指标进行处理
        output_top3 = output  
        target_top3 = target

        if i == 0:
           labels_test = target_top3   # 真实标签
           outputs_test = output_top3  # 模型预测值
        else:
           labels_test = torch.cat((labels_test, target_top3), 0)
           outputs_test = torch.cat((outputs_test, output_top3), 0)

        pred = output.data.gt(0.0).long() # 类型转换

        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)
        three_pred = pred.unsqueeze(1).expand(-1, 3, -1)  # n, 3, 80
        tp_size += (three_pred + original_target).eq(2).sum(dim=0)
        fn_size += (three_pred - original_target).eq(-1).sum(dim=0)
        count += inputs.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()
        this_acc = (this_tp + this_tn).float() / (this_tp + this_tn + this_fp + this_fn).float()

        this_prec = this_tp.float() / (this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        losses.update(float(loss), inputs.size(0))
        prec.update(float(this_prec), inputs.size(0))
        rec.update(float(this_rec), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end) # 耗时
        end = time.time()

        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for i in range(len(tp))]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                  'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
                   i+1, len(val_loader), batch_time=batch_time, loss=losses,
                   prec=prec, rec=rec))
            print('P_C {:.2f} R_C {:.2f} F_C {:.2f} \t P_O {:.2f} R_O {:.2f} F_O {:.2f}'
                  .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))
            print()
    
    print('--------------------------------------------------------------------')
    print("验证集的最终结果为：")
    print(' * P_C {:.2f} R_C {:.2f} F_C {:.2f} \t P_O {:.2f} R_O {:.2f} F_O {:.2f}'
          .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))
    print("-------------------------------top-3----------------------------------")
    # top-3指标 计算
    print("验证集正在计算top-3指标....")
    mAP = calculate_mAP(labels_test, outputs_test)
    pc_top3, rc_top3, f1c_top3, po_top3, ro_top3, f1o_top3 = overall_topk(labels_test, outputs_test, 3) 
    print(' * P_C {:.2f} R_C {:.2f} F_C {:.2f} \t P_O {:.2f} R_O {:.2f} F_O {:.2f} \t mAP {:.2f}'
          .format(pc_top3, rc_top3, f1c_top3, po_top3, ro_top3, f1o_top3, mAP))
    print()
    test_CF, test_OF = mean_f_c, f_o # 返回值
    return test_CF, test_OF


# 模型保存的位置
def save_checkpoint(state, is_best, save_to_checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_to_checkpoint, 'checkpoint_'+filename+'.pth.tar')
    torch.save(state, filepath)
    if is_best:
        # shutil.copyfile(src, dst):将名为src的文件的内容（无元数据）复制到名为dst的文件中，dst必须是完整的目标文件名
        shutil.copyfile(filepath, os.path.join(save_to_checkpoint, 'model_best.pth.tar'))

# 由路径创建文件夹
def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# 学习率调整--原论文
def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

#  学习率调整
def coco_adjust_learning_rate(optimizer, epoch):
    # isinstance(object, classinfo)
    # # 如果参数1的object类型 与参数2的classinfo类型 相同则返回 True，否则返回 False
    if isinstance(optimizer, torch.optim.Adam):  # Adam优化不用调整学习率
        return
    lr = args.lr # 学习率
    # if epoch >= 12:
    #     lr *= 0.1
    if epoch >= 24: # 24
        lr *= 0.1
    if epoch >= 30: # 30
        lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("当前是第 ", epoch)
    print("当前学习率为：", lr)


if __name__ == '__main__':
    main()







