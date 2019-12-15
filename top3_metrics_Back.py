# -*- coding: utf-8 -*-
import numpy as np
import torch


def overall_topk(targets, scores, k):  # labels_test, outputs_test
#    scores = torch.sigmoid(scores)  #  sigmoid激活
    targets = targets.cpu().detach().numpy()
    targets[targets == -1] = 0
    n, c = scores.size()
    scores2 = np.zeros((n, c))-1
    # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
    # 返回一个元组 (values,indices)，其中indices是原始输入张量input中测元素下标。
    # 如果为largest为 False ，则返回最小的 k 个值
    # 如果设定布尔值sorted 为_True_，将会确保返回的 k 个值被排序。
    index = scores.topk(k, 1, True, True)[1].cpu().detach().numpy()
    tmp = scores.cpu().detach().numpy()
    for i in range(n):
        for ind in index[i]:
            scores2[i, ind] = 1 if tmp[i, ind] >= 0 else -1
    OP, OR, OF1, CP, CR, CF1 = evaluation(scores2, targets)
    return OP, OR, OF1, CP, CR, CF1


def evaluation(scores_, targets_):
    n, n_class = scores_.shape
    Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
    for k in range(n_class):
        scores = scores_[:, k]
        targets = targets_[:, k]
        targets[targets == -1] = 0
        Ng[k] = np.sum(targets == 1)
        Np[k] = np.sum(scores >= 0)
        Nc[k] = np.sum(targets * (scores >= 0))
    Np[Np == 0] = 1
    OP = np.sum(Nc) / np.sum(Np)
    OR = np.sum(Nc) / np.sum(Ng)
    OF1 = (2 * OP * OR) / (OP + OR)

    CP = np.sum(Nc / Np) / n_class
    CR = np.sum(Nc / Ng) / n_class
    CF1 = (2 * CP * CR) / (CP + CR)
    OP, OR, OF1, CP, CR, CF1 = OP * 100.0, OR * 100.0, OF1 * 100.0, CP * 100.0, CR * 100.0, CF1 * 100.0
    return OP, OR, OF1, CP, CR, CF1


def calculate_mAP(labels, preds):
    preds = torch.sigmoid(preds)  #  sigmoid激活
    labels = labels.cpu().detach().numpy() # 数据类型转换
    preds = preds.cpu().detach().numpy() # 数据类型转换
    no_examples = labels.shape[0]
    no_classes = labels.shape[1]

    ap_scores = np.empty((no_classes), dtype=np.float)
    for ind_class in range(no_classes):
        ground_truth = labels[:, ind_class]
        out = preds[:, ind_class]

        sorted_inds = np.argsort(out)[::-1] # in descending order
        tp = ground_truth[sorted_inds]
        fp = 1 - ground_truth[sorted_inds]
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        rec = tp / np.sum(ground_truth)
        prec = tp / (fp + tp)

        rec = np.insert(rec, 0, 0)
        rec = np.append(rec, 1)
        prec = np.insert(prec, 0, 0)
        prec = np.append(prec, 0)

        for ind in range(no_examples, -1, -1):
            prec[ind] = max(prec[ind], prec[ind + 1])

        inds = np.where(rec[1:] != rec[:-1])[0] + 1
        ap_scores[ind_class] = np.sum((rec[inds] - rec[inds - 1]) * prec[inds])

    return 100 * np.mean(ap_scores)



