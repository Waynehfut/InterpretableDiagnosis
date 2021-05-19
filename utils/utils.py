# -*- coding: utf-8 -*-
import random

import torch
import numpy as np
from torch import nn
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage.morphology import distance_transform_edt


def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
                pred.double().sum() + truth.double().sum() + 1
        )

        return bce_loss + (1 - dice_coef)


# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class MetricTracker(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
def jaccard_index(input, target):
    intersection = (input * target).long().sum().data.cpu()[0]
    union = (
            input.long().sum().data.cpu()[0]
            + target.long().sum().data.cpu()[0]
            - intersection
    )

    if union == 0:
        return float("nan")
    else:
        return float(intersection) / float(max(union, 1))


# https://github.com/pytorch/pytorch/issues/1249
def dice_coefficient(pred, target):
    score = np.zeros(pred.shape[1])
    for classNum in range(pred.shape[1]):
        c = confusion_matrix(target[:, classNum].long().contiguous().view(-1).cpu().numpy(),
                             pred[:, classNum].contiguous().long().view(-1).cpu().numpy(), labels=[0, 1])
        TP = np.diag(c)
        FP = c.sum(axis=0) - np.diag(c)
        FN = c.sum(axis=1) - np.diag(c)
        TN = c.sum() - (FP + FN + TP)
        # print((2*TP)/(2*TP + FP + FN ))
        score[classNum] = (2 * TP[1] + 1e-10) / (2 * TP[1] + FP[1] + FN[1] + 1e-10)
    return score


def make_loader(twisted_labels, fake_labels, one, zero, bs=24):
    images = torch.cat([twisted_labels, fake_labels], 0)
    labels = torch.cat([one, zero], 0)
    discriminatorDataset = TensorDataset(images, labels)
    discriminatorDataLoader = DataLoader(discriminatorDataset, batch_size=bs, shuffle=True, num_workers=4,
                                         pin_memory=True)
    return discriminatorDataLoader
