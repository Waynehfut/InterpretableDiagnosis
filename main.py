# -*- coding: utf-8 -*-
import os
import time

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"  # for seg
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"  # for class
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # for test
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import shutil
import copy
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
import dataload.augmentation as aug

from models.resegnet import ResegNet
from models.SUMNet import SUMNet
from models.camseg import CasSeg

from config import ex
from utils.utils import set_seed, BCEDiceLoss, MetricTracker, dice_coefficient, jaccard_index
from dataload.endocv20 import eddLoader


def seg_train(trainDataLoader, validDataLoader, net, optimizer, criterion, savePath, scheduler, epochs, model_name):
    trainLoss = []
    validLoss = []
    trainDiceCoeff = []
    validDiceCoeff = []
    start = time.time()
    bestValidDice = np.zeros(5)

    for epoch in range(epochs):
        epochStart = time.time()
        trainRunningLoss = 0
        validRunningLoss = 0
        trainBatches = 0
        validBatches = 0
        trainDice = np.zeros(5)
        validDice = np.zeros(5)
        bestMeanValidDice = 0

        net.train(True)
        for data in tqdm(trainDataLoader, leave=False):
            inputs, labels, clas_label = data
            labels = labels / 255.0

            inputs = inputs.cuda()
            labels = labels.cuda()

            BE_lbl, suspicious_lbl, HGD_lbl, cancer_lbl, polyp_lbl = labels[:, 0, :, :], \
                                                                     labels[:, 1, :, :], \
                                                                     labels[:, 2, :, :], \
                                                                     labels[:, 3, :, :], \
                                                                     labels[:, 4, :, :]
            probs = net(inputs)

            BE_prob, suspicious_prob, HGD_prob, cancer_prob, polyp_prob = probs[:, 0, :, :], \
                                                                          probs[:, 1, :, :], \
                                                                          probs[:, 2, :, :], \
                                                                          probs[:, 3, :, :], \
                                                                          probs[:, 4, :, :]

            BE_prob = BE_prob.cuda()
            suspicious_prob = suspicious_prob.cuda()
            HGD_prob = HGD_prob.cuda()
            cancer_prob = cancer_prob.cuda()
            polyp_prob = polyp_prob.cuda()
            # BE_loss = criterion(BE_prob, BE_lbl)
            suspicious_loss = criterion(suspicious_prob, suspicious_lbl)
            HGD_loss = criterion(HGD_prob, HGD_lbl)
            cancer_loss = criterion(cancer_prob, cancer_lbl)
            polyp_loss = criterion(polyp_prob, polyp_lbl)

            # loss = BE_loss + suspicious_loss + HGD_loss + cancer_loss + polyp_loss
            loss = suspicious_loss + HGD_loss + cancer_loss + polyp_loss
            ##
            preds = (probs > 0.5).float()
            # print(np.unique(preds.cpu()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trainRunningLoss += loss.item()

            trainDice += dice_coefficient(preds, labels)

            trainBatches += 1

        trainLoss.append(trainRunningLoss / trainBatches)

        trainDiceCoeff.append(trainDice / trainBatches)

        # model validation
        net.train(False)
        for data in tqdm(validDataLoader, leave=False):
            inputs, labels, class_labels = data
            labels = labels / 255.0
            BE_lbl, suspicious_lbl, HGD_lbl, cancer_lbl, polyp_lbl = labels[:, 0, :, :], \
                                                                     labels[:, 1, :, :], \
                                                                     labels[:, 2, :, :], \
                                                                     labels[:, 3, :, :], \
                                                                     labels[:, 4, :, :]

            inputs = inputs.cuda()
            labels = labels.cuda()
            BE_lbl = BE_lbl.cuda()
            suspicious_lbl = suspicious_lbl.cuda()
            HGD_lbl = HGD_lbl.cuda()
            cancer_lbl = cancer_lbl.cuda()
            polyp_lbl = polyp_lbl.cuda()
            probs = net(inputs)
            BE_prob, suspicious_prob, HGD_prob, cancer_prob, polyp_prob = probs[:, 0, :, :], \
                                                                          probs[:, 1, :, :], \
                                                                          probs[:, 2, :, :], \
                                                                          probs[:, 3, :, :], \
                                                                          probs[:, 4, :, :]
            BE_prob = BE_prob.cuda()
            suspicious_prob = suspicious_prob.cuda()
            HGD_prob = HGD_prob.cuda()
            cancer_prob = cancer_prob.cuda()
            polyp_prob = polyp_prob.cuda()
            BE_prob, suspicious_prob, HGD_prob, cancer_prob, polyp_prob = probs[:, 0, :, :], \
                                                                          probs[:, 1, :, :], \
                                                                          probs[:, 2, :, :], \
                                                                          probs[:, 3, :, :], \
                                                                          probs[:, 4, :, :]
            # BE_loss = criterion(BE_prob, BE_lbl)
            suspicious_loss = criterion(suspicious_prob, suspicious_lbl)
            HGD_loss = criterion(HGD_prob, HGD_lbl)
            cancer_loss = criterion(cancer_prob, cancer_lbl)
            polyp_loss = criterion(polyp_prob, polyp_lbl)

            # loss = BE_loss + suspicious_loss + HGD_loss + cancer_loss + polyp_loss
            loss = suspicious_loss + HGD_loss + cancer_loss + polyp_loss

            preds = (probs > 0.5).float()

            # validDice += dice_coefficient(preds, labels).item()
            # for classNum in range(5):
            #     validDice[classNum] += dice_coefficient(preds[:,classNum],labels[:,classNum])[1]
            validDice += dice_coefficient(preds, labels)

            validRunningLoss += loss.item()
            validBatches += 1
        scheduler.step(validRunningLoss / validBatches)
        # if validBatches == 4:
        #     break
        validLoss.append(validRunningLoss / validBatches)
        validDiceCoeff.append(validDice / validBatches)

        if validDice[0] > bestValidDice[0]:
            bestValidDice[0] = validDice[0]
            torch.save(net.state_dict(), savePath + '/' + model_name + '_class0_best.pt')
        if validDice[1] > bestValidDice[1]:
            bestValidDice[1] = validDice[1]
            torch.save(net.state_dict(), savePath + '/' + model_name + '_class1_best.pt')
        if validDice[2] > bestValidDice[2]:
            bestValidDice[2] = validDice[2]
            torch.save(net.state_dict(), savePath + '/' + model_name + '_class2_best.pt')
        if validDice[3] > bestValidDice[3]:
            bestValidDice[3] = validDice[3]
            torch.save(net.state_dict(), savePath + '/' + model_name + '_class3_best.pt')
        if validDice[4] > bestValidDice[4]:
            bestValidDice[4] = validDice[4]
            torch.save(net.state_dict(), savePath + '/' + model_name + '_class4_best.pt')
        if np.mean(validDice) > bestMeanValidDice:
            bestMeanValidDice = np.mean(validDice)
            torch.save(net.state_dict(), savePath + '/' + model_name + '_mean_best.pt')

        epochEnd = time.time() - epochStart

        print('Epoch: {:.0f}/{:.0f} | Train Loss: {:.3f} | Valid Loss: {:.3f} |' \
              .format(epoch + 1, epochs, trainRunningLoss / trainBatches, validRunningLoss / validBatches))
        print('Train Dice : {:.3f},{:.3f},{:.3f},{:.3f} '.format(
            trainDice[1] / trainBatches,
            trainDice[2] / trainBatches,
            trainDice[3] / trainBatches,
            trainDice[4] / trainBatches))
        print('Valid Dice : {:.3f},{:.3f},{:.3f},{:.3f} '.format(
            validDice[1] / validBatches,
            validDice[2] / validBatches,
            validDice[3] / validBatches,
            validDice[4] / validBatches))
        print('Time: {:.0f}m {:.0f}s'.format(epochEnd // 60, epochEnd % 60))
        # break
    end = time.time() - start
    print('Training completed in {:.0f}m {:.0f}s'.format(end // 60, end % 60))

    torch.save(trainLoss, savePath + '/' + model_name + 'trainLoss.pt')
    torch.save(validLoss, savePath + '/' + model_name + 'validLoss.pt')
    torch.save(trainDiceCoeff, savePath + '/' + model_name + 'trainDiceCoeff.pt')
    torch.save(validDiceCoeff, savePath + '/' + model_name + 'validDiceCoeff.pt')


def class_train(trainDataLoader, validDataLoader, net, optimizer, criterion, savePath, scheduler, epochs, model_name,
                pretrain_dict):
    trainLoss = []
    validLoss = []
    trainACCCoff = []
    validACCCoff = []
    start = time.time()
    bestACC = 0.0
    # If pretrained, load seg model backbone
    if pretrain_dict:
        net_dict = net.state_dict()
        backbone_dict = {k: v for k, v in pretrain_dict.items() if k in net_dict}
        net_dict.update(backbone_dict)
        net.load_state_dict(net_dict)

    for epoch in range(epochs):
        epochStart = time.time()
        trainRunningLoss = 0
        validRunningLoss = 0
        trainBatches = 0
        validBatches = 0
        trainAccEpo = 0
        validAccEpo = 0
        net.train(True)
        for data in tqdm(trainDataLoader, leave=False):
            inputs, labels, clas_label = data
            inputs = inputs.cuda()
            clas_label = clas_label.float().cuda()
            prob_clas_label = net(inputs)
            prob_clas_label = prob_clas_label.cuda()
            train_loss = criterion(prob_clas_label, clas_label)
            preds = (prob_clas_label > 0.5)  # prob >0.5
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            trainRunningLoss += train_loss.item()
            trainAcc = torch.sum(preds == clas_label.data).cpu().numpy() / torch.sum(preds == preds).cpu().numpy()
            trainAccEpo += trainAcc
            trainBatches += 1
        trainLoss.append(trainRunningLoss / trainBatches)
        trainACCCoff.append(trainAccEpo / trainBatches)

        # validation
        net.train(False)
        for data in tqdm(validDataLoader, leave=False):
            inputs, labels, clas_label = data
            inputs = inputs.cuda()
            clas_label = clas_label.float().cuda()
            prob_clas_label = net(inputs)
            prob_clas_label = prob_clas_label.cuda()
            val_loss = criterion(prob_clas_label, clas_label)
            preds = (prob_clas_label > 0.5)
            validRunningLoss += val_loss.item()
            validAccEpo += torch.sum(preds == clas_label.data).cpu().numpy() / torch.sum(preds == preds).cpu().numpy()
            validBatches += 1
        scheduler.step(validRunningLoss / validBatches)
        validLoss.append(validRunningLoss / validBatches)
        validACCCoff.append(validAccEpo / validBatches)

        if (validAccEpo / validBatches) > bestACC:
            bestACC = validAccEpo
            torch.save(net.state_dict(), savePath + '/' + model_name + '_best.pt')

        epochEnd = time.time() - epochStart
        print('Epoch: {:.0f}/{:.0f} | Train Loss: {:.3f} | Train Acc:{:.3f} | Valid Loss: {:.3f} | Valid Acc:{:.3f} ' \
              .format(epoch + 1, epochs, trainRunningLoss / trainBatches, trainAccEpo / trainBatches,
                      validRunningLoss / validBatches, validAccEpo / validBatches, ))
        print('Time: {:.0f}m {:.0f}s'.format(epochEnd // 60, epochEnd % 60))
    end = time.time() - start
    torch.save(trainLoss, savePath + '/' + model_name + 'trainLoss.pt')
    torch.save(validLoss, savePath + '/' + model_name + 'validLoss.pt')
    torch.save(trainACCCoff, savePath + '/' + model_name + 'trainACCCoeff.pt')
    torch.save(validACCCoff, savePath + '/' + model_name + 'validACCCoeff.pt')
    print('Training completed in {:.0f}m {:.0f}s'.format(end // 60, end % 60))


@ex.automain
def main(_run, _config, _log):
    # setup configure
    _log.info(('Let us using {} GPUs'.format(torch.cuda.device_count())))
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')
    set_seed(_config['seed'])
    savePath = f'{_run.observers[0].dir}/snapshots'
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.set_num_threads(8)

    data_path = _config['path'][_config['dataset']]['data_dir']
    batch_size = _config['batch_size']
    data_aug = aug.Compose([aug.RandomRotate(10), aug.RandomHorizontallyFlip(0.5), aug.RandomHorizontallyFlip(0.5), \
                            aug.AdjustGamma(1.5), aug.AdjustSaturation(0.5), aug.AdjustHue(0.5),
                            aug.AdjustBrightness(0.5), aug.AdjustContrast(0.5), \
                            aug.RandomCrop(256, 256)])
    lr = _config['optima']['lr']
    gamma = _config['optima']['gamma']
    momentum = _config['optima']['momentum']
    weight_decay = _config['optima']['weight_decay']

    trainLoader = eddLoader(
        data_path,
        is_transform=True,
        split='train',
        img_size=(256, 256),
        augmentations=data_aug,
    )
    trainDataLoader = DataLoader(
        trainLoader,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
    )
    v_loader = eddLoader(
        data_path,
        is_transform=True,
        split='val',
        img_size=(256, 256),
        augmentations=data_aug,
    )
    validDataLoader = DataLoader(
        v_loader, batch_size=batch_size, num_workers=4
    )

    work_step = _config['step']
    if work_step == "Seg":
        model_name = _config['model']
        _log.info('###Train {} Model###'.format(model_name))
        if model_name == 'SUMNet':
            res_model = SUMNet()
        if model_name == 'ResegNet':
            res_model = ResegNet(3)

        res_model = nn.DataParallel(res_model.cuda())
        num_epochs = _config['epochs']
        criterion = nn.BCELoss()

        optimizer = Adam(res_model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

        seg_train(trainDataLoader, validDataLoader, res_model, optimizer, criterion, savePath, scheduler, num_epochs,
                  model_name)
    elif work_step == "Class":
        model_name = _config['model']
        is_pretrain = _config['mode']
        if is_pretrain != 'pre_trian':
            _log.info('###Train {} Model###'.format(model_name))
            pretrain_dict = None
        else:
            _log.info('###Pre_train {} Model###'.format(model_name))
            pretrain_dict = torch.load(_config['pre_pth_path'])

        cam_model = CasSeg(3)
        cam_model = nn.DataParallel(cam_model.cuda())
        num_epochs = _config['epochs']
        criterion = nn.BCEWithLogitsLoss()
        optimizer = SGD(cam_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
        print(cam_model)
        class_train(trainDataLoader, validDataLoader, cam_model, optimizer, criterion, savePath, scheduler, num_epochs,
                    model_name, pretrain_dict)
        # cam_model.train()


