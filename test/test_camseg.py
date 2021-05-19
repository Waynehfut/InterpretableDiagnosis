# -*- coding: utf-8 -*-
import os
import torch
from torchvision import transforms
from models.resegnet import ResegNet
from models.camseg import CasSeg
from utils.vis import fetch_cam, fetch_seg

if __name__ == '__main__':

    # seg
    img_path = '../test/'
    img_name = 'EDD2020_INAB0024.jpg'
    seg_type = 'seg'
    if not os.path.exists(img_path + seg_type):
        os.mkdir(img_path + seg_type)
    save_img_name = '{}/{}'.format(seg_type, img_name[:-3])
    model_dict_path = '../runs/InterGI_ResegNet_EndoCV__[train]/ResegNet_mean_best.pt'
    M = 512
    use_gpu = torch.cuda.is_available()
    tform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    seg_model = ResegNet(3)
    seg_contours = fetch_seg(seg_model, model_dict_path, img_path + img_name, M, tform, use_gpu,
                             img_path + save_img_name)
    print("Test seg OK")

    # cam model load
    cam_model = CasSeg(3)
    stated_dict = torch.load(
        '../runs/InterGI_CasSeg_EndoCV__[train]/CasSeg_best.pt')
    target_layer = cam_model.features[-1]

    cam_type = 'cam'
    if not os.path.exists(img_path + cam_type):
        os.mkdir(img_path + cam_type)
    save_img_name = '{}/{}'.format(cam_type, img_name[:-3])
    cam_contours = fetch_cam(cam_model, target_layer, img_path +
                             img_name, img_path + save_img_name, stated_dict)
    print("Test cam OK")
