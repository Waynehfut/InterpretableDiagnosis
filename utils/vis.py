# -*- coding: utf-8 -*-


import torch
import cv2
import numpy as np
from PIL import Image
from collections import OrderedDict

import torchvision.transforms.functional as TF
import tifffile as tiff
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def fetch_cam(model, layer, img_path, save_cam_path, stated_dict):
    # Avoid data parallel problem
    new_state_dict = OrderedDict()
    for k, v in stated_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    image = cv2.imread(img_path)
    rgb_img = np.float32(image) / 255
    input_tensor = TF.to_tensor(image)
    input_tensor.unsqueeze_(0)
    cam = GradCAM(model=model, target_layer=layer, use_cuda=True)
    target_category = 4
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(save_cam_path + '_cam.jpg', visualization)

    cam_mask = np.uint8(grayscale_cam * 255)
    thresh = 150
    ret, thresh_img = cv2.threshold(cam_mask, thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_mask = np.zeros((image.shape[0], image.shape[1]))
    cv2.drawContours(contours_mask, contours, -1, 255, 4)
    return contours_mask


def fetch_seg(net, net_dict_path, img_path, input_size, tform, use_gpu, save_img_path):
    # Avoid data parallel problem
    stated_dict = torch.load(net_dict_path)
    new_state_dict = OrderedDict()
    for k, v in stated_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)
    net.eval()
    if use_gpu:
        net.cuda()
    test_img = Image.open(img_path)
    original_img = cv2.imread(img_path)
    img_orig_size = test_img.size
    test_img = test_img.resize((input_size, input_size))
    test_inp = tform(test_img)
    if use_gpu:
        test_inp = test_inp.cuda()
    probs = net(test_inp.unsqueeze(0)).squeeze(0).cpu()
    preds = (probs > 0.52).float()
    preds[preds > 0] = 255
    pred_np = np.asarray(preds.numpy(), dtype=np.uint8)
    # model predict
    tiff.imwrite(save_img_path + 'seg_mask.tif', pred_np)

    # resize to original size
    pred_data = np.zeros((5, img_orig_size[1], img_orig_size[0]))

    # save pred_mask greyscale
    for classNum in range(5):
        pred_data[classNum] = np.array(Image.fromarray(pred_np[classNum]).resize(img_orig_size))
    pred_grey = pred_data.sum(axis=0)
    pred_grey = cv2.dilate(pred_grey, np.ones((15, 15), np.uint8))
    pred_grey = np.uint8(pred_grey)

    thresh = 100
    ret, thresh_img = cv2.threshold(pred_grey, thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(original_img, contours, -1, (0, 255, 0), 4)
    contours_mask = np.zeros((img_orig_size[1], img_orig_size[0]))
    cv2.drawContours(contours_mask, contours, -1, 255, 4)
    cv2.imwrite(save_img_path + 'seg_contours.jpg', original_img)
    return contours_mask