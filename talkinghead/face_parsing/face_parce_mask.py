#!/usr/bin/python
# -*- encoding: utf-8 -*-
from pathlib import Path
import random

import torch
import os
from .model import BiSeNet
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

class FaceParseMask:
    def __init__(self,place_model_on_cpu=False,run_on_cpu=False,device_id=0):
        self.device_id = device_id
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "cp", "79999_iter.pth")

        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.load_state_dict(torch.load(file_path))

        net = net.to("cpu") # fix gpu to gpu bug
        net = net.to(f"cuda:{device_id}")

        net.eval()
        self.net = net

        # source_algined = Path("/mnt/data2/face_swap_projects/project_qinaide/ramdisk/source_aligned")
        # file_list = os.listdir(source_algined)
        # random.shuffle(file_list)
        # pased = self.process_one_image(str(source_algined/"41958.jpg"))
        # cv2.imwrite("parsing_rgb1.png", pased*10)


    def process_one_image(self,image_path):
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        with torch.no_grad():
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            # img = img.cuda()
            img = img.to(f"cuda:{self.device_id}")

            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return parsing

    def process_one_face(self,pred_rgb):
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        with torch.no_grad():
            # img = Image.open(image_path)
            img = cv2.cvtColor(pred_rgb, cv2.COLOR_BGR2RGB)

            # 将图像从 numpy 数组转换为 PIL 图像
            img = Image.fromarray(img)
            # image = img.resize((512, 512), Image.BILINEAR)

            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            # img = img.cuda()
            img = img.to(f"cuda:{self.device_id}")

            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return parsing.astype(np.uint8)


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return vis_parsing_anno
    # return vis_im


def evaluate(image_path='./imgs/116.jpg', cp='cp/79999_iter.pth'):

    # if not os.path.exists(respth):
    #     os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(cp))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        img = Image.open(image_path)
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # print(parsing)
        # print(np.unique(parsing))

        # vis_parsing_maps(image, parsing, stride=1, save_im=False, save_path=osp.join(respth, dspth))
        return parsing


