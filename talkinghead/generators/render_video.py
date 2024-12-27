import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image


def obtain_seq_index(index, num_frames, radius):
    seq = list(range(index - radius, index + radius + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq


@torch.no_grad()
def get_netG(checkpoint_path, device):
    import yaml

    from generators.face_model import FaceGenerator

    with open("generators/renderer_conf.yaml", "r") as f:
        renderer_config = yaml.load(f, Loader=yaml.FullLoader)

    renderer = FaceGenerator(**renderer_config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    renderer.load_state_dict(checkpoint["net_G_ema"], strict=False)

    renderer.eval()

    return renderer


@torch.no_grad()
def render_video(
    net_G,
    src_img_list,
    face_motion_np,
    # wav_path,
    # output_path,
    device,
    silent=False,
    semantic_radius=13,
    # fps=30,
    split_size=16,
    no_move=False,
    merger=None,
):
    """
    exp: (N, 73)
    """
    target_exp_seq = face_motion_np #np.load(exp_path)
    assert len(target_exp_seq) == len(src_img_list)
    if target_exp_seq.shape[1] == 257:
        exp_coeff = target_exp_seq[:, 80:144]
        angle_trans_crop = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9370641, 126.84911, 129.03864],
            dtype=np.float32,
        )
        target_exp_seq = np.concatenate(
            [exp_coeff, angle_trans_crop[None, ...].repeat(exp_coeff.shape[0], axis=0)],
            axis=1,
        )
        # (L, 73)
    elif target_exp_seq.shape[1] == 73:
        if no_move:
            target_exp_seq[:, 64:] = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9370641, 126.84911, 129.03864],
                dtype=np.float32,
            )
    else:
        raise NotImplementedError


    target_win_exps = []
    for frame_idx in range(len(target_exp_seq)):
        win_indices = obtain_seq_index(
            frame_idx, target_exp_seq.shape[0], semantic_radius
        )
        win_exp = torch.tensor(target_exp_seq[win_indices]).permute(1, 0)
        # (73, 27)
        target_win_exps.append(win_exp)

    target_exp_concat = torch.stack(target_win_exps, dim=0)
    target_splited_exps = torch.split(target_exp_concat, split_size, dim=0)
    src_img_splited = torch.split(src_img_list,split_size, dim=0)

    for win_exp,cur_src_img in zip(target_splited_exps,src_img_splited):
        win_exp = win_exp.to(device)
        cur_src_img = cur_src_img.to(device)

        output_dict = net_G(cur_src_img, win_exp)
        # imggg = output_dict["fake_image"].cpu().clamp_(-1, 1)
        imggg = output_dict["fake_image"].clamp_(-1, 1)
        transformed_gg = ((imggg + 1) / 2 * 255).to(torch.uint8).permute(0, 2, 3, 1)
        transformed_gg = transformed_gg.detach().cpu()
        # face_parser.create_occlusion_mask_batch(transformed_gg.numpy())
        face_numpy = transformed_gg.numpy()
        
        yield face_numpy
