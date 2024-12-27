import threading

import torch
import os

from tqdm import tqdm

from DFLIMG import DFLIMG
from model import BiSeNet

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from pathlib import Path

class FaceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = list(Path(image_dir).glob('*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, str(image_path)

class FaceParseMask:
    def __init__(self, place_model_on_cpu=False, run_on_cpu=False, device_id=0):
        self.device_id = device_id
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "cp", "79999_iter.pth")

        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.load_state_dict(torch.load(file_path))

        net = net.to("cpu")  # fix gpu to gpu bug
        net = net.to(f"cuda:{device_id}")

        net.eval()
        self.net = net

    def process_batch(self, image_dir, batch_size=8, output_dir="output"):
        to_tensor = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        dataset = FaceDataset(image_dir, transform=to_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for imgs, paths in tqdm(dataloader):
                imgs = imgs.to(f"cuda:{self.device_id}")
                outputs = self.net(imgs)[0]
                parsings = outputs.argmax(dim=1).cpu().numpy()

                # self.save_images(output_dir, parsings, paths)
                threading.Thread(target=self.save_images, args=(output_dir, parsings, paths)).start()


    def save_images(self, output_dir, parsings, paths):
        for parsing, path in zip(parsings, paths):
            if True:
                dfimg = DFLIMG.load(Path(path))
                parsing = parsing.astype(np.uint8)
                parsing[parsing == 16] = 0
                parsing[parsing == 14] = 0

                parsing = parsing.astype(np.float32)
                parsing[parsing > 0] = 1.0
                dfimg.set_xseg_mask(parsing)
                dfimg.save()
            else:
                file_name = os.path.basename(path)
                save_path = os.path.join(output_dir, f"{file_name}.png")
                parsing = parsing.astype(np.uint8)
                parsing[parsing == 16] = 0
                parsing[parsing == 14] = 0
                cv2.imwrite(save_path, parsing)

def get_mask_for_source():
    path = Path("/mnt/data/project_qinaide/tangkai/head_aligned")
    model = FaceParseMask()
    model.process_batch(str(path), batch_size=32, output_dir='/mnt/data2/face_swap_projects/project_qinaide/004/mask_debug')

def get_mask_for_episode():
    global model, x, aligned_dir
    # 示例使用
    # net = YourModel()  # 这里应该是你的模型实例
    model = FaceParseMask()
    root_dir = Path("/mnt/data2/face_swap_projects/project_qinaide/004/orig_frames")
    dir_list = os.listdir(root_dir)
    dir_list = [x for x in dir_list if os.path.isdir(root_dir / x)]
    for x in tqdm(dir_list):
        sub_dir = root_dir / x
        aligned_dir = sub_dir / "aligned_new"
        # model.process_batch('/home/xijing/disk1/face_swap_projects/qinaide/tangkai_frames_aligned', batch_size=8, output_dir='/home/xijing/disk1/face_swap_projects/qinaide/tangkai_frames_aligned_mask')
        model.process_batch(str(aligned_dir), batch_size=16, output_dir='/mnt/data2/face_swap_projects/project_qinaide/004/mask_debug')
get_mask_for_episode()
# img_mask = cv2.imread("/mnt/data2/face_swap_projects/project_qinaide/004/mask_debug/00003.jpg.png", cv2.IMREAD_GRAYSCALE)
# print(img_mask)
# get_mask_for_source()

