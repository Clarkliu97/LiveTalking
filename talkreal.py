# General imports for talkreal.py
from io import BytesIO
import cv2
from av import AudioFrame, VideoFrame
import av
import asyncio
import threading
import time
import soundfile as sf
import numpy as np
import resampy
import torch
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
import sys
import os
import torchvision.transforms as transforms
from PIL import Image
import json
import ffmpeg
import librosa
import soundfile
import queue
import math
import random
from typing import Optional
import re

# add talkinghead to sys.path
sys.path.append('talkinghead')

# TalkingHead imports
from talkinghead.configs.default import get_cfg_defaults
from talkinghead.core.networks.diffusion_net import DiffusionNet
from talkinghead.core.networks.diffusion_util import NoisePredictor, VarianceSchedule
from talkinghead.core.utils import (
    crop_src_image,
    get_pose_params,
    get_video_style_clip,
    get_wav2vec_audio_window,
)
from talkinghead.generators.utils import get_netG, render_video
from talkinghead.face_parsing.enhance_talk_face_multiprocess import MergeHeadBody
from talkinghead.face_parsing.face_parser import FaceParser

from basereal import BaseReal
# TTS
from ttsreal import EdgeTTS # VolcTTS

# LLM
# from llm.LLM import ChatGPTLLM, QwenLLM # TODO: add LLM
# from chroma_db.chroma_db import ChromaDatabase
# from llm.prompt_builder import Prompt_Builder

class TalkReal:
    def __init__(self, opt, device) -> None:
        self.opt = opt

        # Load configs
        self.avatar = opt.avatar
        self.fps = opt.fps
        self.cfg_scale = opt.cfg_scale
        self.max_gen_length = opt.max_gen_length

        # load avatar
        self.project_path = f"workspace/{self.avatar}"
        self.src_face_image_path = f"{self.project_path}/frame_faces"
        self.src_full_image_path = f"{self.project_path}/frames"

        # load pose
        self.pose_path = f"{self.project_path}/full_pose.npy"
        pose_np = np.load(self.pose_path, allow_pickle=True)
        angles = pose_np[:, 224:227]
        translations = pose_np[:, 254:257]
        crop = pose_np[:, -3:]
        self.pose_params = np.concatenate((angles, translations, crop), axis=1)
        print('Pose Loaded')

        # load landmarks params
        self.param_list = self.read_numpy()
        print("Landmarks Loaded.")

        # load model configs
        self.device = device
        self.cfg = get_cfg_defaults()
        self.cfg.CF_GUIDANCE.SCALE = self.cfg_scale
        self.cfg.INFERENCE.CHECKPOINT = "talkinghead/checkpoints/denoising_network.pth"
        self.cfg.freeze()
        print("Model Loaded.")

        # load wav2vec2
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("talkinghead/checkpoints/wav2vec2",local_files_only=True)
        self.wav2vec_model = (
			Wav2Vec2Model.from_pretrained("talkinghead/checkpoints/wav2vec2",local_files_only=True)
			.eval()
			.to(device)
		)
        print("wav2vec2 loaded.")

        # load diffusion net
        self.diff_net = self.get_diff_net(self.cfg, self.device).to(self.device)
        self.renderer = get_netG("talkinghead/checkpoints/renderer.pt", self.device)

        style_clip_raw, style_pad_mask_raw = get_video_style_clip(
			self.pose_path, "", style_max_len=256, start_idx=0
		)
        self.style_clip = style_clip_raw.unsqueeze(0).to(device)
        self.style_pad_mask = (
            style_pad_mask_raw.unsqueeze(0).to(device)
            if style_pad_mask_raw is not None
            else None
        )
        print("diffusion net loaded.")

        # load src face images
        self.loaded_src_face = self.load_src_imgs(self.src_face_image_path)
        self.src_face_list = torch.stack(self.loaded_src_face,dim=0)
        assert len(self.src_face_list) == self.pose_params.shape[0]
        print("src face images loaded.")

        # load src full images
        self.loaded_src_full = []
        for i in range(len(self.src_face_list)):
            img = cv2.imread(os.path.join(self.src_full_image_path, f"frame_{i+1:05d}.png"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.loaded_src_full.append(img)
            if i % 100 == 0:
                print(f"loaded {i+1} images.")
        print("src full images loaded.")

        # load crop info
        self.crop_info_path = f"{self.project_path}/frame_faces/config.json"
        with open(self.crop_info_path, 'r') as f:
            self.crop_info = json.load(f)
        self.x1 = self.crop_info['x1']
        self.y1 = self.crop_info['y1']
        self.x2 = self.crop_info['x2']
        self.y2 = self.crop_info['y2']
        self.crop_width = self.x2 - self.x1
        print("crop info loaded.")

        # load face parser
        self.parser = FaceParser()
        print("face parser loaded.")

        # prepare queues
        self.generated_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.video_queue = queue.Queue()
        print("queues created.")

        # load tts
        self.tts = EdgeTTS(opt, self)
        print("tts loaded.")

        # load llm
        # self.llm = ChatGPTLLM(self)
        # self.llm = QwenLLM(self)
        print("llm loaded.")

        # cut off event
        self.cut_off_event = threading.Event()
        threading.Thread(target=self.cut_off_thread, daemon=True).start()

        # start generation process
        self.gen_stop_event = threading.Event()
        self.gen_cut_off_event = threading.Event()
        self.gen_stop_event.clear()
        self.gen_process = threading.Thread(target=self.generate, args=(self.generated_queue, self.audio_queue, self.gen_stop_event, self.gen_cut_off_event))

        self.gen_process.daemon = True # Ensures the process will exit when the main thread exits
        self.gen_process.start()

        self.audio_tracks = []
        self.video_tracks = []

        # # debug only: feed audio to audio queue in loop
        # self.feed_process = threading.Thread(target=self.feed_audio_auto_loop, args=())
        # self.feed_process.daemon = True
        # self.feed_process.start()


    def load_src_imgs(self,src_dir):
        src_img_path = src_dir
        file_list = [n for n in os.listdir(src_img_path) if n[-4:] == ".jpg" or n[-4:] == ".png"]
        file_list.sort()
        src_img_tensor_list = []
        for f_name in file_list:
            frame = cv2.imread(os.path.join(src_img_path, f_name))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            src_img_raw = Image.fromarray(frame)
            image_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
                ]
            )
            src_img_tensor = image_transform(src_img_raw)
            src_img_tensor_list.append(src_img_tensor)
        return src_img_tensor_list
    

    def read_numpy(self):
        # read all landmark .npy file from the face folder
        file_list = [n for n in os.listdir(self.src_face_image_path) if n[-4:] == ".npy"]
        file_list.sort()
        param_list = []
        for f_name in file_list:
            param = np.load(os.path.join(self.src_face_image_path, f_name), allow_pickle=True)
            param_list.append(param)
        return param_list
    

    @torch.no_grad()
    def get_diff_net(self,cfg, device):
        diff_net = DiffusionNet(
            cfg=cfg,
            net=NoisePredictor(cfg),
            var_sched=VarianceSchedule(
                num_steps=cfg.DIFFUSION.SCHEDULE.NUM_STEPS,
                beta_1=cfg.DIFFUSION.SCHEDULE.BETA_1,
                beta_T=cfg.DIFFUSION.SCHEDULE.BETA_T,
                mode=cfg.DIFFUSION.SCHEDULE.MODE,
            ),
        )
        checkpoint = torch.load(cfg.INFERENCE.CHECKPOINT, map_location=device)
        model_state_dict = checkpoint["model_state_dict"]
        diff_net_dict = {
            k[9:]: v for k, v in model_state_dict.items() if k[:9] == "diff_net."
        }
        diff_net.load_state_dict(diff_net_dict, strict=True)
        diff_net.eval()

        return diff_net
    

    def cut_off_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.create_task(self.cut_off_loop(loop))
        print("run cut_off_loop in thread")
        loop.run_forever()


    async def cut_off_loop(self, loop: asyncio.AbstractEventLoop):
        while True:
            await loop.run_in_executor(None, self.cut_off_event.wait)
            await self.cut_off()
            self.cut_off_event.clear()


    async def cut_off(self):
        asyncio.create_task(self.llm.cut_off())
    

    def put_echo_msg_txt(self, msg):
        self.tts.put_msg_txt(msg)


    def put_chat_msg_txt(self, msg):
        self.llm.put_msg_txt(msg)


    async def parent_cut_off(self):
        self.audio_queue.queue.clear()


    def feed_audio(self, soundfile, sample_rate):
        audio_data = resampy.resample(soundfile, sample_rate, 16000)
        self.audio_queue.put(audio_data)


    def feed_audio_auto_loop(self):
        # DEBUG only
        audio_path = "tmp.wav"
        # feed audio to audio queue once every 70s with initial delay of 20s
        while True:
            # initial delay
            time.sleep(10)
            # feed audio to queue
            audio_data, sample_rate = sf.read(audio_path)
            # resample to 16k
            audio_data = resampy.resample(audio_data, sample_rate, 16000)
            self.audio_queue.put(audio_data)
            # wait
            time.sleep(50)


    def generate(self, generated_queue, audio_queue, gen_stop_event, gen_cut_off_event):
        index_list = []
        audio = None

        # generate frames TODO:
        pass
