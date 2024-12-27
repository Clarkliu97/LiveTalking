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

# TTS
from ttsreal import VolcTTS

# LLM
# from llm.LLM import ChatGPTLLM, QwenLLM # TODO: add LLM
# from chroma_db.chroma_db import ChromaDatabase
# from llm.prompt_builder import Prompt_Builder