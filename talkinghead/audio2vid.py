import os
import hashlib
import subprocess
import time
from PIL import Image
import cv2
import numpy as np
import torch
import torchaudio
import torchvision
import transformers
import copy
import json
import resampy
import soundfile as sf
import math
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
import torchvision.transforms as transforms
from configs.default import get_cfg_defaults
from core.networks.diffusion_net import DiffusionNet
from core.networks.diffusion_util import NoisePredictor, VarianceSchedule
from core.utils import (
    crop_src_image,
    get_pose_params,
    get_video_style_clip,
    get_wav2vec_audio_window,
)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'face_parsing'))

from face_parsing.enhance_talk_face_multiprocess import MergeHeadBody
from generators.utils import get_netG, render_video, render_video_generator
from talkinghead.face_parsing.face_parser import FaceParser

print(f"transformers version:{transformers.__version__}")
class TalkingPuppet:
	def __init__(self, id):
		self.id = id
		self.project_path = f"workspace/{self.id}"
		self.src_face_image_path = f"{self.project_path}/frame_faces"
		self.src_full_image_path = f"{self.project_path}/frames"

		# load pose
		self.pose_path = f"{self.project_path}/full_pose.npy"
		pose_np = np.load(self.pose_path, allow_pickle=True)
		angles = pose_np[:, 224:227]
		translations = pose_np[:, 254:257]
		crop = pose_np[:, -3:]
		self.pose_params = np.concatenate((angles, translations, crop), axis=1)
		self.max_gen_len = 30000 
		print("poses loaded.")

		# load landmarks params
		self.param_list = self.read_numpy()
		print("landmarks loaded.")

		# load cfg
		self.cfg_scale = 0.0
		self.device = device = torch.device("cuda")
		self.cfg = get_cfg_defaults()
		self.cfg.CF_GUIDANCE.SCALE = self.cfg_scale
		self.cfg.INFERENCE.CHECKPOINT = 'talkinghead/checkpoints/denoising_network.pth'
		self.cfg.freeze()
		print("cfg loaded.")

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

		# create tmp folder 
		tmp_dir = f"tmp_data"
		os.makedirs(tmp_dir, exist_ok=True)
		self.tmp_dir = tmp_dir

		# load face parser
		self.parser = FaceParser()
		print("face parser loaded.")
		print("TalkingPuppet init done.")
		print("********************************************************")

	def pose_param_convert_reverse_indices(self,reverse_index_list:list):
		full_pose_len = len(self.src_face_list)
		result = [full_pose_len-x-1 for x in reverse_index_list]
		return result

	def load_pose_params_indices(self,need_length, src_img_index, is_reverse_src_img):
		left_size = len(self.src_face_list) - src_img_index
		if left_size >= need_length:
			result = list(range(src_img_index, src_img_index + need_length))
			if is_reverse_src_img:
				result = self.pose_param_convert_reverse_indices(result)
			src_img_index += need_length
			if src_img_index == len(self.src_face_list):
				src_img_index = 0
				is_reverse_src_img = not is_reverse_src_img
		else:
			need_more_len = need_length - left_size
			result = list(range(src_img_index, len(self.src_face_list)))
			if is_reverse_src_img:
				result = self.pose_param_convert_reverse_indices(result)

			is_reverse_src_img = not is_reverse_src_img

			cycle_number = need_more_len // len(self.src_face_list)
			cycle_left_num = need_more_len % len(self.src_face_list)
			cycle_idx = 0
			while cycle_idx < cycle_number:
				this_idx_list = list(range(0, len(self.src_face_list)))
				if is_reverse_src_img:
					this_idx_list = self.pose_param_convert_reverse_indices(this_idx_list)
				is_reverse_src_img = not is_reverse_src_img
				cycle_idx += 1
				result += this_idx_list

			this_idx_list = list(range(0, cycle_left_num))
			if is_reverse_src_img:
				this_idx_list = self.pose_param_convert_reverse_indices(this_idx_list)
			result += this_idx_list
			src_img_index = cycle_left_num

		return result

	def read_numpy(self):
		# read all landmark .npy file from the face folder
		file_list = [n for n in os.listdir(self.src_face_image_path) if n[-4:] == ".npy"]
		file_list.sort()
		param_list = []
		for f_name in file_list:
			param = np.load(os.path.join(self.src_face_image_path, f_name), allow_pickle=True)
			param_list.append(param)
		return param_list
	
	def load_src_imgs(self,src_dir):
		src_img_path = src_dir
		file_list = [n for n in os.listdir(src_img_path) if n[-4:] == ".jpg" or n[-4:] == ".png"]
		file_list.sort()
		src_img_list = []
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
			src_img = image_transform(src_img_raw)
			src_img_list.append(src_img)
		return src_img_list

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

	@torch.no_grad()
	def inference_one_video(self,
			cfg,
			audio_raw,
			diff_net,
			device,
			max_audio_len=None,
			sample_method="ddim",
			ddim_num_step=10,
	):
		# audio_raw = audio_data = np.load(audio_path)
		#
		# if max_audio_len is not None:
		# 	audio_raw = audio_raw[: max_audio_len * 50]
		gen_num_frames = len(audio_raw) // 2

		audio_win_array = get_wav2vec_audio_window(
			audio_raw,
			start_idx=0,
			num_frames=gen_num_frames,
			win_size=cfg.WIN_SIZE,
		)

		audio_win = torch.tensor(audio_win_array).to(device)
		audio = audio_win.unsqueeze(0)


		gen_exp_stack = diff_net.sample(
			audio,
			self.style_clip,
			self.style_pad_mask,
			output_dim=cfg.DATASET.FACE3D_DIM,
			use_cf_guidance=cfg.CF_GUIDANCE.INFERENCE,
			cfg_scale= 1.0, # cfg.CF_GUIDANCE.SCALE,
			sample_method=sample_method,
			ddim_num_step=ddim_num_step,
		)
		gen_exp = gen_exp_stack[0].cpu().numpy()

		return gen_exp

	def process_wav(self, wav_path, sub_project_path):
		# read sound file
		audio, sr = sf.read(wav_path)
		if sr != 16000:
			audio = resampy.resample(audio, sr, 16000)
		
		inputs = self.wav2vec_processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
		inputs = inputs.input_values.to(self.device).float()
		with torch.no_grad():
			audio_embedding = self.wav2vec_model(inputs, return_dict=False)[0]
		audio_embedding = audio_embedding[0].cpu().numpy()

		total_audio_frame_number = len(audio) // 320 # 320 samples per audio frame
		total_frame_number = total_audio_frame_number // 2 # 640 samples per video frame

		# pad the audio embedding to match the number of frames
		audio_embedding = np.pad(audio_embedding, ((0, total_audio_frame_number - len(audio_embedding)), (0, 0)), mode='edge')

		# save wav_16k 
		wav_16k_path = f"{sub_project_path}/audio_16k.wav"
		sf.write(wav_16k_path, audio, 16000)

		return audio_embedding, total_frame_number, wav_16k_path
	
	def _to_mp4(self, width, height, output_dir, shared_list):
		fourcc = cv2.VideoWriter.fourcc(*"mp4v")
		video = cv2.VideoWriter(output_dir, fourcc, 25, (width, height))
		counter = 0
		print("mp4 merge start")
		while True:
			if counter == len(shared_list):
				break
			if shared_list[counter] is not None:
				t0 = time.time()
				frame = shared_list[counter]
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				video.write(frame)
				counter += 1
				# print(f"frame {counter} written to mp4 in {time.time()-t0} seconds")
			else:
				# time.sleep(0)
				pass
		t0 = time.time()
		video.release()

		# mp4v to h264
		# os.system(f"ffmpeg -loglevel error -y -i {output_dir} -c:v libx264 -c:a aac -strict experimental {output_dir.replace('.mp4', '_h264.mp4')}")

		print(f"video saved in {time.time()-t0} seconds")
		print("Video saved")

	def _to_mp4_progress(self, width, height, output_dir, shared_list, progress_queue):
		fourcc = cv2.VideoWriter.fourcc(*"mp4v")
		video = cv2.VideoWriter(output_dir, fourcc, 25, (width, height))
		counter = 0
		total_frames = len(shared_list)
		print("mp4 merge start")
		while True:
			if counter == total_frames:
				break
			if shared_list[counter] is not None:
				t0 = time.time()
				frame = shared_list[counter]
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				video.write(frame)
				counter += 1
				# Calculate and put progress into the queue
				progress = (counter / total_frames) * 100
				progress_queue.put(progress)
			else:
				# time.sleep(0)
				pass
		t0 = time.time()
		video.release()
		# Ensure to put 100% progress at the end
		progress_queue.put(100)

	def monitor_progress(self, progress_queue, mp4_thread):
		while mp4_thread.is_alive() or not progress_queue.empty():
			try:
				progress = progress_queue.get()
				yield progress
			except queue.Empty:
				continue

	def merge_single_image(self, mask, landmark, img, i, idx, mp4_shared_list, mp4_shared_list_lock): 
		orig_pic = self.loaded_src_full[idx]
		orig_pic_np = np.array(orig_pic)
		full_box_mask = np.zeros((self.crop_width, self.crop_width), dtype=np.uint8)
		renzhong_cord = landmark[28]
		mask_start_height = int(renzhong_cord[1])
		box_mask_size = (self.crop_width - mask_start_height, self.crop_width)
		box_mask = self.parser.create_static_box_mask(box_mask_size, 0.3, (0, 0, 0, 0))
		box_mask = np.ones_like(box_mask)
		full_box_mask[mask_start_height:, :] = box_mask

		mask = np.zeros(full_box_mask.shape[:2], dtype=np.uint8)
		for j in range(1, 15):
			cv2.line(mask, (int(landmark[j][0]), int(landmark[j][1])), (int(landmark[j+1][0]), int(landmark[j+1][1])), (255, 255, 255), 1)
		cv2.line(mask, (int(landmark[15][0]), int(landmark[15][1])), (int(landmark[28][0]), int(landmark[28][1])), (255, 255, 255), 1)
		cv2.line(mask, (int(landmark[28][0]), int(landmark[28][1])), (int(landmark[1][0]), int(landmark[1][1])), (255, 255, 255), 1)
		index_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 28]
		cv2.fillPoly(mask, [landmark[index_list].astype(np.int32)], (255, 255, 255))

		talk_frame_mask = mask
		occlusion_mask = talk_frame_mask.clip(0, 1).astype(np.float32)
		talk_frame_mask = (cv2.GaussianBlur(occlusion_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2

		talk_vision_512 = cv2.resize(img, (self.crop_width, self.crop_width))
		talk_frame_mask_512 = cv2.resize(talk_frame_mask, (self.crop_width, self.crop_width))
		crop_mask = np.minimum.reduce([full_box_mask, talk_frame_mask_512])

		crop_mask = np.expand_dims(crop_mask, axis=-1)

		orig_face = orig_pic_np[self.y1:self.y2, self.x1:self.x2]

		merged_face = talk_vision_512 * crop_mask + (1 - crop_mask) * orig_face
		orig_pic_np[self.y1:self.y2, self.x1:self.x2] = merged_face

		# return orig_pic_np
		mp4_shared_list_lock.acquire()
		mp4_shared_list[i] = orig_pic_np
		mp4_shared_list_lock.release()

	def get_mask(self, imgs):
		for i, img in enumerate(imgs):
			mask = self.parser.create_occlusion_mask(img)
			yield mask

	def generate(self, wav_path):
		# subworkspace
		t1 = time.time()
		with open(wav_path, 'rb') as f:
			wav_data = f.read()
		sub_project_hash = hashlib.sha384(wav_data).hexdigest()
		sub_project_path = f"{self.project_path}/results/{sub_project_hash}"
		os.makedirs(sub_project_path, exist_ok=True)
		f.close()
		print(f"sub_project_path:{sub_project_path}")

		# process wav file
		t2 = time.time()
		audio_embedding, total_frame_number, wav_16k_path = self.process_wav(wav_path, sub_project_path)
		output_name = os.path.basename(wav_path)
		print(f"audio_embedding shape:{audio_embedding.shape}; total_frame_number:{total_frame_number}")

		# get index list
		t3 = time.time()
		index_list = self.load_pose_params_indices(total_frame_number, 0, False)
		index_list = np.array(index_list)
		pose_params = self.pose_params[index_list]
		src_img_list = self.src_face_list[index_list]
		print(f"pose_params shape:{pose_params.shape}; src_img_list shape:{src_img_list.shape}")

		with torch.no_grad():
			# generate face motion
			t4 = time.time()
			gen_exp = self.inference_one_video(
				self.cfg,
				audio_embedding,
				self.diff_net,
				self.device,
				max_audio_len=self.max_gen_len,
			)
			face_motion_np = np.concatenate((gen_exp, pose_params), axis=1)
			print(f"face_motion_np shape:{face_motion_np.shape}")

			# render video
			t5 = time.time()
			imgs = render_video(
				self.renderer,
				src_img_list,
				face_motion_np,
				self.device,
			)
			# mask tasks
			t6 = time.time()
			masks_iter = self.get_mask(imgs)

			# to mp4
			t7 = time.time()
			silent_video_path = f"{sub_project_path}/{output_name}_silent.mp4"
			height, width, _ = self.loaded_src_full[0].shape
			mp4_shared_list = [None] * len(imgs)
			mp4_shared_list_lock = threading.Lock()
			mp4_thread = threading.Thread(target=self._to_mp4, args=(width, height, silent_video_path, mp4_shared_list))
			mp4_thread.daemon = True
			mp4_thread.start()

			# merge images
			param_list = copy.deepcopy(self.param_list)
			param_list = [param_list[i] for i in index_list]
			loaded_src_full = copy.deepcopy(self.loaded_src_full)
			loaded_src_full = [loaded_src_full[i] for i in index_list]

			with ThreadPoolExecutor(max_workers=24) as executor: 
				futures = []
				for i, idx in enumerate(index_list):
					futures.append(executor.submit(self.merge_single_image, next(masks_iter), param_list[i], imgs[i], i, idx, mp4_shared_list, mp4_shared_list_lock))

			for future in futures:
				future.result()
			mp4_thread.join()

			t8 = time.time()
				
			output_video_path = f"{sub_project_path}/{output_name}.mp4"

			os.system(f"ffmpeg -loglevel error -y -i {silent_video_path} -i {wav_16k_path} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {output_video_path}")

			t9 = time.time()

			print(f"{output_video_path};\ntotal time:{t9-t1};\nsubworkspace:{t2-t1};\nprocess wav file:{t3-t2};\nget index list:{t4-t3};\ngenerate face motion:{t5-t4};\nrender video:{t6-t5};\nmask tasks:{t7-t6};\nto_mp4:{t8-t7};\nffmpeg:{t9-t8}")

			# clear vram cache
			torch.cuda.empty_cache()

			return output_video_path
			
	def generate_generator(self, wav_path): 
		progress = 0
		
		# subworkspace
		t1 = time.time()
		with open(wav_path, 'rb') as f:
			wav_data = f.read()
		sub_project_hash = hashlib.sha384(wav_data).hexdigest()
		sub_project_path = f"{self.project_path}/results/{sub_project_hash}"
		os.makedirs(sub_project_path, exist_ok=True)
		f.close()
		print(f"sub_project_path:{sub_project_path}")
		progress += 1
		yield progress

		# process wav file
		t2 = time.time()
		audio_embedding, total_frame_number, wav_16k_path = self.process_wav(wav_path, sub_project_path)
		output_name = os.path.basename(wav_path)
		print(f"audio_embedding shape:{audio_embedding.shape}; total_frame_number:{total_frame_number}")
		progress += 1
		yield progress

		# get index list
		t3 = time.time()
		index_list = self.load_pose_params_indices(total_frame_number, 0, False)
		index_list = np.array(index_list)
		pose_params = self.pose_params[index_list]
		src_img_list = self.src_face_list[index_list]
		print(f"pose_params shape:{pose_params.shape}; src_img_list shape:{src_img_list.shape}")
		progress += 1
		yield progress

		with torch.no_grad():
			# generate face motion
			t4 = time.time()
			gen_exp = self.inference_one_video(
				self.cfg,
				audio_embedding,
				self.diff_net,
				self.device,
				max_audio_len=self.max_gen_len,
			)
			face_motion_np = np.concatenate((gen_exp, pose_params), axis=1)
			print(f"face_motion_np shape:{face_motion_np.shape}")
			progress += 7 
			yield progress

			# render video
			t5 = time.time()
			imgs_gen = render_video_generator(
				self.renderer,
				src_img_list,
				face_motion_np,
				self.device,
			)
			p = 0
			while True:
				try:
					p = next(imgs_gen)
					yield 10 + 0.1 * p  # Adjusted progress calculation
				except StopIteration as e:
					imgs = e.value  # Capture the final returned value
					break

			# mask tasks
			t6 = time.time()
			masks_iter = self.get_mask(imgs)
			masks = []
			for i, mask in enumerate(masks_iter):
				masks.append(mask)
				progress = 20 + 50 * i / len(imgs) # 20 + 0.5 * 100 = 70
				yield progress

			# to mp4
			t7 = time.time()
			mp4_progress = 0
			silent_video_path = f"{sub_project_path}/{output_name}_silent.mp4"
			height, width, _ = self.loaded_src_full[0].shape
			mp4_shared_list = [None] * len(imgs)
			mp4_shared_list_lock = threading.Lock()
			mp4_progress_queue = queue.Queue()
			mp4_thread = threading.Thread(target=self._to_mp4_progress, args=(width, height, silent_video_path, mp4_shared_list, mp4_progress_queue))
			mp4_thread.daemon = True
			mp4_thread.start()

			# merge images
			param_list = copy.deepcopy(self.param_list)
			param_list = [param_list[i] for i in index_list]
			loaded_src_full = copy.deepcopy(self.loaded_src_full)
			loaded_src_full = [loaded_src_full[i] for i in index_list]

			with ThreadPoolExecutor(max_workers=24) as executor: 
				futures = []
				for i, idx in enumerate(index_list):
					futures.append(executor.submit(self.merge_single_image, masks[i], param_list[i], imgs[i], i, idx, mp4_shared_list, mp4_shared_list_lock))

			for future in futures:
				future.result()

			for progress in self.monitor_progress(mp4_progress_queue, mp4_thread):
				progress = 70 + 20 * progress / 100 # 70 + 0.2 * 100 = 90
				yield progress
			mp4_thread.join()

			t8 = time.time()

			output_video_path = f"{sub_project_path}/{output_name}.mp4"

			os.system(f"ffmpeg -loglevel error -y -i {silent_video_path} -i {wav_16k_path} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {output_video_path}")
			# mp4v to h264
			output_video_path_h264 = output_video_path.replace('.mp4', '_h264.mp4')
			os.system(f"ffmpeg -loglevel error -y -i {output_video_path} -c:v libx264 -c:a aac -strict experimental {output_video_path_h264}")
			
			t9 = time.time()
			print(f"{output_video_path_h264};\ntotal time:{t9-t1};\nsubworkspace:{t2-t1};\nprocess wav file:{t3-t2};\nget index list:{t4-t3};\ngenerate face motion:{t5-t4};\nrender video:{t6-t5};\nmask tasks:{t7-t6};\nto_mp4:{t8-t7};\nffmpeg:{t9-t8}")

			# clear vram cache
			torch.cuda.empty_cache()
			progress = 100
			yield progress

			return output_video_path_h264

if __name__ == "__main__":
	talking_puppet = TalkingPuppet('92e87ec85560e81d70fd8103c3fa60fa170be1e146372f92557d5d97a66a8a00380090c49e7f3dc50d85e2b3bfbd1b24')
	# talking_puppet.shutup()
	# talking_puppet.generate("inputs/audio_short.mp3")
	# talking_puppet.generate("data/audio/hushi_news.wav")
	# talking_puppet.generate("data/audio/trump_30s.aac")
	# talking_puppet.generate("test_submit.mp3")
	gen = talking_puppet.generate_generator("test_submit.mp3")
	while True:
		try:
			progress = next(gen)
			print(f"progress:{progress}")
		except StopIteration as e:
			output_video_path = e.value
			break
		except Exception as e:
			print(f"error:{e}")
			break
	print(f"output_video_path:{output_video_path}")