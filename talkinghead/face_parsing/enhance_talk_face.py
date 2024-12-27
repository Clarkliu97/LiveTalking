import json
import multiprocessing
import os
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from face_parsing.face_parser import FaceParser

class MergeHeadBody:
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir
        self.head_dir = os.path.join(workspace_dir, "frame_faces")
        self.body_dir = os.path.join(workspace_dir, "frames")
        self.face_parser = FaceParser()
        self.face_parser2 = FaceParser()

        self.body_pic_list = self.get_ext_file_list(self.body_dir)
        self.body_pic_list.sort()
        self.param_list = self.read_numpy()
        
        self.frame_name_patten = "frame_{:05d}.png"
        self.lock = threading.Lock()
        with open(os.path.join(self.head_dir,"config.json"), 'r') as fp:
            self.crop_info = json.load(fp)
        
        self.mask_list = []

        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()

        self.worker_thread = threading.Thread(target=self._process_tasks, args=(self.face_parser,))
        self.worker_thread.daemon = True  # Ensures the thread will exit when the main thread exits
        self.worker_thread.start()

        self.worker_thread2 = threading.Thread(target=self._process_tasks, args=(self.face_parser2,))
        self.worker_thread2.daemon = True  # Ensures the thread will exit when the main thread exits
        self.worker_thread2.start()

        self.mask_task_lock = threading.Lock()
        self.mask_task_list = []
        
        self.out_dir= os.path.join(workspace_dir, "merged")
        os.makedirs(self.out_dir, exist_ok=True)

        self.merged_list = []


    def _process_tasks(self, parser):
        while True:
            start_index, masks = self.task_queue.get()
            if start_index is None and masks is None:  # Use a sentinel value to exit the loop
                break
            # file_path = os.path.join(self.output_dir, file_name)
            result = parser.create_occlusion_mask_batch(masks)
            self.mask_list.append(result)
        
    def add_mask_task(self, start_index, masks):
        self.task_queue.put((start_index, masks))

    def get_result(self):
        return self.result_queue.get()
    
    def stop_mask_thread(self):
        t1 = time.time()
        self.task_queue.put((None, None))  # Add sentinel to stop the worker thread
        self.task_queue.put((None, None))  # Add sentinel to stop the worker thread
        self.worker_thread.join()
        self.worker_thread2.join()
        # self.mask_list = np.concatenate(self.mask_list,axis=0)
        print(f"waiting masks :{time.time()-t1}")

    def read_numpy(self):
        head_dir = Path(self.head_dir)
        file_list = self.get_ext_file_list(self.head_dir,[".npy"])
        file_list.sort()
        np_list = []
        for np_path in file_list:
            data = np.load(str(head_dir/np_path),allow_pickle=True)
            np_list.append(data)
        return np_list

    def get_ext_file_list(self, path,file_exts=('.jpg','.jpeg','.png')):
        selected_list = []
        file_list = os.listdir(path)
        root_path = Path(path)
        for f_name in file_list:
            # ddd = os.path.splitext("coeff")
            # print(ddd[1] in file_exts)
            file_path = root_path/f_name
            if os.path.isdir(file_path):
                continue
            extension = os.path.splitext(f_name)[1]
            if extension in file_exts: #== '.jpg' or extension == '.jpeg' or extension == '.png':
                selected_list.append(f_name)
        return selected_list
    
    def merge_single_image_process(self,mask,landmark,img,orig_jpg_pic_path, output_dir,file_name, idx):
        x1 = self.crop_info['x1']
        y1 = self.crop_info['y1']
        x2 = self.crop_info['x2']
        y2 = self.crop_info['y2']
        width = x2 - x1 # 439

        t_oo = time.time()
        talk_pic_np = img[:, :, ::-1]
        # landmark = self.param_list[idx]
        orig_pic_np = cv2.imread(os.path.join(self.body_dir, orig_jpg_pic_path))
        full_box_mask = np.zeros((width, width), np.float32) # 
        renzhong_cord = landmark[28]
        mask_start_height = int(renzhong_cord[1])
        box_mask_size = (width - mask_start_height, width)
        box_mask = self.face_parser.create_static_box_mask(box_mask_size, 0.3, (0, 0, 0, 0))
        full_box_mask[mask_start_height:, :] = box_mask

        talk_frame_mask = mask


        occlusion_mask = talk_frame_mask.transpose(0, 1, 2).clip(0, 1).astype(np.float32)
        talk_frame_mask = (cv2.GaussianBlur(occlusion_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2


        talk_vision_512 = cv2.resize(talk_pic_np, (width, width))
        talk_frame_mask_512 = cv2.resize(talk_frame_mask, (width, width))
        crop_mask = np.minimum.reduce([full_box_mask, talk_frame_mask_512])
        crop_mask = np.expand_dims(crop_mask, axis=-1)

        orig_face = orig_pic_np[y1:y2, x1:x2]

        merged_face = talk_vision_512 * crop_mask + (1 - crop_mask) * orig_face
        orig_pic_np[y1:y2, x1:x2] = merged_face

        # cv2.imwrite(f"{output_dir}/{file_name}", orig_pic_np)
        self.merged_list[idx] = orig_pic_np
        t_end = time.time()
        # print(f"pp total:{t_end - t_oo}")

    def merge_numpy(self, img_numpy, index_list, output_dir):
        self.stop_mask_thread()
        length = len(img_numpy)
        height, width, _ = cv2.imread(os.path.join(self.body_dir, self.body_pic_list[0])).shape
        self.merged_list = [None] * length
        if type(self.mask_list) == list:
            self.mask_list = np.concatenate(self.mask_list,axis=0)
        with ThreadPoolExecutor(max_workers=24) as executor:  # Adjust the number of workers as needed
            futures = [executor.submit(self.merge_single_image_process, self.mask_list[idx],self.param_list[idx], img_numpy[idx], self.body_pic_list[index_list[idx]], output_dir,self.frame_name_patten.format(idx), idx) for idx in range(len(img_numpy))]

        merge_thread = threading.Thread(target=self._to_mp4, args=(length, width, height, output_dir,))
        merge_thread.daemon = True  # Ensures the thread will exit when the main thread exits
        merge_thread.start()

        # Optional: Wait for all futures to complete if you need to ensure all work is done
        for future in futures:
            future.result()
        
        merge_thread.join()
        print("All threads joined")

def add_background_single_image(image, mask, background): 
    # Create an alpha channel from the mask
    alpha = mask.astype(image.dtype)

    # Normalize the alpha mask to [0, 1]
    alpha = alpha / 255.0

    # Convert images to float for blending
    image = image.astype(float)
    background = background.astype(float)

    # Make the alpha mask 3-channel
    alpha = cv2.merge([alpha, alpha, alpha])

    # Blend the images using the alpha mask
    blended = image * alpha + background * (1 - alpha)

    # Convert the result back to uint8
    blended = blended.astype(np.uint8)

    return blended

class MergeTask:
    def __init__(self, merger, idx, head, body, landmarkers, body_mask, background) -> None:
        self.merger = merger
        self.idx = idx
        self.head = head
        self.body = body
        self.landmarkers = landmarkers
        self.body_mask = body_mask
        self.background = background

        self.result = None
    
    def composite(self):
        if self.body_mask is None or self.background is None:
            return

        self.result = add_background_single_image(self.body, self.body_mask, self.background)

    def merge(self, face_parser):
        if self.head is None:
            self.result = self.body
            self.merger.handle_task_done(self)

            return

        landmark = self.landmarkers

        # mask = face_parser.create_occlusion_mask_batch(self.head)
        mask = face_parser.create_occlusion_mask(self.head)

        x1 = self.merger.crop_info['x1']
        y1 = self.merger.crop_info['y1']
        x2 = self.merger.crop_info['x2']
        y2 = self.merger.crop_info['y2']
        width = x2 - x1 # 439

        t_oo = time.time()
        talk_pic_np = self.head # [:, :, ]

        # landmark = self.param_list[idx]
        orig_pic_np = self.body
        full_box_mask = np.zeros((width, width), np.float32) # 
        renzhong_cord = landmark[28]
        mask_start_height = int(renzhong_cord[1])
        box_mask_size = (width - mask_start_height, width)
        box_mask = face_parser.create_static_box_mask(box_mask_size, 0.3, (0, 0, 0, 0))
        full_box_mask[mask_start_height:, :] = box_mask

        talk_frame_mask = mask[0]

        # occlusion_mask = talk_frame_mask.transpose(0, 1, 2).clip(0, 1).astype(np.float32)
        talk_frame_mask = mask #(cv2.GaussianBlur(occlusion_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2


        talk_vision_512 = cv2.resize(talk_pic_np, (width, width))
        talk_frame_mask_512 = cv2.resize(talk_frame_mask, (width, width))
        crop_mask = np.minimum.reduce([full_box_mask, talk_frame_mask_512])
        crop_mask = np.expand_dims(crop_mask, axis=-1)
        # crop_mask = crop_mask[:, ::-1, :]
        orig_face = orig_pic_np[y1: y2, x1: x2]

        merged_face = talk_vision_512 * crop_mask + (1 - crop_mask) * orig_face
        orig_pic_np[y1:y2, x1:x2] = merged_face

        self.result = orig_pic_np

        self.composite()

        self.merger.handle_task_done(self)

class HeadBodyMerger(threading.Thread):
    def __init__(self, crop_info, max_workers = 1) -> None:
        super().__init__(name="MergeHeadBody", daemon=True)
        self._on_merge = None
        self.task_list: list = None
        self.task_list_lock = threading.Lock()
        self.last_completed_task = -1

        self.send_task_index = -1
        self.task_queue = queue.Queue()
        self.wait_complete_event = threading.Event()
        self.wait_batch_complete_event = threading.Event()
        self.crop_info = crop_info

        self.max_workers = max_workers

        self.init_process()

    def init_process(self):
        for _ in range(self.max_workers):
            face_parser = FaceParser()
            threading.Thread(target=self.merge_process, args=(face_parser,), daemon=True).start()

    def start_batch(self, batch_size):
        with self.task_list_lock:
            self.task_list = [None] * batch_size
            # clear all events
            self.wait_batch_complete_event.clear()
            self.wait_complete_event.clear()

    def end_batch(self):
        # 清空任务列表
        while not self.task_queue.empty():
            self.task_queue.get()

        with self.task_list_lock:
            self.task_list = None
            self.last_completed_task = -1
            self.send_task_index = -1
            self.wait_batch_complete_event.set()

    def handle_task_done(self, task):
        with self.task_list_lock:
            if self.task_list is None:
                return
            task_idx = task.idx
            if task_idx == self.last_completed_task + 1:
                self.last_completed_task += 1
                for i in range(task_idx + 1, len(self.task_list)):
                    if self.task_list[i] is not None and self.task_list[i].result is not None:
                        self.last_completed_task = i
                    else:
                        break
                self.wait_complete_event.set()

    def merge_process(self, face_parser):
        while True:
            task = self.task_queue.get()
            task.merge(face_parser)
            # print(threading.current_thread().native_id)

    def merge(self, idx, head, body, landmarks, body_mask=None, background=None):
        task = MergeTask(self, idx, head, body, landmarks, body_mask, background)
        with self.task_list_lock:
            if self.task_list is None:
                raise Exception("No batch started")
            self.task_list[idx] = task
            self.task_queue.put(task)

    def on_merge(self, callback):
        self._on_merge = callback

    def wait_compelte(self):
        self.wait_batch_complete_event.wait()
        self.wait_batch_complete_event.clear()
        # reset all
        with self.task_list_lock:
            self.task_list = None
            self.last_completed_task = -1
            self.send_task_index = -1

    def get_remaining_frames(self):
        with self.task_list_lock:
            return self.last_completed_task - self.send_task_index

    def run(self):
        while True:
            if self.send_task_index == self.last_completed_task:
                self.wait_complete_event.wait()
                self.wait_complete_event.clear()
            
            with self.task_list_lock:    
                while self.send_task_index < self.last_completed_task:
                    st = time.time()
                    # print("Loop start", st)
                    self.send_task_index += 1
                    # print("ON MERGE:", self.send_task_index)
                    self._on_merge(self.task_list[self.send_task_index].result)
                    # print("Loop end:", time.time(), time.time() - st)
                # print(self.send_task_index, len(self.task_list))
                if self.send_task_index == len(self.task_list) - 1:
                    self.wait_batch_complete_event.set()


if __name__ == "__main__":
    pass
