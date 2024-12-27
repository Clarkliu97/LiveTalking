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
        self.body_pic_img_list = [cv2.imread(os.path.join(self.body_dir, pic)) for pic in self.body_pic_list]

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

        self.mp4_writing_event = threading.Event()
        self.mp4_writing_event.clear()

        self.out_dir= os.path.join(workspace_dir, "merged")
        os.makedirs(self.out_dir, exist_ok=True)

        self.merged_list = []

    def _to_mp4(self, length, width, height, output_dir, event):
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        video = cv2.VideoWriter(output_dir, fourcc, 25, (width, height))
        event.wait()
        counter = 0
        print("mp4 merge start")
        while True:
            if counter == length:
                break
            if self.merged_list[counter] is not None:
                t0 = time.time()
                frame = self.merged_list[counter]
                video.write(frame)
                counter += 1
                # print(f"frame {counter} written to mp4 in {time.time()-t0} seconds")
            else:
                time.sleep(0)
        t0 = time.time()
        video.release()

        # mp4v to h264
        # os.system(f"ffmpeg -loglevel error -y -i {output_dir} -c:v libx264 -c:a aac -strict experimental {output_dir.replace('.mp4', '_h264.mp4')}")

        print(f"video saved in {time.time()-t0} seconds")
        print("Video saved")

    def _to_stream(self, length, width, height, event): 
        # write to gstreamer
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        pipeline = f"appsrc ! videoconvert ! x264enc ! mp4mux streamable=true fragment-duration=1000 ! tcpclientsink host=0.0.0.0 port=50000"
        video = cv2.VideoWriter(pipeline, fourcc, 25, (width, height))
        event.wait()
        counter = 0
        print("streaming start")
        while True:
            if counter == length:
                break
            if self.merged_list[counter] is not None:
                t0 = time.time()
                frame = self.merged_list[counter]
                video.write(frame)
                counter += 1
                # print(f"frame {counter} written to stream in {time.time()-t0} seconds")
            else:
                time.sleep(0)



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
    
    def merge_single_image_process(self,mask,landmark,img,orig_jpg_pic_path, output_dir,file_name, idx, index_list):
        x1 = self.crop_info['x1']
        y1 = self.crop_info['y1']
        x2 = self.crop_info['x2']
        y2 = self.crop_info['y2']
        width = x2 - x1 # 439

        t_oo = time.time()
        talk_pic_np = img[:, :, ::-1]
        # landmark = self.param_list[idx]
        # follow index list
        body_pic_img_list = [self.body_pic_img_list[i] for i in index_list]
        orig_pic_np = body_pic_img_list[idx]
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
        # print(f"frame {idx} merged in {t_end-t_oo} seconds")

    def merge_numpy(self, img_numpy, index_list, output_dir):
        self.stop_mask_thread()
        length = len(img_numpy)
        height, width, _ = cv2.imread(os.path.join(self.body_dir, self.body_pic_list[0])).shape
        self.merged_list = [None] * length
        

        if type(self.mask_list) == list:
            self.mask_list = np.concatenate(self.mask_list,axis=0)

        merge_thread = threading.Thread(target=self._to_mp4, args=(length, width, height, output_dir, self.mp4_writing_event))
        # merge_thread = threading.Thread(target=self._to_stream, args=(length, width, height, self.mp4_writing_event))
        merge_thread.daemon = True  # Ensures the thread will exit when the main thread exits
        merge_thread.start()

        # follow index list
        param_list = [self.param_list[i] for i in index_list]
        body_pic_list = [self.body_pic_img_list[i] for i in index_list]

        with ThreadPoolExecutor(max_workers=24) as executor:  # Adjust the number of workers as needed
            futures = [executor.submit(self.merge_single_image_process, self.mask_list[idx], param_list[idx], img_numpy[idx], body_pic_list[index_list[idx]], output_dir,self.frame_name_patten.format(idx), idx, index_list) for idx in range(len(img_numpy))]

        # Optional: Wait for all futures to complete if you need to ensure all work is done
        for future in futures:
            future.result()
        self.mp4_writing_event.set()
        merge_thread.join()
        print("All threads joined")


if __name__ == "__main__":
    pass
