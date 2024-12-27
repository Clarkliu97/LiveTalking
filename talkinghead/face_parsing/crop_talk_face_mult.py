import os
import multiprocessing
import time

import cv2
import numpy as np
from tqdm import tqdm

from face_parsing.face_parser import warp_face_by_face_landmark_5, FaceParser

# from face_parser import get_many_faces
# from utils.face_helper import warp_face_by_face_landmark_5
face_parser = FaceParser()

max_queue_size = 10  # 设置最大队列大小
def read_images(image_paths, queue, max_queue_size):
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"none :{image_path}")
        file_name = os.path.basename(image_path)
        while queue.qsize() >= max_queue_size:
            # 如果队列已满，等待一段时间
            time.sleep(0.1)
        queue.put((image,file_name))


def crop_dir_tqdm(input_dir,out_dir):
    os.makedirs(out_dir, exist_ok=True)
    image_list = os.listdir(input_dir)
    image_list = [os.path.join(input_dir, x) for x in image_list if x[-4:] == ".jpg" or x[-4:] == ".png"]
    # queue_payloads = image_list

    queue = multiprocessing.Queue()

    # 创建多个进程读取图片并放入队列中
    num_processes = 4
    processes = []
    chunk_size = len(image_list) // num_processes
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_processes - 1 else len(image_list)
        process = multiprocessing.Process(target=read_images, args=(image_list[start_idx:end_idx], queue, max_queue_size))
        process.start()
        processes.append(process)

    count = 0
    for _ in tqdm(range(len(image_list))):
        image_frame,file_name = queue.get()
        if image_frame is None:
            break
        face = face_parser.get_many_faces(image_frame)[0]

        landmark_data = face['landmark']['5']
        # crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(image_frame, landmark_data, "arcface_128_v2", (128,128))
        # inverse_matrix = cv2.invertAffineTransform(affine_matrix)
        crop_vision_512, affine_matrix_512 = warp_face_by_face_landmark_5(image_frame, landmark_data, "ffhq_512", (512, 512))
        crop_vision_256 = cv2.resize(crop_vision_512, (256, 256))
        face_512 = face_parser.get_many_faces(crop_vision_512)[0]
        landmark_512 = face_512['landmark']['68']

        # write_files(affine_matrix_512, crop_vision_256, file_name, landmark_512, out_dir)

        wprocess = multiprocessing.Process(target=write_files, args=(affine_matrix_512, crop_vision_256, file_name, landmark_512, landmark_data, out_dir))
        wprocess.start()

        count += 1

    # # 在这里对图像进行处理，这里只是简单地显示图像的大小
    # print("Image size:", image.shape)

    # 等待所有读取图片的进程完成
    for process in processes:
        process.join()

    # 将结束标志放入队列，通知处理图像的进程退出
    for _ in range(num_processes):
        queue.put(None)


def write_files(affine_matrix_512, crop_vision_256, file_name, landmark_512, landmark_ori, out_dir):
    cv2.imwrite(os.path.join(out_dir, file_name), crop_vision_256)
    numpy_dict = {"affine_matrix_512": affine_matrix_512, "landmark_512": landmark_512}
    np.save(os.path.join(out_dir, f"{file_name}.npy"), numpy_dict, allow_pickle=True)


if __name__ == "__main__":
    t1 = time.time()
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../talking_frames"))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../talking_frames/face_crop"))
    crop_dir_tqdm(input_dir,out_dir)
    t2 = time.time() - t1
    print(f"{t2}")




