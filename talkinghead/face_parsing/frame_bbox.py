import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from face_parsing.face_parser import FaceParser

class FrameBBox:
    def __init__(self):
        self.face_parser = FaceParser()

    def get_ext_file_list(self, path, file_exts=('.jpg', '.jpeg', '.png')):
        selected_list = []
        file_list = os.listdir(path)
        root_path = Path(path)
        for f_name in file_list:
            # ddd = os.path.splitext("coeff")
            # print(ddd[1] in file_exts)
            file_path = root_path / f_name
            if os.path.isdir(file_path):
                continue
            extension = os.path.splitext(f_name)[1]
            if extension in file_exts:  # == '.jpg' or extension == '.jpeg' or extension == '.png':
                selected_list.append(f_name)
        return selected_list

    def get_box(self,frame_path):
        root_path = Path(frame_path)
        file_list = self.get_ext_file_list(frame_path)
        frame_path = file_list[0]
        full_path = root_path/frame_path
        frame_img = cv2.imread(str(full_path))
        face = self.face_parser.get_many_faces(frame_img)[0]
        landmark = face['landmark']['68']
        left_eye_corner = landmark[36]
        right_eye_corner = landmark[45]
        center_point = (int(right_eye_corner[0] + left_eye_corner[0]) / 2), (int(left_eye_corner[1] + right_eye_corner[1]) / 2)
        jaw = landmark[8]
        eye_jaw_distance = jaw[1] - center_point[1]
        bottom_half_value = eye_jaw_distance/0.9
        up_half_value = int(bottom_half_value*0.95)
        rect_width = int(up_half_value + bottom_half_value)
        left_corner = center_point[0]- int(rect_width/2),center_point[1]-up_half_value
        right_bottom = center_point[0]+ int(rect_width/2),center_point[1]+bottom_half_value
        x1, x2, y1, y2 = int(left_corner[0]), int(right_bottom[0]), int(left_corner[1]), int(right_bottom[1])
        width = x2 - x1
        height = y2 - y1
        if width != height:
            diff = abs(width - height)
            if width > height:
                x2 -= diff
            else:
                y2 -= diff
        return x1,y1,x2,y2
        
    def get_big_box(self,frame_path):
        root_path = Path(frame_path)
        file_list = self.get_ext_file_list(frame_path)
        x1s, y1s, x2s, y2s = [], [], [], []
        for frame_path in tqdm(file_list):
            full_path = root_path/frame_path
            frame_img = cv2.imread(str(full_path))
            face = self.face_parser.get_many_faces(frame_img)[0]
            landmark = face['landmark']['68']
            left_eye_corner = landmark[36]
            right_eye_corner = landmark[45]
            center_point = (int(right_eye_corner[0] + left_eye_corner[0]) / 2), (int(left_eye_corner[1] + right_eye_corner[1]) / 2)
            jaw = landmark[8]
            eye_jaw_distance = jaw[1] - center_point[1]
            bottom_half_value = eye_jaw_distance/0.9
            up_half_value = int(bottom_half_value*0.95)
            rect_width = int(up_half_value + bottom_half_value)
            left_corner = center_point[0]- int(rect_width/2),center_point[1]-up_half_value
            right_bottom = center_point[0]+ int(rect_width/2),center_point[1]+bottom_half_value
            x1, x2, y1, y2 = int(left_corner[0]), int(right_bottom[0]), int(left_corner[1]), int(right_bottom[1])
            x1s.append(x1)
            y1s.append(y1)
            x2s.append(x2)
            y2s.append(y2)
        x1 = min(x1s)
        y1 = min(y1s)
        x2 = max(x2s)
        y2 = max(y2s)

        # make square
        width = x2 - x1
        height = y2 - y1
        if width != height:
            diff = abs(width - height)
            if width > height:
                y2 += diff
            else:
                x2 += diff
        print(f"x1:{x1},y1:{y1},x2:{x2},y2:{y2}")
        return x1,y1,x2,y2
                
            

    def crop_head(self,frame_path,out_put_path):
        out_put_path = Path(out_put_path)
        out_put_path.mkdir(exist_ok=True)
        x1,y1,x2,y2 = self.get_big_box(frame_path)
        face_config = {"x1":x1,"y1":y1,"x2":x2,"y2":y2}
        with open(out_put_path/"config.json", 'w') as json_file:
            json.dump(face_config, json_file, indent=4)
        file_list = self.get_ext_file_list(frame_path)
        root_path = Path(frame_path)
        for frame_path in tqdm(file_list):
            full_path = root_path / frame_path
            image = cv2.imread(str(full_path))
            cropped_image = image[y1:y2, x1:x2]
            face = self.face_parser.get_many_faces(cropped_image)[0]
            lmk = face['landmark']['68']
            np.save(str(out_put_path/f"{frame_path}.npy"), lmk, allow_pickle=True)

            head = cv2.resize(cropped_image,(256,256))
            #os.path.basename()

            cv2.imwrite(str(out_put_path/frame_path), head)

if __name__ == "__main__":
    bbx = FrameBBox()
    # bbx.crop_head("../talking_frames","../talking_frames/new_crop")
    bbx.crop_head("../workspace/251249b2321c4e77d290dd78b130271d94409ba191570df2e53763bd4e69c2d883952b96b3206f245cffeb2ba59668ac/frames", "../workspace/251249b2321c4e77d290dd78b130271d94409ba191570df2e53763bd4e69c2d883952b96b3206f245cffeb2ba59668ac/frame_faces")
