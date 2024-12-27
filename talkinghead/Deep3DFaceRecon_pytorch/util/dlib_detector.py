import bz2
import urllib.request
import os
import dlib
import cv2
import numpy as np


class FaceLandmarksDetector:
    def __init__(self, model_path='util/shape_predictor_68_face_landmarks.dat'):
        self.model_path = model_path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = self.load_predictor()

    def load_predictor(self):
        if not os.path.exists(self.model_path):
            self.download_model()
        return dlib.shape_predictor(self.model_path)

    def download_model(self):
        model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        model_bz2_path = self.model_path + '.bz2'

        print("Downloading the model file...")
        urllib.request.urlretrieve(model_url, model_bz2_path)

        print("Extracting the model file...")
        with open(self.model_path, 'wb') as new_file, bz2.BZ2File(model_bz2_path, 'rb') as file:
            for data in iter(lambda: file.read(100 * 1024), b''):
                new_file.write(data)
        print("Extraction complete.")

    def get_landmarks(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if not faces:
            return None  # No faces detected

        face_landmarks = []
        for face in faces:
            landmarks = self.predictor(gray, face)
            landmarks_list = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
            face_landmarks.append(landmarks_list)

        return face_landmarks

def get_landmarks5(pic_path):
    ld_list = get_landmarks68(pic_path)
    result_list = []
    for lm in ld_list:
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        # print(lm[lm_idx,:])
        # # print(lm[lm_idx[0], :])
        # print(lm[lm_idx[[1, 2]], :])
        # print(np.mean(lm[lm_idx[[1, 2]], :], 0))
        # print("----")
        #
        # print(lm[lm_idx[[3, 4]], :])
        # print(np.mean(lm[lm_idx[[3, 4]], :], 0))
        # print("----")

        lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
            lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
        lm5p = lm5p[[1, 2, 0, 3, 4], :]
        # print(lm5p)
        result_list.append(lm5p)
    return result_list[0]
def get_landmarks68(pic_path):
    detector = FaceLandmarksDetector()
    landmarks = detector.get_landmarks(pic_path)
    result_list = []
    if landmarks:
        for i, face_landmarks in enumerate(landmarks):
            result_list.append(np.array(face_landmarks))
            # print(f"Face {i + 1} Landmarks:")
            # points_list = []
            # for j, (x, y) in enumerate(face_landmarks):
            #     # print(f"  Landmark {j + 1}: ({x}, {y})")
            #     points_list.append((x,y))

    else:
        print("No faces detected.")
    return result_list

if __name__ == "__main__":
    # 示例用法
    result_list = get_landmarks5('/home/xijing/hz_work/Deep3DFaceRecon_pytorch/datasets/examples/000002.jpg')
    print(result_list[0])
