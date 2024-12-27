# This is the entry point to the talking head project
import uuid
import os
import hashlib
from PIL import Image
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Deep3DFaceRecon_pytorch'))

from face_parsing.frame_bbox import FrameBBox
from Deep3DFaceRecon_pytorch.coeff_detector import get_data_path, CoeffDetector, KeypointExtractor, InferenceOptions

bbox = FrameBBox()

root_dir = os.path.dirname(__file__)

def main(): 
    # create workspace folder
    workspace_dir = 'workspace'
    os.makedirs(workspace_dir, exist_ok=True)

    print("Current working directory: ", os.getcwd())

    file_dir = 'inputs'
    file_path = os.path.join(file_dir, 'liuzhenqi.mp4')
    file = open(file_path, 'rb')
    file_hash = hashlib.sha384(file.read()).hexdigest()
    file.close()

    # check if the file is video or image
    vid_exts = ['.mp4', '.mov']
    img_exts = ['.png', '.jpg', '.jpeg']
    ext = os.path.splitext(file_path)[1].lower()
    if ext in vid_exts:
        video = True
    elif ext in img_exts:
        video = False
    else:
        raise ValueError('Unsupported file format')

    # create a folder for the project
    project_folder = os.path.join(workspace_dir, file_hash)
    if os.path.exists(project_folder):
        # remove the folder if it already exists
        os.system(f'rm -r {project_folder}')
    os.makedirs(project_folder)

    # copy the file to the folder
    os.system(f'cp {file_path} {project_folder}')

    file_project_path = os.path.join(project_folder, os.path.basename(file_path))

    if video: 
        # write a video flag file to the project folder
        video_flag_file = os.path.join(project_folder, 'video')
        os.system(f'touch {video_flag_file}')

    if video: 
        if ext == '.mp4':
            # mp4 to frames
            frames_folder = os.path.join(project_folder, 'frames')
            os.makedirs(frames_folder)
            os.system(f'ffmpeg -i {file_project_path} -vf fps=25 {frames_folder}/frame_%05d.png')
        elif ext == '.mov':
            # mov to frames
            frames_folder = os.path.join(project_folder, 'frames')
            os.makedirs(frames_folder)
            os.system(f'ffmpeg -i {file_project_path} -vf fps=25 -pix_fmt rgba {frames_folder}/frame_%05d.png')

        # crop the faces
        frame_faces_folder = os.path.join(project_folder, 'frame_faces')
        os.makedirs(frame_faces_folder)
        bbox.crop_head(frames_folder, frame_faces_folder)

        # get face 3d motion coefficients
        opt = InferenceOptions().parse()  #
        opt.bfm_folder = os.path.join(root_dir, 'Deep3DFaceRecon_pytorch', 'BFM')
        opt.checkpoints_dir = os.path.join(root_dir, 'Deep3DFaceRecon_pytorch', 'checkpoints')
        coeff_detector = CoeffDetector(opt)
        kp_extractor = KeypointExtractor()

        face_coeff_folder = os.path.join(project_folder, 'face_coeff')
        face_keypoint_folder = os.path.join(project_folder, 'face_keypoint')
        os.makedirs(face_coeff_folder)
        os.makedirs(face_keypoint_folder)
        pose_path = os.path.join(project_folder, 'full_pose.npy')

        image_names, keypoint_names = get_data_path(frame_faces_folder, face_keypoint_folder)
        coeff_3dmm_list = []
        for image_name, keypoint_name in zip(image_names, keypoint_names):
            image = Image.open(image_name)
            if not os.path.isfile(keypoint_name):
                lm = kp_extractor.extract_keypoint(image, keypoint_name)
                if lm is None:
                    os.remove(image_name)
                    print(f"removed {image_name}")
                    continue
            else:
                lm = np.loadtxt(keypoint_name).astype(np.float32)
                lm = lm.reshape([-1, 2]) 
            predicted = coeff_detector(image, lm)
            name = os.path.splitext(os.path.basename(image_name))[0]
            np.savetxt(
                "{}/{}_3dmm_coeff.txt".format(face_coeff_folder, name), 
                predicted['coeff_3dmm'].reshape(-1))
            coeff_3dmm_list.append(predicted['coeff_3dmm'][0])

        full_coeff_np = np.array(coeff_3dmm_list)
        np.save(pose_path,full_coeff_np)

        # Get Mask Video (Optional)
        file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
        mask_file_path = os.path.join(file_dir, file_name_no_ext + '_mask' + ext)
        if os.path.exists(mask_file_path):
            if ext == '.mp4':
                # copy the mask file to the project folder
                os.system(f'cp {mask_file_path} {project_folder}')
                mask_frame_folder = os.path.join(project_folder, 'frame_masks')
                os.makedirs(mask_frame_folder)
                os.system(f'ffmpeg -i {mask_file_path} -vf fps=25 {mask_frame_folder}/frame_%05d.png')
            elif ext == '.mov':
                # copy the mask file to the project folder
                os.system(f'cp {mask_file_path} {project_folder}')
                mask_frame_folder = os.path.join(project_folder, 'frame_masks')
                os.makedirs(mask_frame_folder)
                os.system(f'ffmpeg -i {mask_file_path} -vf fps=25 -pix_fmt rgba {mask_frame_folder}/frame_%05d.png')
            
            # copy backgrounds
            bg_folder = "inputs/background"
            os.system(f'cp -r {bg_folder} {project_folder}')
        else: 
            print("Mask file not found " + mask_file_path)

    else: 
        face_folder = os.path.join(project_folder, 'face')
        os.makedirs(face_folder)
        bbox.crop_head(project_folder, face_folder)


if __name__ == '__main__': 
    main()