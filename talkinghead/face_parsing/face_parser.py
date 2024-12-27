
import cv2
import numpy
import numpy as np
import onnxruntime
import os
import threading

FACE_ANALYSER = None
def resolve_relative_path(relatvie_path):
	# script_dir = os.path.dirname(os.path.abspath(__file__))
	return os.path.abspath(os.path.join(os.path.dirname(__file__), relatvie_path))

def apply_execution_provider_options(execution_providers,device_id=0):
	execution_providers_with_options = []

	for execution_provider in execution_providers:
		if execution_provider == 'CUDAExecutionProvider':
			execution_providers_with_options.append((execution_provider,
			{
				'cudnn_conv_algo_search': 'DEFAULT',
				'device_id': device_id
			}))
		else:
			execution_providers_with_options.append(execution_provider)
	return execution_providers_with_options

def unpack_resolution(resolution : str):
	width, height = map(int, resolution.split('x'))
	return width, height

def resize_frame_resolution(vision_frame, max_width : int, max_height : int):
	height, width = vision_frame.shape[:2]

	if height > max_height or width > max_width:
		scale = min(max_height / height, max_width / width)
		new_width = int(width * scale)
		new_height = int(height * scale)
		return cv2.resize(vision_frame, (new_width, new_height))
	return vision_frame

def convert_face_landmark_68_to_5(landmark_68):
	left_eye = numpy.mean(landmark_68[36:42], axis = 0)
	right_eye = numpy.mean(landmark_68[42:48], axis = 0)
	nose = landmark_68[30]
	left_mouth_end = landmark_68[48]
	right_mouth_end = landmark_68[54]
	face_landmark_5 = numpy.array([ left_eye, right_eye, nose, left_mouth_end, right_mouth_end ])
	return face_landmark_5


def apply_nms(bounding_box_list, iou_threshold : float):
	keep_indices = []
	dimension_list = numpy.reshape(bounding_box_list, (-1, 4))
	x1 = dimension_list[:, 0]
	y1 = dimension_list[:, 1]
	x2 = dimension_list[:, 2]
	y2 = dimension_list[:, 3]
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	indices = numpy.arange(len(bounding_box_list))
	while indices.size > 0:
		index = indices[0]
		remain_indices = indices[1:]
		keep_indices.append(index)
		xx1 = numpy.maximum(x1[index], x1[remain_indices])
		yy1 = numpy.maximum(y1[index], y1[remain_indices])
		xx2 = numpy.minimum(x2[index], x2[remain_indices])
		yy2 = numpy.minimum(y2[index], y2[remain_indices])
		width = numpy.maximum(0, xx2 - xx1 + 1)
		height = numpy.maximum(0, yy2 - yy1 + 1)
		iou = width * height / (areas[index] + areas[remain_indices] - width * height)
		indices = indices[numpy.where(iou <= iou_threshold)[0] + 1]
	return keep_indices

TEMPLATES =\
{
	'arcface_112_v1': numpy.array(
	[
		[ 0.35473214, 0.45658929 ],
		[ 0.64526786, 0.45658929 ],
		[ 0.50000000, 0.61154464 ],
		[ 0.37913393, 0.77687500 ],
		[ 0.62086607, 0.77687500 ]
	]),
	'arcface_112_v2': numpy.array(
	[
		[ 0.34191607, 0.46157411 ],
		[ 0.65653393, 0.45983393 ],
		[ 0.50022500, 0.64050536 ],
		[ 0.37097589, 0.82469196 ],
		[ 0.63151696, 0.82325089 ]
	]),
	'arcface_128_v2': numpy.array(
	[
		[ 0.36167656, 0.40387734 ],
		[ 0.63696719, 0.40235469 ],
		[ 0.50019687, 0.56044219 ],
		[ 0.38710391, 0.72160547 ],
		[ 0.61507734, 0.72034453 ]
	]),
	'ffhq_512': numpy.array(
	[
		[ 0.37691676, 0.46864664 ],
		[ 0.62285697, 0.46912813 ],
		[ 0.50123859, 0.61331904 ],
		[ 0.39308822, 0.72541100 ],
		[ 0.61150205, 0.72490465 ]
	])
}

normalized_landmark = \
numpy.array(
	[
		[ 0.34191607, 0.46157411 ],
		[ 0.65653393, 0.45983393 ],
		[ 0.50022500, 0.64050536 ],
		[ 0.37097589, 0.82469196 ],
		[ 0.63151696, 0.82325089 ]
	])

def norm_face_by_face_landmark_5(temp_vision_frame, face_landmark_5):
	crop_size = (112, 112)
	normed_template = normalized_landmark * crop_size
	affine_matrix = cv2.estimateAffinePartial2D(face_landmark_5, normed_template, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
	crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size, borderMode = cv2.BORDER_REPLICATE, flags = cv2.INTER_AREA)
	return crop_vision_frame, affine_matrix

def warp_face_by_face_landmark_5(temp_vision_frame, face_landmark_5, template, crop_size):
	normed_template = TEMPLATES.get(template) * crop_size
	affine_matrix = cv2.estimateAffinePartial2D(face_landmark_5, normed_template, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
	crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size, borderMode = cv2.BORDER_REPLICATE, flags = cv2.INTER_AREA)
	return crop_vision_frame, affine_matrix

def warp_face_by_translation(temp_vision_frame, translation, scale, crop_size):
	affine_matrix = numpy.array([[ scale, 0, translation[0] ], [ 0, scale, translation[1] ]])
	crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size)
	return crop_vision_frame, affine_matrix


face_detector_yoloface_path = resolve_relative_path('cp/yoloface_8n.onnx')
face_recognizer_arcface_inswapper_path = resolve_relative_path('cp/arcface_w600k_r50.onnx')
face_landmarker_path = resolve_relative_path('cp/2dfan4.onnx')
face_occluder_path = resolve_relative_path("cp/face_occluder.onnx")

execution_providers = ['CUDAExecutionProvider']

face_detector_score = 0.5
face_detector_size = '640x640'

face_model_lock = threading.Lock()
class FaceParser:
	def __init__(self,device_id=0):
		self.face_detector = onnxruntime.InferenceSession(face_detector_yoloface_path, providers = apply_execution_provider_options(execution_providers,device_id=device_id))
		self.face_recognizer = onnxruntime.InferenceSession(face_recognizer_arcface_inswapper_path, providers = apply_execution_provider_options(execution_providers,device_id=device_id))
		self.face_landmarker = onnxruntime.InferenceSession(face_landmarker_path, providers = apply_execution_provider_options(execution_providers,device_id=device_id))
		self.face_occluder = onnxruntime.InferenceSession(face_occluder_path, providers = apply_execution_provider_options(execution_providers,device_id=device_id))
	def detect_with_yoloface(self,vision_frame, face_detector_size : str):
		face_detector = self.face_detector
		face_detector_width, face_detector_height = unpack_resolution(face_detector_size)
		temp_vision_frame = resize_frame_resolution(vision_frame, face_detector_width, face_detector_height)
		ratio_height = vision_frame.shape[0] / temp_vision_frame.shape[0]
		ratio_width = vision_frame.shape[1] / temp_vision_frame.shape[1]
		bounding_box_list = []
		face_landmark5_list = []
		score_list = []

		#with THREAD_SEMAPHORE:
		detections = face_detector.run(None,
		{
			face_detector.get_inputs()[0].name: self.prepare_detect_frame(temp_vision_frame, face_detector_size)
		})
		detections = numpy.squeeze(detections).T
		bounding_box_raw, score_raw, face_landmark_5_raw = numpy.split(detections, [ 4, 5 ], axis = 1)
		keep_indices = numpy.where(score_raw > face_detector_score)[0]
		if keep_indices.any():
			bounding_box_raw, face_landmark_5_raw, score_raw = bounding_box_raw[keep_indices], face_landmark_5_raw[keep_indices], score_raw[keep_indices]
			for bounding_box in bounding_box_raw:
				bounding_box_list.append(numpy.array(
				[
					(bounding_box[0] - bounding_box[2] / 2) * ratio_width,
					(bounding_box[1] - bounding_box[3] / 2) * ratio_height,
					(bounding_box[0] + bounding_box[2] / 2) * ratio_width,
					(bounding_box[1] + bounding_box[3] / 2) * ratio_height
				]))
			face_landmark_5_raw[:, 0::3] = (face_landmark_5_raw[:, 0::3]) * ratio_width
			face_landmark_5_raw[:, 1::3] = (face_landmark_5_raw[:, 1::3]) * ratio_height
			for face_landmark_5 in face_landmark_5_raw:
				face_landmark5_list.append(numpy.array(face_landmark_5.reshape(-1, 3)[:, :2]))
			score_list = score_raw.ravel().tolist()
		return bounding_box_list, face_landmark5_list, score_list


	def prepare_detect_frame(self,temp_vision_frame, face_detector_size : str):
		face_detector_width, face_detector_height = unpack_resolution(face_detector_size)
		detect_vision_frame = numpy.zeros((face_detector_height, face_detector_width, 3))
		detect_vision_frame[:temp_vision_frame.shape[0], :temp_vision_frame.shape[1], :] = temp_vision_frame
		detect_vision_frame = (detect_vision_frame - 127.5) / 128.0
		detect_vision_frame = numpy.expand_dims(detect_vision_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
		return detect_vision_frame


	def create_faces(self,vision_frame , bounding_box_list , face_landmark5_list , score_list ):
		faces = []
		if face_detector_score > 0:
			sort_indices = numpy.argsort(-numpy.array(score_list))
			bounding_box_list = [ bounding_box_list[index] for index in sort_indices ]
			face_landmark5_list = [ face_landmark5_list[index] for index in sort_indices ]
			score_list = [ score_list[index] for index in sort_indices ]
			keep_indices = apply_nms(bounding_box_list, 0.4)
			for index in keep_indices:
				bounding_box = bounding_box_list[index]
				face_landmark_68 = self.detect_face_landmark_68(vision_frame, bounding_box)
				landmark  =\
				{
					'5': convert_face_landmark_68_to_5(face_landmark_68),
					'68': face_landmark_68
				}
				# score = score_list[index]
				embedding, normed_embedding = self.calc_embedding(vision_frame, landmark['5'])
				# gender, age = 0,0 # detect_gender_age(vision_frame, bounding_box)
				face_dict = {"landmark":landmark,"bbox":bounding_box,"embedding":embedding,"normed_embedding":normed_embedding}
				faces.append(face_dict)
		return faces


	def calc_embedding(self,temp_vision_frame, face_landmark_5):
		face_recognizer = self.face_recognizer
		crop_vision_frame, matrix = norm_face_by_face_landmark_5(temp_vision_frame, face_landmark_5)
		crop_vision_frame = crop_vision_frame / 127.5 - 1
		crop_vision_frame = crop_vision_frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)
		crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis = 0)
		embedding = face_recognizer.run(None,
		{
			face_recognizer.get_inputs()[0].name: crop_vision_frame
		})[0]
		embedding = embedding.ravel()
		normed_embedding = embedding / numpy.linalg.norm(embedding)
		return embedding, normed_embedding


	def detect_face_landmark_68(self,temp_vision_frame, bounding_box):
		face_landmarker = self.face_landmarker
		scale = 195 / numpy.subtract(bounding_box[2:], bounding_box[:2]).max() # w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
		translation = (256 - numpy.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
		crop_vision_frame, affine_matrix = warp_face_by_translation(temp_vision_frame, translation, scale, (256, 256))
		crop_vision_frame = crop_vision_frame.transpose(2, 0, 1).astype(numpy.float32) / 255.0
		face_landmark_68 = face_landmarker.run(None,
		{
			face_landmarker.get_inputs()[0].name: [ crop_vision_frame ]
		})[0]
		face_landmark_68 = face_landmark_68[:, :, :2][0] / 64
		face_landmark_68 = face_landmark_68.reshape(1, -1, 2) * 256
		face_landmark_68 = cv2.transform(face_landmark_68, cv2.invertAffineTransform(affine_matrix))
		face_landmark_68 = face_landmark_68.reshape(-1, 2)
		return face_landmark_68



	def get_first_face(self,vision_frame, position : int = 0):
		many_faces = self.get_many_faces(vision_frame)
		if many_faces:
			try:
				return many_faces[position]
			except IndexError:
				return many_faces[-1]
		return None

	def get_one_face(self,vision_frame):
		return self.get_first_face(vision_frame)

	def get_many_faces(self,vision_frame):
		with face_model_lock:
			faces = []
			try:

				bounding_box_list, face_landmark5_list, score_list = self.detect_with_yoloface(vision_frame, face_detector_size)
				faces = self.create_faces(vision_frame, bounding_box_list, face_landmark5_list, score_list)

			except (AttributeError, ValueError):
				pass
			return faces

	def face_loc_matrix(self,landmark_data, temp_vision_frame):
		model_template = "arcface_128_v2"  # get_options('model').get('template')
		model_size = (128, 128)  # get_options('model').get('size')
		# print(target_face)
		# print(type(target_face))

		warp_result = warp_face_by_face_landmark_5(temp_vision_frame, landmark_data, model_template, model_size)
		crop_vision_frame, affine_matrix = warp_result
		inverse_matrix = cv2.invertAffineTransform(affine_matrix)
		crop_vision_512, affine_matrix_enhanced = warp_face_by_face_landmark_5(temp_vision_frame, landmark_data, "ffhq_512", (512, 512))
		affine_matrix_enhanced_trans = np.eye(3)
		affine_matrix_enhanced_trans[:2, :] = affine_matrix_enhanced
		inverse_matrix_trans = np.eye(3)
		inverse_matrix_trans[:2, :] = inverse_matrix
		merged_matrix = np.dot(affine_matrix_enhanced_trans, inverse_matrix_trans)
		merged_matrix = merged_matrix[:2, :]
		return affine_matrix_enhanced, crop_vision_512, crop_vision_frame, merged_matrix

	def get_label_face_crop_data(self,temp_vision_frame, landmark):
		affine_matrix_enhanced, crop_vision_512, crop_vision_frame1, merged_matrix = self.face_loc_matrix(landmark, temp_vision_frame)
		crop_vision_frame = self.prepare_crop_frame(crop_vision_frame1)
		return crop_vision_frame, affine_matrix_enhanced, merged_matrix, crop_vision_512

	@staticmethod
	def prepare_crop_frame(crop_vision_frame):
		model_mean = [0., 0., 0.]  # get_options('model').get('mean')
		model_standard_deviation = [1., 1., 1.]  # get_options('model').get('standard_deviation')
		crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
		crop_vision_frame = (crop_vision_frame - model_mean) / model_standard_deviation
		crop_vision_frame = crop_vision_frame.transpose(2, 0, 1)
		crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis=0).astype(numpy.float32)
		return crop_vision_frame

	def get_label_face_mask(self,crop_vision_frame, frame_numer):
		crop_mask_list = []
		if True:  # 'occlusion' in utils.globals.face_mask_types:
			occlusion_mask = self.create_occlusion_mask(crop_vision_frame)
			crop_mask_list.append(occlusion_mask)
			face_mask_blur = 0.3
			face_mask_padding = (0, 0, 0, 0)
			full_mask_size = crop_vision_frame.shape[:2][::-1]
			mask_start_height = 0
			if True:
				loc_face = self.get_one_face(crop_vision_frame)
				if loc_face:
					if False and frame_numer >= 291 and frame_numer <= 355:
						eye_bottom_left = loc_face['landmark']['68'][41]
						eye_bottom_right = loc_face['landmark']['68'][46]
						mask_start_height = int((eye_bottom_left[1] + eye_bottom_right[1]) / 2) + 20
					else:
						if False:
							eye_brow_left = loc_face['landmark']['68'][19]
							eye_brow_right = loc_face['landmark']['68'][24]
							mask_start_height = int((eye_brow_left[1] + eye_brow_right[1]) / 2) - 60
						else:
							eye_brow_list = loc_face['landmark']['68'][17:27]
							eye_brow_list = [x[1] for x in eye_brow_list]
							mask_start_height = int(min(eye_brow_list)) - 30

					# eye_brow_left = loc_face['landmark']['68'][19]
					# eye_brow_right = loc_face['landmark']['68'][24]
					# mask_start_height1 = int((eye_brow_left[1] + eye_brow_right[1]) / 2) - 30
					# print(f"mask_start_height:{mask_start_height},mask_start_height1:{mask_start_height1}")
			full_box_mask = np.zeros(full_mask_size, numpy.float32)
			if mask_start_height < 0:
				mask_start_height = 0
			box_mask_size = (full_mask_size[0] - mask_start_height, full_mask_size[1])
			box_mask = self.create_static_box_mask(box_mask_size, face_mask_blur, face_mask_padding)
			full_box_mask[mask_start_height:, :] = box_mask
			crop_mask_list.append(full_box_mask)

		# cv2.imshow('occlusion_mask', (occlusion_mask*255).astype(np.int8))
		# cv2.imshow('full_box_mask', full_box_mask)
		# cv2.imshow('crop_vision_frame', crop_vision_frame)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		crop_mask = numpy.minimum.reduce(crop_mask_list).clip(0, 1)
		return crop_mask

	@staticmethod
	def normalize_crop_frame(crop_vision_frame):
		crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
		crop_vision_frame = (crop_vision_frame * 255.0).round()
		crop_vision_frame = crop_vision_frame[:, :, ::-1]
		return crop_vision_frame

	@staticmethod
	def create_static_box_mask(crop_size, face_mask_blur, face_mask_padding):
		blur_amount = int(crop_size[0] * 0.5 * face_mask_blur)
		blur_area = max(blur_amount // 2, 1)
		box_mask = numpy.ones(crop_size, numpy.float32)
		box_mask[:max(blur_area, int(crop_size[1] * face_mask_padding[0] / 100)), :] = 0
		box_mask[-max(blur_area, int(crop_size[1] * face_mask_padding[2] / 100)):, :] = 0
		box_mask[:, :max(blur_area, int(crop_size[0] * face_mask_padding[3] / 100))] = 0
		box_mask[:, -max(blur_area, int(crop_size[0] * face_mask_padding[1] / 100)):] = 0
		if blur_amount > 0:
			box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)

		# cv2.imshow('box_mask', box_mask)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		return box_mask

	def create_occlusion_mask_batch(self,crop_vision_frames):
		prepare_vision_frame = crop_vision_frames.astype(numpy.float32)/255
		occlusion_masks = self.face_occluder.run(None,
		                                        {
			                                        self.face_occluder.get_inputs()[0].name: prepare_vision_frame
		                                        })[0]
		result_mask = occlusion_masks.clip(0, 1)


		# occlusion_mask = cv2.resize(occlusion_mask, crop_vision_frame.shape[:2][::-1])
		if False:
			occlusion_mask = result_mask[0]
			occlusion_mask = occlusion_mask.transpose(0, 1, 2).clip(0, 1).astype(numpy.float32)
			occlusion_mask = (cv2.GaussianBlur(occlusion_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
			cv2.imwrite("mask.png",(occlusion_mask*255).astype(np.uint8))
		return result_mask

	def create_occlusion_mask(self,crop_vision_frame):
		#face_occluder = get_face_occluder()
		input_shape = self.face_occluder.get_inputs()[0].shape[1:3]
		prepare_vision_frame = cv2.resize(crop_vision_frame, tuple(input_shape))
		prepare_vision_frame = numpy.expand_dims(prepare_vision_frame, axis=0).astype(numpy.float32) / 255
		prepare_vision_frame = prepare_vision_frame.transpose(0, 1, 2, 3)
		occlusion_mask = self.face_occluder.run(None,
		                                         {
			                                         self.face_occluder.get_inputs()[0].name: prepare_vision_frame
		                                         })[0][0]
		occlusion_mask = occlusion_mask.transpose(0, 1, 2).clip(0, 1).astype(numpy.float32)
		occlusion_mask = cv2.resize(occlusion_mask, crop_vision_frame.shape[:2][::-1])
		occlusion_mask = (cv2.GaussianBlur(occlusion_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2

		# cv2.imshow('occlusion_mask', occlusion_mask)
		# cv2.imshow('crop_vision_frame', crop_vision_frame)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		return occlusion_mask