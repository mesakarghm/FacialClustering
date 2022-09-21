import os

import cv2
import numpy as np

from face_detection_api import FaceDetector

fd = FaceDetector()
scales = [1024, 1980]
target_size = scales[0]
max_size = scales[1]
out_path = "./data/cropped_faces_shk"
if not os.path.exists(out_path):
    os.makedirs(out_path)
for path in os.listdir('./data/face'):
    print(f"Processing for {path}....")
    count = 0
    for img in os.listdir(os.path.join("./data/face/",path)):
        count +=1
        print(f"Processing image {count} of {path}")
        face_path = os.path.join(f"./data/face/{path}", img)
        img = cv2.imread(face_path)
        im_shape = img.shape
        target_size = scales[0]
        max_size = scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        scales_img = [im_scale]
        output = fd.detect_face(face_path)
        bboxes, probs = output['bboxes'], output['probs']
        if len(bboxes):
            for i, box in enumerate(bboxes):
                if probs[i] > 0.9:
                    box = np.asarray(box, dtype = int)
                    height = box[2] - box[0]
                    width = box[3] - box[1]
                    crop_img = img[box[1]:box[1]+width, box[0]:box[0]+height]
                    try:
                        crop_img = cv2.resize(crop_img,(160,160), interpolation = cv2.INTER_AREA)
                        filename = os.path.join(out_path,f"{path}_{count}_{i}"+".jpg")
                        cv2.imwrite(filename, crop_img)
                    except Exception as e:
                        print(e)
                        continue
        else:
            print(f"no bbox for {face_path}")



