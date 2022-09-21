import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image


class FaceDetector:
    def __init__(self):
        self.obj_threshold, self.factor = [0.7, 0.8, 0.8], 0.709
        self.gpu_id = 0
        self.min_size = 60
        self.device = torch.device(
            "cuda:" + str(self.gpu_id)
            if self.gpu_id >= 0 and torch.cuda.is_available()
            else "cpu"
        )
        self.detector = MTCNN(
            margin=0,
            min_face_size=self.min_size,
            thresholds=self.obj_threshold,
            factor=self.factor,
            keep_all=True,
            device=self.device,
        )

    def detect_face(self, image_path):
        try:
            pil_image = Image.open(image_path)
            h, w = pil_image.size
            if h < 160 or w < 160:
                raise Exception("Image tooooo small!@@@")
            with torch.no_grad():
                bboxes, probs = self.detector.detect(pil_image)
            if isinstance(bboxes, np.ndarray) and bboxes.size > 0:
                min_size = self.min_size
                keep = np.where((bboxes[:, 2] - bboxes[:, 0] > min_size) & (bboxes[:, 3] - bboxes[:, 1] > min_size))[0]
                bboxes = bboxes[keep]
                probs = probs[keep]
                output = {"bboxes": bboxes.tolist(), "probs": probs.tolist()}
            else:
                output = {"bboxes": [], "probs": []}
        except Exception as e:
            print(e)
            output = {"bboxes": [], "probs": []}
        return output

