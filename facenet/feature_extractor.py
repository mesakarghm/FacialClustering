import logging
import time
from datetime import datetime

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms

# ------------------------------------------------------------------------------------------ #


class FacenetFeatureExtractor:
    def __init__(self, model_url=None, size=None, device=None, logger=None):
        self.logger = logger or logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        try:
            self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.size = size or (160,160)

            self.facenet = InceptionResnetV1(pretrained=model_url or 'vggface2').eval().to(self.device)
            self.face_net_transform = transforms.Compose([transforms.Resize(self.size),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize([0.5, 0.5, 0.5], [0.50196, 0.50196, 0.50196])
                                                          ])
            self.embedding = np.array([])
        except Exception as e:
            self.logger.error("init facenet:", e)
            raise
    # ------------------------------------------------------------------------------------------ #

    def extract_face_embedding(self, face_img, normalize=False):
        embedding = np.array([])
        if isinstance(face_img, np.ndarray):
            face_img = Image.fromarray(face_img)
        face_img_ = self.face_net_transform(face_img)
        with torch.no_grad():
            aligned = torch.stack([face_img_]).to(self.device)
            vec = self.facenet(aligned).detach().cpu()
            embedding = self.normalize(vec).numpy() if normalize else vec.numpy()
        return embedding
    # ------------------------------------------------------------------------------------------ #

    def extract_face_embeddings_batch(self, face_img_list, normalize=False, batch_size=4):
        embedding = np.array([])
        batch_faces = []
        try:
            for i, face in enumerate(face_img_list):
                if isinstance(face, np.ndarray):
                    self.logger.info(f"shape of image {i} is {face.size}")
                    batch_faces.append(self.face_net_transform(Image.fromarray(face)))
                else:
                    batch_faces.append(self.face_net_transform(face))
            with torch.no_grad():
                batches_ = [batch_faces[i:i + batch_size] for i in range(0, len(batch_faces), batch_size)]
                for b in batches_:
                    aligned = torch.stack(b).to(self.device)
                    vec = self.facenet(aligned).detach().cpu()
                    res = self.normalize(vec).numpy() if normalize else vec.numpy()
                    embedding = np.concatenate((embedding, res), axis=0) if embedding.size > 0 else res
                    # self.embedding = self.normalize(vec).numpy() if normalize else vec.numpy()
            self.logger.info(f"facenet embedding {len(self.embedding)}")
            return embedding
        except Exception as e:
            self.logger.error(f"facenet: extract_face_embeddings_batch: {e}", exc_info=True)
            self.logger.error(f"type of face {type(face)} {face.size} in index {i} out of {len(face_img_list)} images.")
            self.logger.error(f"next image {type(face_img_list[i+1])} in index {i+1} previosu image of {type(face_img_list[i-1])} ")
            face.save("error.png", format="png")
            # import pdb
            # pdb.set_trace()
            raise
    # ------------------------------------------------------------------------------------------ #

    def normalize(self, v, axis=1):
        norm = torch.norm(v, 2, axis, True)
        return torch.div(v, norm)


if __name__=="__main__":
    import sys
    model = FacenetFeatureExtractor()
    # img = cv2.imread(sys.argv[1])
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=Image.open(sys.argv[1])
    # img = model.face_net_transform(np.array(img))
    img = model.face_net_transform(img)
    features =model.extract_face_embedding(img)
    print (features)
