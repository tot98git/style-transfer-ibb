
import cv2
import numpy as np
from utils import laplacian_blend, laplacian_pyramind


class Image:
    def __init__(self, img, raw=False):
        self.orig_img = img if raw else cv2.imread(img)
        self.img = self.orig_img
        self.face = None

    def get_img(self):
        return self.img

    def detectFaces(self, img):
        cascadeFace = cv2.CascadeClassifier(
            "utils/haarcascade_frontalface_default.xml")
        detectionList = cascadeFace.detectMultiScale(img, 1.05, 5)

        return detectionList

    def process_face(self, scaling_factor=1):
        faces = self.detectFaces(self.orig_img)

        if len(faces) == 0:
            return False

        dim = self.orig_img.shape[:2]
        x, y, w, h = faces[0]
        x = x - ((scaling_factor-1) * x)
        y = y - ((scaling_factor-1) * y)
        w = w + ((scaling_factor-1) * w)
        h = h + ((scaling_factor-1) * h)

        x = int(x) if x < dim[0] else dim[0]
        y = int(y) if y < dim[1] else dim[1]
        w = int(w)
        h = int(h)

        self.coords = x, y, w, h
        face = self.img[y:y+h, x:x+w]
        self.face = face
        self.loc = (x, y)

        return face

    def resize_img(self, dim, img):
        self.resize_factor = img.shape[0]/dim[0], img.shape[1]/dim[1]
        self.orig_dim = img.shape

        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    def decompose(self, img, name):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype('float32')
        L, A, B = cv2.split(img)

        return L, A, B

    def apply_transformation(self, trns, face_mask):
        img = cv2.resize(trns, self.orig_dim[:2], interpolation=cv2.INTER_AREA)

        face_mask = np.int32((face_mask*self.resize_factor))
        container = np.zeros(self.orig_img.shape, dtype=np.uint8)

        container[self.loc[1]:self.loc[1]+img.shape[1],
                  self.loc[0]: self.loc[0] + img.shape[0]] = img

        face_mask_calc = face_mask + self.loc
        src_mask = np.zeros(container.shape[:2], dtype=np.uint8)
        cv2.fillPoly(src_mask, [face_mask_calc], (255, 255, 255))

        dst = cv2.bitwise_or(self.orig_img, self.orig_img, mask=src_mask)
        base_src_mask = src_mask.copy()
        container_final = cv2.bitwise_or(
            container, container, mask=base_src_mask)
        cv2.imwrite('tmp/container_final.png', container_final)

        final = np.array(self.orig_img)-dst

        return final + container_final


if __name__ == "__main__":
    Image('warped.png').decompose()
