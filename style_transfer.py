import cv2
from delaunay import warp_image
import numpy as np
from FaceAlign import FaceAlign
from Image import Image
from utils import alpha_blend,  get_detail_layer, skin_detail_transfer, transform_with_mask
from indices import FACE
from time import time
import sys
import getopt


class StyleTransfer:
    def __init__(self):
        self.fa = FaceAlign()
        self.WORKING_DIM = (250, 250)

    def process_face(self, face):
        """
            Receives an Image object and returns the collected landmarks, the bounded face region and a face mask
        """
        f = face.process_face(1.1)
        f = face.resize_img(self.WORKING_DIM, f)
        landmarks, face, face_mask = self.fa.detect(f)

        return landmarks, face, face_mask

    def update_face(self, example_src, datatype="String"):
        """
            Esentially used for initializing the example image. For RealtimeTransfer, this is used to update the example image with the latest frame. Also decomposes E to CIELAB layers 
        """
        f1 = Image(example_src, raw=datatype != "String")
        self.f1 = f1

        landmarks_source, face1, face_mask = self.process_face(f1)
        self.landmarks_source = landmarks_source
        self.face1 = face1
        self.face_mask = face_mask
        src_mask = np.zeros(self.WORKING_DIM, dtype=np.uint8)
        cv2.fillPoly(src_mask, [self.face_mask], (255, 255, 255))
        self.src_mask = src_mask
        self.mask_transformer = transform_with_mask(self.landmarks_source)

        f1_L, f1_A, f1_B = self.f1.decompose(self.face1, "face1")
        f1_large_scale, f1_detail = get_detail_layer(f1_L, 10)

        self.f1_A = f1_A
        self.f1_B = f1_B

        self.f1_large_scale = f1_large_scale
        self.f1_detail = f1_detail

    def initialize_reference(self, reference_src):
        """
            Initializes the reference image
        """
        f2 = Image(reference_src)
        self.f2 = f2
        landmarks_target, face2, _ = self.process_face(f2)
        self.landmarks_target = landmarks_target
        self.face2 = face2

    def process_reference_layers(self):
        """
            Decomposes R in CIELAB layers
        """
        f2_L, f2_A, f2_B = self.f2.decompose(self.warped_img, "face2")
        f2_large_scale, f2_detail = get_detail_layer(f2_L, 100)

        self.f2_A = f2_A
        self.f2_B = f2_B
        self.f2_large_scale = f2_large_scale
        self.f2_detail = f2_detail

    def warp_img(self, warping_type="tps"):
        """
        Warps image. Must be called after the layers have been intialised and processed. Affine warping is faster but buggy. TPS slower but works better with higher dim. images
        """
        if warping_type == "affine":
            warped_img = warp_image(self.face2, np.uint8(
                self.landmarks_target), np.uint8(self.landmarks_source))
        else:
            warped_img = self.fa.warpImg(
                self.landmarks_source, self.landmarks_target, self.face2)

        self.warped_img = warped_img

    def get_isolated_faces(self):
        isolated_orig_face = cv2.bitwise_or(
            self.face1, self.face1, mask=self.src_mask)
        isolated_test_face = cv2.bitwise_or(
            self.warped_img, self.warped_img, mask=self.src_mask)
        cv2.imwrite('final/isolated.png', isolated_orig_face)
        cv2.imwrite("final/warped.png", self.warped_img)

        return isolated_orig_face, isolated_test_face

    def transform(self, path="", mode="write", blending_ratio=1, detail_power_1=.4, detail_power_2=.5):
        """
            Core transforming method.
        """
        BLENDING_RATIO = blending_ratio

        DETAIL_POWER_1 = detail_power_1
        DETAIL_POWER_2 = detail_power_2

        finalA = self.mask_transformer(
            alpha_blend, self.f1_A, self.f2_A, BLENDING_RATIO)
        finalB = self.mask_transformer(
            alpha_blend, self.f1_B, self.f2_B, BLENDING_RATIO)
        detail = self.mask_transformer(
            skin_detail_transfer, self.f1_detail, self.f2_detail, DETAIL_POWER_1, DETAIL_POWER_2)

        width, height = self.f1_large_scale.shape
        center = (int(height/2), int(width/2))

        landmarks_target = self.landmarks_source[np.array(FACE)]
        src_mask = np.zeros(self.landmarks_source.shape, dtype=np.uint8)
        large_scale = cv2.seamlessClone(
            self.f2_large_scale, self.f1_large_scale, src_mask, center, cv2.MIXED_CLONE)

        img = cv2.merge([large_scale+detail, finalA, finalB]).astype('uint8')

        img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)

        isolated_transformation = cv2.bitwise_or(
            img, img, mask=self.src_mask)

        transformed = self.f1.apply_transformation(img, self.face_mask)

        if mode == "write":
            cv2.imwrite(f"{path}.png", isolated_transformation)

            cv2.imwrite(f"{path}_complete.png", transformed)

        return transformed

    def main(self, argv):
        test_src = "poses/no_makeup.jpeg"
        test_ref = "poses/makeup_lightskin.jpeg"
        warping = "tps"
        dim = (250, 250)
        alpha, gamma = 1, 0.5
        path = f"tmp/result"

        opts, args = getopt.getopt(
            argv,
            "a:g:s:r:w:d:p",
            [
                "alpha=",
                "gamma=",
                "src=",
                "ref=",
                "warping=",
                "dim=",
                "path="
            ],
        )

        for curr_opt, curr_arg in opts:
            if curr_opt == "-a" or curr_opt == "--alpha":
                alpha = float(curr_arg)

            if curr_opt == "-g" or curr_opt == "--gamma":
                gamma = float(curr_arg)

            if curr_opt == "--ref":
                test_ref = curr_arg

            if curr_opt == "--src":
                test_src = curr_arg

            if curr_opt == "-w" or curr_opt == "--warping":
                warping = curr_arg

            if curr_opt == "-d" or curr_opt == "--dim":
                dim = tuple(curr_arg)
                self.WORKING_DIM = dim

            if curr_opt == "--path":
                path = curr_arg

        print(
            f"Transfering styles from {test_ref} to {test_src} with alpha={alpha} and gamma={gamma}.")
        self.update_face(test_src)
        self.initialize_reference(test_ref)
        self.warp_img(warping)
        self.process_reference_layers()

        self.transform(path, alpha, gamma, 1-gamma)


if __name__ == "__main__":
    s = StyleTransfer()
    s.main(sys.argv[1:])
