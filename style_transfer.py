import cv2
import numpy as np
from FaceAlign import FaceAlign
from Image import Image
from utils import alpha_blend, get_detail_layer, skin_detail_transfer, transform_with_mask
from indices import FACE


class StyleTransfer:
    def main(self):
        example_src = "poses/no_makeup.jpeg"
        target_src = "poses/makeup_hanna.jpeg"

        f1 = Image(example_src)
        f2 = Image(target_src)

        WORKING_DIM = (500, 500)
        BLENDING_RATIO = .5

        DETAIL_POWER_1 = .7
        DETAIL_POWER_2 = .2
        f1_f = f1.process_face()
        f1_f = f1.resize_img(WORKING_DIM, f1_f)

        f2_f = f2.process_face()
        f2_f = f2.resize_img(WORKING_DIM, f2_f)

        fa = FaceAlign()

        landmarks_source, face1, face_mask = fa.detect(f1_f)
        landmarks_target, face2, _ = fa.detect(f2_f)
        warped_img = fa.warpImg(landmarks_source, landmarks_target, face2)

        f1_L, f1_A, f1_B = f1.decompose(face1, "face1")
        f2_L, f2_A, f2_B = f2.decompose(warped_img, "face2")

        f1_large_scale, f1_detail = get_detail_layer(f1_L)
        f2_large_scale, f2_detail = get_detail_layer(f2_L)

        landmarks_target = np.int32(landmarks_target)
        landmarks_source = np.int32(landmarks_source)

        mask_transformer = transform_with_mask(landmarks_source)

        finalA = mask_transformer(alpha_blend, f1_A, f2_A, BLENDING_RATIO)
        finalB = mask_transformer(alpha_blend, f1_B, f2_B, BLENDING_RATIO)
        detail = mask_transformer(
            skin_detail_transfer, f1_detail, f2_detail, DETAIL_POWER_1, DETAIL_POWER_2)

        width, height = f1_large_scale.shape
        center = (int(height/2), int(width/2))

        landmarks_target = landmarks_source[np.array(FACE)]
        src_mask = np.zeros(landmarks_source.shape, dtype=np.uint8)
        large_scale = cv2.seamlessClone(
            f2_large_scale, f1_large_scale, src_mask, center, cv2.MIXED_CLONE)

        img = cv2.merge([large_scale+detail, finalA, finalB]).astype('uint8')

        img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)

        transformed = f1.apply_transformation(img, face_mask)

        cv2.imwrite('tmp/result.png', transformed)
        cv2.imwrite('tmp/corrected.png', img)
        cv2.imwrite('tmp/face1.png', face1)
        cv2.imwrite('tmp/face2.png', face2)
        cv2.imwrite('tmp/warped.png', warped_img)
        cv2.imwrite('tmp/finalA.png', finalA)
        cv2.imwrite('tmp/finalB.png', finalB)
        cv2.imwrite('tmp/f1_large_scale.png', f1_large_scale)
        cv2.imwrite('tmp/f2_large_scale.png', f2_large_scale)
        cv2.imwrite('tmp/detail.png', f1_detail)
        cv2.imwrite('tmp/detail2.png', f2_detail)
        cv2.imwrite('tmp/final_detail.png', detail)
        cv2.imwrite('tmp/large_scale.png', large_scale)


if __name__ == "__main__":
    StyleTransfer().main()
