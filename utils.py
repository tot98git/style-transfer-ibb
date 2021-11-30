import numpy as np
import cv2
import math
from indices import LEFT_EYE_INNER, MOUTH, MOUTH_INNER, RIGHT_EYE_INNER


def alpha_blend(layer1, layer2, gamma, ldmrk_src):
    mask_persistent = generate_persistent_mask(layer1, ldmrk_src)
    mask_temp = generate_persistent_mask(layer2, ldmrk_src)
    layer2 = (layer2-mask_temp)+(mask_persistent)

    blend = cv2.addWeighted(
        layer1, 1-gamma, layer2, gamma, 0.0)

    return blend


def get_detail_layer(layer):
    smooth = cv2.bilateralFilter(layer, 50, 100, 100)

    return smooth, layer - smooth


def skin_detail_transfer(layer1, layer2, gamma1, gamma2, ldmrk_src):
    mask_persistent = generate_persistent_mask(layer1, ldmrk_src)
    mask_temp = generate_persistent_mask(layer2, ldmrk_src)
    layer2 = (layer2-mask_temp)+(mask_persistent)

    blend = cv2.addWeighted(
        layer1, gamma1, layer2, gamma2, 0.0)

    return blend


def generate_persistent_mask(layer, landmarks_source):
    landmarks_source = np.int32(landmarks_source)
    left_eyes_coords = landmarks_source[np.array(LEFT_EYE_INNER)]
    right_eyes_coords = landmarks_source[np.array(RIGHT_EYE_INNER)]
    mouth_coords = landmarks_source[np.array(MOUTH_INNER)]

    src_mask = np.zeros(layer.shape[:2], dtype=np.uint8)
    left_eye_mask = cv2.fillPoly(src_mask, [left_eyes_coords], (255, 255, 255))
    left_eye_mask = cv2.bitwise_and(layer, layer, mask=src_mask)

    right_eye_mask = cv2.fillPoly(
        src_mask, [right_eyes_coords], (255, 255, 255))
    right_eye_mask = cv2.bitwise_and(layer, layer, mask=src_mask)

    mouth_mask = cv2.fillPoly(
        src_mask, [mouth_coords], (255, 255, 255))
    mouth_mask = cv2.bitwise_and(layer, layer, mask=src_mask)

    final = cv2.bitwise_or(cv2.bitwise_or(
        left_eye_mask, right_eye_mask), mouth_mask)
    cv2.imwrite('tmp/dst.png', final)
    return final


def transform_with_mask(mask_points):
    def wrapper(transformer, *args): return transformer(*args, mask_points)
    return wrapper
