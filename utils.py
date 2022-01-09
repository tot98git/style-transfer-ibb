import numpy as np
import cv2
import math
from indices import LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER, MOUTH, MOUTH_INNER, RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER


def alpha_blend(layer1, layer2, gamma, ldmrk_src):
    persistent_face, temp_eyes, pers_lips = generate_persistent_mask(
        layer1, ldmrk_src)
    temp_face, persistent_eyes, temp_lips = generate_persistent_mask(
        layer2, ldmrk_src)
    layer1 = (layer1 - temp_eyes)+persistent_eyes
    layer2 = (layer2-temp_face)+(persistent_face)-temp_lips+pers_lips

    blend = cv2.addWeighted(
        layer1, 1-gamma, layer2, gamma, 0.0)

    return blend


def laplacian_blend(layer1, layer2, gamma):
    A = (1-gamma) * layer1
    B = gamma*layer2

    blocks = 5
    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in range(blocks):
        G = cv2.pyrDown(G)
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in range(blocks):
        G = cv2.pyrDown(G)
        gpB.append(G)
    # generate Laplacian Pyramid for A
    lpA = [gpA[blocks-1]]
    for i in range(blocks-1, 0, -1):
        size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
        GE = cv2.pyrUp(gpA[i], dstsize=size)
        L = cv2.subtract(gpA[i-1], GE)
        lpA.append(L)

    # generate Laplacian Pyramid for B
    lpB = [gpB[blocks-1]]
    for i in range(blocks-1, 0, -1):
        size = (gpB[i-1].shape[1], gpB[i-1].shape[0])
        GE = cv2.pyrUp(gpB[i], dstsize=size)
        L = cv2.subtract(gpB[i-1], GE)
        lpB.append(L)

    LS = []
    for la, lb in zip(lpA, lpB):

        blend = la + lb

        LS.append(blend)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, blocks):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize=size)
        ls_ = cv2.add(ls_, LS[i])

    return ls_


def test_blend(layer1, layer2, gamma, ldmrk_src):
    mean, std = layer1.mean(), layer1.std()
    mean1, std2 = layer2.mean(), layer2.std()
    mask_persistent = generate_persistent_mask(layer1, ldmrk_src)
    mask_temp = generate_persistent_mask(layer2, ldmrk_src)

    # layer2 = (std/std2) * layer2
    layer2 -= mean
    layer2 *= std2
    blend = cv2.addWeighted(
        layer1, 1-gamma, layer2, gamma, 0.0)

    return blend


def get_detail_layer(layer, size):
    smooth = cv2.bilateralFilter(layer, size, 100, 100)

    return smooth, layer - smooth


def skin_detail_transfer(layer1, layer2, gamma1, gamma2, ldmrk_src):
    persistent_face, temp_eyes, pers_lips = generate_persistent_mask(
        layer1, ldmrk_src)
    temp_face, persistent_eyes, temp_lips = generate_persistent_mask(
        layer2, ldmrk_src)
    layer1 = (layer1-temp_eyes) + persistent_eyes
    layer2 = (layer2-temp_face) + (persistent_face)-persistent_eyes

    blend = cv2.addWeighted(
        layer1, gamma1, layer2, gamma2, 0.0)

    # blend[blend > 255] = 255
    # blend[blend < 0] = 0

    return blend


def generate_persistent_mask(layer, landmarks_source):
    src_mask = np.zeros(layer.shape[:2], dtype=np.uint8)
    mask_builder = prepare_mask(layer, src_mask)

    landmarks_source = np.int32(landmarks_source)
    left_eyes_coords = landmarks_source[np.array(LEFT_EYE_INNER)]
    left_eyeshadow_coords = landmarks_source[np.array(LEFT_EYE_OUTER)]
    right_eyes_coords = landmarks_source[np.array(RIGHT_EYE_INNER)]
    right_eyeshadow_coords = landmarks_source[np.array(
        RIGHT_EYE_OUTER)]
    mouth_coords = landmarks_source[np.array(MOUTH_INNER)]
    full_mouth_coords = landmarks_source[np.array(MOUTH)]

    left_eye_mask = mask_builder(left_eyes_coords)
    right_eye_mask = mask_builder(right_eyes_coords)
    mouth_mask = mask_builder(mouth_coords)
    full_mouth_mask = mask_builder(full_mouth_coords)

    left_eyeshadow_mask = mask_builder(left_eyeshadow_coords)
    left_eyeshadow_mask = left_eyeshadow_mask - left_eye_mask

    right_eyeshadow_mask = mask_builder(right_eyeshadow_coords)
    right_eyeshadow_mask = right_eyeshadow_mask - right_eye_mask

    lips_mask = full_mouth_mask - mouth_mask

    face_mask = cv2.bitwise_or(cv2.bitwise_or(
        left_eye_mask, right_eye_mask), mouth_mask)
    eyeshadow_mask = cv2.bitwise_or(left_eyeshadow_mask, right_eyeshadow_mask)
    cv2.imwrite('tmp/dst.png', lips_mask)

    return face_mask, eyeshadow_mask, lips_mask


def prepare_mask(layer, src_mask):
    def build_mask(coords):
        base_mask = src_mask.copy()
        base_layer = layer.copy()
        mask = cv2.fillPoly(base_mask, [coords], (255, 255, 255))
        mask = cv2.bitwise_and(base_layer, base_layer, mask=base_mask)
        return mask

    return build_mask


def transform_with_mask(mask_points):
    def wrapper(transformer, *args): return transformer(*args, mask_points)
    return wrapper


def image_stats(image):
    # compute the mean and standard deviation of each channel
    (l, a, b) = image
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())
    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)


def laplacian_pyramind(A, B, m, num_levels=4):
    # assume mask is float32 [0,1]

    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.astype('float32').copy()
    GM[GM == 255] = 1.0
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]

    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpA = [gpA[num_levels-1]]
    lpB = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1, 0, -1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i], dstsize=size))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i], dstsize=size))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1])  # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        ls = la*gm + lb*(1.0-gm)
        ls = ls.astype('float32')
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, num_levels):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize=size)
        ls_ = cv2.add(ls_, np.float32(LS[i]))
        ls_[ls_ > 255] = 255
        ls_[ls_ < 0] = 0

    return ls_
