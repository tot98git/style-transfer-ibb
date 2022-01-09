from typing import SupportsComplex
import cv2
import mediapipe as mp
import numpy as np
import math
from indices import FACE, MOUTH


class FaceAlign:
    def detectFaces(self, img):
        cascadeFace = cv2.CascadeClassifier(
            "utils/haarcascade_frontalface_default.xml")
        detectionList = cascadeFace.detectMultiScale(img, 1.05, 5)

        return detectionList

    @staticmethod
    def annotate_img(mp_drawing, mp_face_mesh, mp_drawing_styles, annotated_image, face_landmarks):
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())

    def detect(self, face):
        # INITIALIZING OBJECTS
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh

        drawing_spec = mp_drawing.DrawingSpec(
            thickness=1, circle_radius=1)

        landmarks = []
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:

            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

            for face_landmarks in results.multi_face_landmarks:

                readable_landmarks = np.array([(p.x*face.shape[1], p.y*face.shape[0])
                                               for p in face_landmarks.landmark])
                landmarks.append(readable_landmarks)

        landmarks_target = np.int32(landmarks[0])[np.array(FACE)]
        src_mask = np.zeros((face.shape[0], face.shape[1]), dtype=np.uint8)
        cv2.fillPoly(src_mask, [landmarks_target], (255, 255, 255))

        dst = cv2.bitwise_and(face, face, mask=src_mask)

        cv2.imwrite('tmp/fc.png', dst)
        return landmarks[0], face, landmarks_target

    @staticmethod
    def calc_dist(p, q):
        try:
            return math.sqrt(math.pow((p[1]-p[0]), 2) - math.pow((q[1]-q[0]), 2))
        except:
            print(p)

    def warpImg(self, img1, img2, imgTarget):
        tps = cv2.createThinPlateSplineShapeTransformer()
        sshape = np.array(img1, np.float32)
        tshape = np.array(img2, np.float32)

        sshape = sshape.reshape(1, -1, 2)
        tshape = tshape.reshape(1, -1, 2)

        good_matches = [cv2.DMatch(i, i, 0)
                        for i in range(len(sshape[0])-10)]

        tps.estimateTransformation(sshape, tshape, good_matches)

        out_img = tps.warpImage(imgTarget)
        return out_img

    def main(self):
        landmarks_source, face1 = self.detect("toti_profile.jpg")
        landmarks_target, face2 = self.detect("target.jpg")
        cv2.imwrite('face1.png', face1)
        cv2.imwrite('face2.png', face2)

        self.warpImg(landmarks_target, landmarks_source, face1)
