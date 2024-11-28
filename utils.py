import platform
import subprocess
import typing

import cv2
import dlib
import numpy as np

_IMAGE_TYPE = cv2.Mat | np.ndarray[typing.Any, np.dtype] | np.ndarray
_FACE_CASCADE = cv2.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')
_EYE_CASCADE = cv2.CascadeClassifier('./xml/haarcascade_eye_tree_eyeglasses.xml')
_FACE_DETECTOR = dlib.get_frontal_face_detector()
_SHAPE_PREDICTOR = dlib.shape_predictor('./predictor/shape_predictor_68_face_landmarks.dat')


def get_processor_info():
    processor_info = ""

    if platform.system() == "Windows":
        processor_info = platform.processor()
    elif platform.system() == "Darwin":
        processor_info = subprocess.check_output(['/usr/sbin/sysctl', "-n", "machdep.cpu.brand_string"]).strip()
    elif platform.system() == "Linux":
        processor_info = subprocess.check_output(['sudo', 'dmidecode', '--type', 'processor'])
    return processor_info.decode('utf-8').replace("_", "").strip()


def eye_aspect_ratio(eye: _IMAGE_TYPE) -> np.float64:
    """
    Вычисление EAR показателей для глаз
    :param eye: Изображение глаза
    :return: Коэффициент ear
    """
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])

    ear = (A + B) / (2.0 * C)
    return ear


def get_ear_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_rects = _FACE_DETECTOR(gray, 1)

    if len(face_rects) == 0:
        return None

    ears_values: tuple = ()
    for rect in face_rects:
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        if w == 0 or h == 0:
            continue

        face = gray[y:y + h, x:x + w]

        rect = dlib.rectangle(0, 0, w, h)
        landmarks = _SHAPE_PREDICTOR(face, rect)

        left_eye_landmarks = []
        right_eye_landmarks = []

        for i in range(36, 42):
            left_eye_landmarks.append((landmarks.part(i).x, landmarks.part(i).y))

        for i in range(42, 48):
            right_eye_landmarks.append((landmarks.part(i).x, landmarks.part(i).y))

        left_eye_ear = eye_aspect_ratio(np.array(left_eye_landmarks))
        right_eye_ear = eye_aspect_ratio(np.array(right_eye_landmarks))

        # for marker in left_eye_landmarks:
        #     marker_x, marker_y = marker
        #     cv2.circle(face, (marker_x, marker_y), radius=2, color=(0, 255, 0), thickness=-1)
        #
        # for marker in right_eye_landmarks:
        #     marker_x, marker_y = marker
        #     cv2.circle(face, (marker_x, marker_y), radius=2, color=(0, 255, 0), thickness=-1)
        #
        # cv2.imshow("face", face)

        ears_values = (float(left_eye_ear), float(right_eye_ear))

    return ears_values
