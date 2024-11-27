import cv2
import numpy as np


def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        else:
            biggest = np.array([biggest], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None

    return biggest


def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6:  # if there are more than 5 non working ports stop the testing.
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." % dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" % (dev_port, h, w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." % (dev_port, h, w))
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports, non_working_ports


def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = (x, y, w, h)
        else:
            right_eye = (x, y, w, h)

    return left_eye, right_eye


def get_eye_parameters(image: cv2.Mat):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # eye_region = gray[y:y + h, x:x + w]
    # eye_color = img[y:y + h, x:x + w]

    # Определяем нижнее и верхнее веко с помощью градиента
    h, w, _ = image.shape

    eye_roi = gray[int(h / 3):int(2 * h / 3), :]
    _, thresh = cv2.threshold(eye_roi, 60, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Находим контуры для измерения расстояния между веками
    if contours:
        contour = max(contours, key=cv2.contourArea)
        (x_min, y_min, w_min, h_min) = cv2.boundingRect(contour)
        distance = h_min

        # Используем метод для нахождения зрачка
        ret, thresh_eye = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours_eye, _ = cv2.findContours(thresh_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_eye:
            contour_eye = max(contours_eye, key=cv2.contourArea)
            (x_eye, y_eye, w_eye, h_eye) = cv2.boundingRect(contour_eye)
            pupil_size = w_eye * h_eye

            return distance, pupil_size

    return None, None