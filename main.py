import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./xml/haarcascade_eye.xml')
eye_eyeglasses_cascade = cv2.CascadeClassifier('./xml/haarcascade_eye_tree_eyeglasses.xml')


cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)

    output = frame.copy()
    for (x, y, w, h) in faces:
        thresh = cv2.inRange(gray[y:y+h, x:x+w], (128), (255))
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        list_of_pts = [pt[0] for ctr in contours for pt in ctr]

        contour = np.array(list_of_pts).reshape((-1, 1, 2)).astype(np.int32)
        contour = cv2.convexHull(contour)

        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 1, cv2.LINE_AA)

        face_frame = gray[y:y+h, x:x+w]
        cv2.imshow("face", face_frame)

        eyes_glasses = eye_eyeglasses_cascade.detectMultiScale(face_frame, 1.1, 5)

        width = np.size(face_frame, 1)  # get face frame width
        height = np.size(face_frame, 0)  # get face frame height

        for (ex, ey, ew, eh) in eyes_glasses:
            if ey > height / 2:
                pass

            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 255), 2)

    cv2.imshow('Eye Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
