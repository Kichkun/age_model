import math
import warnings
from copy import copy

from keras.preprocessing import image

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import json
import cv2
from PIL import Image
from age_model import Age


def distance(a, b):
    x1 = a[0];
    y1 = a[1]
    x2 = b[0];
    y2 = b[1]

    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


def get_opencv_path():
    opencv_home = cv2.__file__
    folders = opencv_home.split(os.path.sep)[0:-1]

    path = folders[0]
    for folder in folders[1:]:
        path = path + "/" + folder

    face_detector_path = path + "/data/haarcascade_frontalface_default.xml"
    eye_detector_path = path + "/data/haarcascade_eye.xml"

    if os.path.isfile(face_detector_path) != True:
        raise ValueError("Confirm that opencv is installed on your environment! Expected path ", face_detector_path,
                         " violated.")

    return path + "/data/"


def detectFace(img, target_size=(224, 224), grayscale=False, enforce_detection=True):

    # -----------------------

    exact_image = False
    if type(img).__module__ == np.__name__:
        exact_image = True

    # -----------------------

    opencv_path = get_opencv_path()
    face_detector_path = opencv_path + "haarcascade_frontalface_default.xml"
    eye_detector_path = opencv_path + "haarcascade_eye.xml"

    if os.path.isfile(face_detector_path) != True:
        raise ValueError("Confirm that opencv is installed on your environment! Expected path ", face_detector_path,
                         " violated.")

    # --------------------------------

    face_detector = cv2.CascadeClassifier(face_detector_path)
    eye_detector = cv2.CascadeClassifier(eye_detector_path)

    if exact_image != True:  # image path passed as input

        if os.path.isfile(img) != True:
            raise ValueError("Confirm that ", img, " exists")

        img = cv2.imread(img)

    img_raw = img.copy()

    # --------------------------------

    faces = []

    try:
        faces = face_detector.detectMultiScale(img, 1.3, 5)
    except:
        pass

    # print("found faces in ",image_path," is ",len(faces))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        detected_face = img[int(y):int(y + h), int(x):int(x + w)]
        detected_face_gray = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)

        # ---------------------------
        # face alignment

        eyes = eye_detector.detectMultiScale(detected_face_gray)

        if len(eyes) >= 2:
            # find the largest 2 eye
            base_eyes = eyes[:, 2]

            items = []
            for i in range(0, len(base_eyes)):
                item = (base_eyes[i], i)
                items.append(item)

            df = pd.DataFrame(items, columns=["length", "idx"]).sort_values(by=['length'], ascending=False)

            eyes = eyes[df.idx.values[0:2]]

            # -----------------------
            # decide left and right eye

            eye_1 = eyes[0];
            eye_2 = eyes[1]

            if eye_1[0] < eye_2[0]:
                left_eye = eye_1
                right_eye = eye_2
            else:
                left_eye = eye_2
                right_eye = eye_1

            # -----------------------
            # find center of eyes

            left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
            left_eye_x = left_eye_center[0];
            left_eye_y = left_eye_center[1]

            right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
            right_eye_x = right_eye_center[0];
            right_eye_y = right_eye_center[1]

            # -----------------------
            # find rotation direction

            if left_eye_y > right_eye_y:
                point_3rd = (right_eye_x, left_eye_y)
                direction = -1  # rotate same direction to clock
            else:
                point_3rd = (left_eye_x, right_eye_y)
                direction = 1  # rotate inverse direction of clock

            # -----------------------
            # find length of triangle edges

            a = distance(left_eye_center, point_3rd)
            b = distance(right_eye_center, point_3rd)
            c = distance(right_eye_center, left_eye_center)

            # -----------------------
            # apply cosine rule

            if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

                cos_a = (b * b + c * c - a * a) / (2 * b * c)
                angle = np.arccos(cos_a)  # angle in radian
                angle = (angle * 180) / math.pi  # radian to degree

                # -----------------------
                # rotate base image

                if direction == -1:
                    angle = 90 - angle

                img = Image.fromarray(img_raw)
                img = np.array(img.rotate(direction * angle))

                # you recover the base image and face detection disappeared. apply again.
                faces = face_detector.detectMultiScale(img, 1.3, 5)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    detected_face = img[int(y):int(y + h), int(x):int(x + w)]

        # -----------------------

        # face alignment block end
        # ---------------------------

        # face alignment block needs colorful images. that's why, converting to gray scale logic moved to here.
        if grayscale == True:
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)

        detected_face = cv2.resize(detected_face, target_size)

        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        # normalize input in [0, 1]
        img_pixels /= 255

        return img_pixels

    else:

        if (exact_image == True) or (enforce_detection != True):

            if grayscale == True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = cv2.resize(img, target_size)
            img_pixels = image.img_to_array(img)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255
            return img_pixels
        else:
            raise ValueError(
                "Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")


def analyze(images, enforce_detection=True):
    if type(images) == list:
        imgs = copy(images)
        bulkProcess = True
    else:
        imgs = [images]
        bulkProcess = False

    age_model = Age.loadModel()

    resp_objects = []

    global_pbar = tqdm(range(0, len(imgs)), desc='Analyzing')

    # for img_path in img_paths:
    for j in global_pbar:
        img = imgs[j]

        resp_obj = "{"

        img_224 = detectFace(img, target_size=(224, 224), grayscale=False,
                             enforce_detection=enforce_detection)
        age_predictions = age_model.predict(img_224)[0, :]
        apparent_age = Age.findApparentAge(age_predictions)

        resp_obj += "\"age\": %s" % (apparent_age)

        resp_obj += "}"

        resp_obj = json.loads(resp_obj)

        if bulkProcess == True:
            resp_objects.append(resp_obj)
        else:
            return resp_obj

    if bulkProcess == True:
        resp_obj = "{"

        for i in range(0, len(resp_objects)):
            resp_item = json.dumps(resp_objects[i])

            if i > 0:
                resp_obj += ", "

            resp_obj += "\"instance_" + str(i + 1) + "\": " + resp_item
        resp_obj += "}"
        resp_obj = json.loads(resp_obj)
        return resp_obj
