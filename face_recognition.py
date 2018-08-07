import dlib
import numpy as np
import cv2
import os
import json

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
image_test = './face_test/'
threshold = 0.4


def find_most_likely_face(face_descriptor):
    face_repo = np.loadtxt('face_feature_vec.txt', dtype=float)  # 载入本地人脸特征向量
    face_labels = open('label.txt', 'r')
    label = json.load(face_labels)  # 载入本地人脸库的标签
    face_labels.close()

    face_distance = face_descriptor - face_repo
    euclidean_distance = 0
    if len(label) == 1:
        euclidean_distance = np.linalg.norm(face_distance)
    else:
        euclidean_distance = np.linalg.norm(face_distance, axis=1, keepdims=True)
    min_distance = euclidean_distance.min()
    print('distance: ', min_distance)
    if min_distance > threshold:
        return 'other'
    index = np.argmin(euclidean_distance)

    return label[index]


def recognition(img):
    dets = detector(img, 1)
    bb = np.zeros(4, dtype=np.int32)
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

        bb[0] = np.maximum(d.left(), 0)
        bb[1] = np.maximum(d.top(), 0)
        bb[2] = np.minimum(d.right(), img.shape[1])
        bb[3] = np.minimum(d.bottom(), img.shape[0])
        rec = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
        shape = sp(img, rec)
        face_descriptor = facerec.compute_face_descriptor(img, shape)

        class_pre = find_most_likely_face(face_descriptor)
        print(class_pre)
        cv2.rectangle(img, (rec.left(), rec.top()), (rec.right(), rec.bottom()), (0, 255, 0), 2)
        cv2.putText(img, class_pre, (rec.left(), rec.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('image', img)
    cv2.waitKey()

# 开始一张一张索引目录中的图像
for file in os.listdir(image_test):
    if '.jpg' in file or '.png' in file:
        fileName = file.split('.')[0]
        print('current image: ', file)
        img = cv2.imread(os.path.join(image_test, file))  # 使用opencv读取图像数据
        if img.shape[0] * img.shape[1] > 400000:  # 对大图可以进行压缩，阈值可以自己设置
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        recognition(img)
