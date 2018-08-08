import dlib
import numpy as np
import cv2
import os
import json
import argparse


def find_most_likely_face(face_descriptor):
    face_repo = np.loadtxt(FLAGS.feature_dir, dtype=float)  # 载入本地人脸特征向量
    face_labels = open(FLAGS.label_dir, 'r')
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
    if min_distance > FLAGS.threshold:
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


def main():
    # 开始一张一张索引目录中的图像
    for file in os.listdir(FLAGS.test_faces):
        if '.jpg' in file or '.png' in file:
            print('current image: ', file)
            img = cv2.imread(os.path.join(FLAGS.test_faces, file))  # 使用opencv读取图像数据
            if img.shape[0] * img.shape[1] > 400000:  # 对大图可以进行压缩，阈值可以自己设置
                img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            recognition(img)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reco_model', type=str, help='the path of model used for recognising',
                        default='dlib_face_recognition_resnet_model_v1.dat')
    parser.add_argument('--shape_predictor', type=str, help='the path of shape predictor',
                        default='shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--test_faces', type=str, help='use the faces to test the model`s accuracy',
                        default='./face_test')
    parser.add_argument('--label_dir', type=str, help='the labels of the input faces',
                        default='./label.txt')
    parser.add_argument('--feature_dir', type=str, help='the features of the input faces',
                        default='./face_feature_vec.txt')
    parser.add_argument('--threshold', type=float,
                        help='the threshold is used to determine whether the input face belongs to the known faces',
                        default=0.4)
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed

if __name__ == '__main__':
    FLAGS, unparsed = parse_arguments()
    print(FLAGS)
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(FLAGS.shape_predictor)
    facerec = dlib.face_recognition_model_v1(FLAGS.reco_model)

    main()
