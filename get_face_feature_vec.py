import dlib
import numpy as np
import cv2
import os
import json
import argparse


def return_face_features(path_img):
    img = cv2.imread(path_img)
    if img.shape[0] * img.shape[1] > 400000:  # 对大图可以进行压缩，阈值可以自己设置
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    dets = detector(img, 1)  # 使用检测算子检测人脸，返回的是所有的检测到的人脸区域
    print("检测的人脸图像：", path_img, "\n")
    d = dets[0]     # 默认处理第一个检测到的人脸区域
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(d.left(), 0)
    bb[1] = np.maximum(d.top(), 0)
    bb[2] = np.minimum(d.right(), img.shape[1])
    bb[3] = np.minimum(d.bottom(), img.shape[0])

    rec = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
    shape = sp(img, rec)  # 获取landmark
    face_descriptor = facerec.compute_face_descriptor(img, shape)  # 使用resNet获取128维的人脸特征向量
    face_array = np.array(face_descriptor).reshape((1, 128))  # 转换成numpy中的数据结构

    # 显示人脸区域
    cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
    cv2.waitKey(2)
    cv2.imwrite(os.path.join(FLAGS.detected_faces, path_img.split('\\')[-1]), img)
    cv2.imshow('image', img)
    cv2.waitKey(1000)

    return face_array


def main():
    data = np.zeros((1, 128))  # 定义一个128维的空向量data
    label = []  # 定义空的list存放人脸的标签
    for file in os.listdir(FLAGS.input_faces):  # 遍历目录下的文件夹及文件
        path = os.path.join(FLAGS.input_faces, file)
        if os.path.isdir(path):     # 如果是目录
            feature_tmp = np.zeros((1, 128))
            label_name = file
            img_num = 0
            for image in os.listdir(path):
                if '.jpg' in image or '.png' in image:
                    img_num += 1
                    file_name = image.split('.')[0]
                    file_path = os.path.join(path, image)
                    print('current image: {}, current label: {}'.format(file_path, label_name))
                    feature_tmp += return_face_features(file_path)
            if img_num > 0:
                feature = feature_tmp / img_num
                data = np.concatenate((data, feature))  # 保存每个人的人脸特征
                label.append(label_name)  # 保存标签

    data = data[1:, :]  # 因为data的第一行是128维0向量，所以实际存储的时候从第二行开始
    np.savetxt(FLAGS.feature_dir, data, fmt='%f')  # 保存人脸特征向量合成的矩阵到本地
    label_file = open(FLAGS.label_dir, 'w')
    json.dump(label, label_file)  # 使用json保存list到本地
    label_file.close()
    cv2.destroyAllWindows()  # 关闭所有的窗口


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reco_model', type=str, help='the path of model',
                        default='dlib_face_recognition_resnet_model_v1.dat')
    parser.add_argument('--shape_predictor', type=str, help='the path of shape predictor',
                        default='shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--input_faces', type=str, help='the path of input faces data',
                        default='./face_repo')
    parser.add_argument('--detected_faces', type=str, help='the path to store the detected faces',
                        default='./face_detect')
    parser.add_argument('--test_faces', type=str, help='use the faces to test the model`s accuracy',
                        default='./face_test')
    parser.add_argument('--label_dir', type=str, help='the labels of the input faces',
                        default='./label.txt')
    parser.add_argument('--feature_dir', type=str, help='the features of the input faces',
                        default='./face_feature_vec.txt')
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed

if __name__ == '__main__':
    FLAGS, unparsed = parse_arguments()
    detector = dlib.get_frontal_face_detector()
    # detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    sp = dlib.shape_predictor(FLAGS.shape_predictor)
    facerec = dlib.face_recognition_model_v1(FLAGS.reco_model)

    main()
