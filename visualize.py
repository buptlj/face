import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import cv2


def create_sprite_image(images):
    if isinstance(images, list):
        images = np.array(images)
    # 获取图像的高和宽
    img_h = images.shape[1]
    img_w = images.shape[2]
    # 对图像数目开方，并向上取整，得到sprite图每边的图像数目
    num = int(np.ceil(np.sqrt(images.shape[0])))
    # 初始化sprite图
    sprite_image = np.zeros([img_h*num, img_w*num, 3], dtype=np.uint8)
    # 为每个小图像赋值
    for i in range(num):
        for j in range(num):
            cur = i * num + j
            if cur < images.shape[0]:
                sprite_image[i*img_h:(i+1)*img_h, j*img_w:(j+1)*img_w] = images[cur]

    return sprite_image


def visualisation(images, sprite_file, meta_file):
    # 定义一个新向量保存输出层向量的取值
    img_list = images.reshape(images.shape[0], -1)
    y = tf.Variable(img_list, name='images')
    summary_writer = tf.summary.FileWriter('./log')

    # ProjectorConfig帮助生成日志文件
    config = projector.ProjectorConfig()
    # 添加需要可视化的embedding
    embedding = config.embeddings.add()
    # 将需要可视化的变量与embedding绑定
    embedding.tensor_name = y.name

    # 指定embedding每个点对应的标签信息，
    # 这个是可选的，没有指定就没有标签信息
    embedding.metadata_path = meta_file
    # 指定embedding每个点对应的图像，
    # 这个文件也是可选的，没有指定就显示一个圆点
    embedding.sprite.image_path = sprite_file
    # 指定sprite图中单张图片的大小
    embedding.sprite.single_image_dim.extend([100, 100])

    # 将projector的内容写入日志文件
    projector.visualize_embeddings(summary_writer, config)

    # 生成会话，初始化新声明的变量并将需要的日志信息写入文件。
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join('./log', "model"), 1)

    summary_writer.close()


def img_data(img_path):
    imgs = np.array([], dtype=np.uint8)
    labels = []
    for file in os.listdir(img_path):
        if '.jpg' in file or '.png' in file:
            file_path = os.path.join(img_path, file)
            print('current image: {}'.format(file_path))
            label = file.split('_')[0]
            img = cv2.imread(file_path)
            img = cv2.resize(img, (100, 100))
            imgs = np.append(imgs, img)
            labels.append(label)

    imgs = np.reshape(imgs, (-1, 100, 100, 3))
    return imgs, labels


if __name__ == '__main__':
    input_data = './face_detect'
    log_dir = './log'
    sprite_file = 'img_sprite.png'
    meta_file = 'img_meta.tsv'
    imgs, labels = img_data(input_data)
    sprite_image = create_sprite_image(imgs)

    # 存储展示图像
    path_mnist_sprite = os.path.join(log_dir, sprite_file)
    cv2.imwrite(path_mnist_sprite, sprite_image)
    cv2.imshow('image', sprite_image)
    cv2.waitKey()

    # 存储每个下标对应的标签
    path_metadata = os.path.join(log_dir, meta_file)
    with open(path_metadata, 'w') as f:
        f.write('Index\tLabel\n')
        for index, label in enumerate(labels):
            f.write('{}\t{}\n'.format(index, label))

    visualisation(imgs, sprite_file, meta_file)
