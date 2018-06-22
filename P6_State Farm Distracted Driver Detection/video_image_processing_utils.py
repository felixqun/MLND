#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @license : Copyright(C)2018, Yu Q. Studio, All rights reserved. 
# @Author  : Yu Q.
# @Time    : 2018/3/19 17:35
# @Software: PyCharm
"""
    基于OpenCV、matplotlib的视频、图片处理的工具类
"""

import os
import pickle
import cv2 as cv
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
import matplotlib.pyplot as plt

import file_operation_utils as fou

def read_video_file(mkv_path):
    """
    使用OpenCV以视频流方式读入视频，返回每帧图片数组，数组shape为(img_Height,img_Width,img_Channel)，img_Channel顺序为RGB

    Args:
        mkv_path: str, 视频文件路径(相对路径或绝对路径)
    Returns:
        numpy.ndarray: 读取成功返回4D图片数据列表，shape为(frame_Count,img_Height,img_Width,img_Channel)；读取失败返回None
    """
    cap = cv.VideoCapture(mkv_path)
    if(not cap):
        return None
    img_array = []
    ret = True
    while(ret):
        ret, frame = cap.read()
        if not frame is None:
            img_array.append(frame)
    cap.release()
    return img_array

def read_image_file(img_path, flags=cv.IMREAD_COLOR, dtype_float32=True, display=False, target_size=None):
    """
    读入图片，返回RGB顺序的图片数组；如果需要的话，显示图片、缩放图片

    Args:
        img_path: str, 图片文件路径(相对路径或绝对路径)
        flags: int, 指定读入图片的方式，默认为正常读入，cv.IMREAD_GRAYSCALE：读入灰度图片，cv.IMREAD_UNCHANGED：alpha通道读入
        dtype_float32: bool, 是否将图片数组元素转换成np.float32, 默认为True
        display: bool, 是否显示图片，默认不显示，设置为True时显示读入的图片
        target_size: (int,int), 读入图片后进行缩放，目标图片尺寸，默认为None，图片保持原尺寸
    Returns:
        numpy.ndarray: 读入成功返回3D图片数组，否则返回None
    """
    img_array = cv.imread(img_path, flags)
    img_array = transfer_img_BRG_to_RGB_mat(img_array)
    if display:
        file_name = img_path.split("/")[-1]
        cv.imshow(file_name, img_array)
        cv.waitKey(0)
        cv.destroyAllWindows()
    if dtype_float32:
        img_array = np.array(img_array, dtype=np.float32)
    if target_size is not None:
        img_array = img_resize(img_array, target_size)
    return img_array

def read_image_file_preprocess(img_name, target_img_shape, preprocess_fun=None):
    """
    读入单张图片并进行预处理，返回单张图片数组

    Args:
        img_name: str, 图片路径名称
        target_img_shape: (int, int, int), 图片shape:(height, width, channel)
        preprocess_fun: function, 图片数组预处理函数
    Returns:
        numpy.array: 图片数组(3D)
    """
    img_arr = read_image_file(img_name)
    img_arr = img_resize(img_arr, (target_img_shape[0], target_img_shape[1]))
    if preprocess_fun is not None:
        img_arr = preprocess_fun(img_arr)
    return img_arr

def transfer_img_BRG_to_RGB_mat(BRG_order_mat_data):
    """
    opencv的接口采用图片的BGR顺序，在使用matplotlib.pyplot等显示或处理时，其顺序为RGB；
    本函数将BGR顺序的图片数组转换为RBG顺序

    Args:
        BRG_order_mat_data: numpy.ndarray, BGR顺序的图片数组
    Returns:
        numpy.ndarray: RGB顺序的图片数组
    """
    b, g, r = cv.split(BRG_order_mat_data)
    return cv.merge([r, g, b])
    # 另一种方式：利用3D数组翻转
    # return BRG_order_mat_data[:, :, ::-1]

def show_image_by_mat_data(mat_data, RGB_order=True, img_name="show_img"):
    """
    依据图片列表数据显示图片，图片数据维度为(img_Height，img_Width，img_Channel),当图片顺序为BGR时，显示正常图片，CV中所有默认顺序均为BGR

    Args:
        mat_data: numpy.ndarray, 3D图片数组
        RGB_order: bool, 真表示以RGB顺序显示，假表示以BGR顺序显示，默认为真
        img_name: str, 图片名称
    Returns:
        无
    """
    if RGB_order:
        mat_data = mat_data[:, :, ::-1]
    cv.imshow(img_name, mat_data)
    cv.waitKey(0)
    # cv.destroyAllWindows()

def show_img_plt(mat_data):
    """
    使用matplotlib.pyplot的imshow函数显示图片，图片的通道顺序为RGB

    Args:
        mat_data: numpy.array, 图片数组
    Returns:
        无
    """
    plt.imshow(mat_data)
    plt.axis('off')
    plt.show()

def hide_subplot(subplot):
    """
    去掉子图的坐标刻度、方框线图示

    Args:
        subplot: Axes, 子图(Axes类的对象)
    Returns:
        无
    """
    subplot.set_axis_off()
    subplot.set_xticks([])
    subplot.set_yticks([])

def draw_bar_in_subplot(subplot, x_label_list, y, title, show_text=False, text_size=7):
    """
    在子图中绘制柱状图

    Args:
        subplot: Axes, 子图(Axes类的对象)
        x_label_list: list, 字符串列表，用于在x轴下方显示刻度标记
        y: list, 待绘图数据列表或数组
        title: str, 子图名称
        show_text: bool, 在柱状图上显示文本，默认不显示(参数值为False)，需显示时将其设置为True
        text_size: int, 柱状图文本字号
    Returns:
        Axes: 子图
    """
    x = np.arange(len(y)) + 1
    subplot.bar(x, y, width=0.7, align='center', color='b', alpha=0.8)
    subplot.set_xticks(x)
    subplot.set_xticklabels(x_label_list)
    subplot.set_title(title)
    if show_text:
        subplot.set_ylim(0, max(list(y)) + text_size*3)
        for a,b in zip(x,y):
            subplot.text(a, b+0.05, '%.0f' % b, ha='center', va='bottom', fontsize=text_size)
    return subplot

def draw_bar(plot, x_label_list, y, title='', show_text=False, text_size=7):
    """
    绘制柱状图

    Args:
        plot: matplotlib.pyplot, matplotlib.pyplot对象
        x_label_list: list, 字符串列表，用于在x轴下方显示刻度标记
        y: list, 待绘图数据列表或数组
        title: str, 子图名称
        show_text: bool, 在柱状图上显示文本，默认不显示(参数值为False)，需显示时将其设置为True
        text_size: int, 柱状图文本字号
    Returns:
        matplotlib.pyplot: 子图
    """
    x = np.arange(len(y)) + 1
    plot.bar(x, y, width=0.7, align='center', color='b', alpha=0.8)
    plot.xticks(x, x_label_list)
    plot.title(title)
    if show_text:
        plot.ylim(0, max(list(y)) + text_size * 3)
        for a,b in zip(x,y):
            plot.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=text_size)
    return plot

def img_resize(img_mat_data, target_width_height_tuple=(0, 0), target_fx=0, target_fy=0, interpolation=cv.INTER_AREA):
    """
    依据图片数组，缩放图片

    Args:
        img_mat_data: numpy.ndarray, 原图片数组
        target_width_height_tuple: (int,int), 缩放输出图片的(宽度, 高度)，该参数必须为仅含2个大于0的元素的元组；
                                     该参数与after_fx、after_fy同时有值时，只有该参数生效；
                                     该参数与after_fx、after_fy不能同时为0
        target_fx: float, 缩放后的x方向比例
        target_fy: float, 缩放后的y方向比例
        interpolation: int, 缩放变换方法，参数取值同原参数，默认为cv.INTER_AREA，其它参数可取: cv.INTER_NEAREST, cv.INTER_LINEAR(原函数默认实参),cv.INTER_CUBIC,cv.INTER_LANCZOS4
    Returns:
        numpy.ndarray: 缩放后的图片数组
    """
    assert isinstance(target_width_height_tuple, tuple) and len(target_width_height_tuple) == 2, \
        'target_width_height_tuple parameter must be a 2 elements tuple'
    return cv.resize(img_mat_data, target_width_height_tuple, fx=target_fx, fy=target_fy, interpolation=interpolation)

def img_preprocess_and_save(normalize, one_hot_encode, features, labels, total_label_list, filename):
    """
    预处理图片数据并将图片数据存储为文件；预处理包括标准化处理样本特征、one-hot编码样本标记

    Args:
        normalize: function, 标准化处理函数
        one_hot_encode: function, one-hot编码函数
        features: numpy.ndarray, 特征列表
        labels: numpy.ndarray, 样本标记列表
        total_label_list: list, 所有标记构成的列表
        filename: str, 待保存的文件名称
    Returns:
        无
    """
    features = normalize(features)
    labels = one_hot_encode(labels, total_label_list)
    pickle.dump((features, labels), open(filename, 'wb'))

def img_normalize(x, min=0, max=1):
    """
    将图片数组(值区间为(0,255))进行标准化，使其值区间为(0,1)；x的shape为(batch,img_Height，img_Width，img_Channel)或者(img_Height，img_Width，img_Channel)

    Args:
        x: numpy.ndarray, 图片列表数据，形状为(img_Height，img_Width，img_Channel)
        min: int, 标准化处理后的数据下限
        max: int, 标准化处理后的数据上限
    Returns:
        numpy.ndarray: 标准化之后的图片数组
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    scale_min = 0
    scale_max = 255
    return min + (((x - scale_min) * (max - min)) / (scale_max - scale_min))

def img_normalize_reverse(normalized_x):
    """
    还原标准化处理后的图片数组

    Args:
        normalized_x: numpy.ndarray, 3D图片数组，形状为(img_Height，img_Width，img_Channel)
    Returns:
        numpy.ndarray: 3D图片数组，最内层元素值区间为[0,255]
    """
    a = 0
    b = 1
    scale_min = 0
    scale_max = 255
    mat_data = scale_min + (((normalized_x - a) * (scale_max - scale_min)) / (b - a))
    return mat_data.astype(np.uint8)

def gen_one_hot_mat(label_list_len):
    """
    依据所有标记构成的列表的长度，将标记转换为2D列表(方阵)

    Args:
        label_list_len: list, 所有标记构成的列表的长度
    Returns:
        numpy.ndarray: 2D数组
    """
    encoder = LabelBinarizer()
    seed_vector = np.arange(label_list_len)
    encoder.fit(seed_vector)
    labels = encoder.transform(seed_vector)
    return labels.astype(np.int8)
    # return labels.astype(np.float32)

def label_one_hot_encode(y, total_label_list):
    """
    依据标记列表，获取指定标记的one-hot编码(2D列表)

    Args:
        y: list, 标记列表
        total_label_list: list, 所有标记构成的列表
    Returns:
        numpy.ndarray: one-hot编码后的2D数组
    """
    oh_mat = gen_one_hot_mat(len(total_label_list))
    y_l = list(y)
    for index in np.arange(len(y_l)):
        y_label = y_l[index]
        for label_index, label_name in zip(np.arange(len(total_label_list)), total_label_list):
            if y_label == label_name:
                y_l[index] = oh_mat[label_index]
                break
    return np.array(y_l)

def img_mat_save_features_labels_in_batch(X, y, total_y, normalize, one_hot_encode, file_name_prefix, file_name_suffix='.p', batch_size=0):
    """
    图片为4D数组时，将图片样本特征、样本标记分别经过标准化、one-hot编码后分批存成数据文件

    Args:
        X: numpy.ndarray, 样本特征
        y: numpy.ndarray, 样本标记
        total_y: list, 所有样本标记列表
        normalize: function, 标准化处理函数
        one_hot_encode: function, one-hot编码函数
        file_name_prefix: str, 待保存的数据文件前缀
        file_name_suffix: str, 待保存的数据文件后缀
        batch_size: int, 每批样本数量,默认为0(共1批)
    Returns:
        int: 分批数量(生成的数据文件个数)
    """
    if len(X) != len(y):
        return None

    # 分批写入
    total_count = len(y)
    batch_size = total_count if batch_size < 1 or batch_size > total_count else batch_size
    batch_count = int(np.ceil(total_count / batch_size))
    for batch_i in tqdm(range(batch_count), desc='Saving 4D images data files', unit='file'):
        start = batch_i * batch_size
        end = min((batch_i + 1) * batch_size, total_count)
        X_split = X[start:end]
        y_split = y[start:end]
        img_preprocess_and_save(normalize,
                                one_hot_encode,
                                X_split,
                                y_split,
                                total_y,
                                file_name_prefix + str(batch_i) + file_name_suffix)
    return batch_count

def img_save_features_labels_in_batch(img_file_dir, X_img_names, y, total_y, normalize, one_hot_encode, data_file_dir, file_name_prefix, file_name_suffix='.p', batch_size=0, target_height_width_tuple=(0, 0)):
    """
    依据图片名称列表及标记列表，将图片分批读入内存，并对图片4D数组样本特征、样本标记分别经过标准化、one-hot编码后存成数据文件；该操作会覆盖同名文件

    Args:
        img_file_dir: str, 图片所在目录
        X_img_names: list, (样本特征)图片名称列表
        y: list, 样本标记
        total_y: list, 所有样本标记列表
        normalize: function, 标准化处理函数
        one_hot_encode: function, one-hot编码函数
        data_file_dir: str, 待保存数据文件的目录
        file_name_prefix: str, 待保存的数据文件前缀
        file_name_suffix: str, 待保存的数据文件后缀
        batch_size: int, 每批样本数量,默认为0(共1批)
        target_height_width_tuple: (int,int), 缩放输出图片的(宽度, 高度)，该参数默认为(0,0)，不对原图进行缩放
    Returns:
        int: 分批数量(生成的数据文件个数)
    """
    assert len(X_img_names) == len(y), 'len(X_img_names) is not equal len(y).'
    fou.mkdir(data_file_dir)
    # 分批读取图片、写入数据文件
    total_count = len(y)
    batch_size = total_count if batch_size < 1 or batch_size > total_count else batch_size
    batch_count = int(np.ceil(total_count / batch_size))
    for batch_i in tqdm(range(batch_count), desc='Reading and saving 4D images data files', unit='batch'):
        start = batch_i * batch_size
        end = min((batch_i + 1) * batch_size, total_count)
        X_name_split = X_img_names[start:end]
        y_split = y[start:end]
        # 读取图片
        X_split = img_read_to_array_batch(img_file_dir, X_name_split, target_height_width_tuple)
        # 存储图片
        img_preprocess_and_save(normalize,
                                one_hot_encode,
                                X_split,
                                y_split,
                                total_y,
                                os.path.join(data_file_dir, file_name_prefix + str(batch_i) + file_name_suffix))
    return batch_count


def img_read_to_array(img_dir, img_names, target_img_shape, preprocess_fun):
    """
    指定目录，依据图片名称将其读入内存并进行预处理，形成4D数组; 读取时依据名称列表及图片形状一次性分配内存

    Args:
        img_dir: str, 图片所在目录
        img_names: list, 图片名称列表
        target_img_shape: (int, int, int), 图片shape:(height, width, channel)
        preprocess_fun: function, 图片数组预处理函数
    Returns:
        numpy.array: 图片数组(4D)
    """
    img_arr = np.empty((len(img_names), *target_img_shape), dtype=np.float32)
    for i in tqdm(range(len(img_names)), desc='Loading image data', unit='files'):
        img = read_image_file(os.path.join(os.path.join(img_dir, img_names[i])))
        img = img_resize(img, (target_img_shape[0], target_img_shape[1]))

        # img = cv.imread(os.path.join(img_dir, img_names[i]))
        # img = cv.resize(img, (target_img_shape[0], target_img_shape[1]))
        # img = img[:, :, ::-1]
        # img = np.array(img, dtype=np.float32)

        img = preprocess_fun(img)
        img_arr[i] = img
    return img_arr

def img_read_to_array_batch(img_file_dir, img_name_list, target_width_height_tuple=(0, 0)):
    """
    指定目录，依据图片名称将其读入内存，形成4D数组;
    图片数组列表添加方式为append，适用于分批读入的情况

    Args:
        img_file_dir: str, 图片所在目录
        img_name_list: list, 图片名称数组
        target_width_height_tuple: (int,int), 缩放输出图片的(宽度, 高度)，该参数必须为仅含2个大于0的元素的元组
    Returns:
        numpy.ndarray: 4D图片数据数组
    """
    img_data_list = []
    for img_name in img_name_list:
        img_data_mat = read_image_file(os.path.join(img_file_dir, img_name))
        assert img_data_mat is not None, "Image file not found. Image path:{}".format(os.path.join(img_file_dir, img_name))
        if target_width_height_tuple[0] < 1 or target_width_height_tuple[1] < 1:
            img_data_list.append(img_data_mat)
        else:
            img_data_list.append(img_resize(img_data_mat, target_width_height_tuple))
    return np.array(img_data_list)


def img_batch_features_labels(features, labels, batch_size):
    """
    依据每批次样本行数量返回样本特征及标记，供img_load_features_labels_in_batch调用

    Args:
        features: numpy.ndarray, 样本特征数组
        labels: numpy.ndarray, 样本标记数组
        batch_size: int, 每批次的样本(行)数量
    Returns:
        generator: 返回指定数量的样本
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

def img_load_features_labels_in_batch(file_dir, file_prefix, batch_id, batch_size, file_suffix='.p'):
    """
    读取指定目录中的保存为数据文件的样本

    Args:
        file_dir: str, 数据文件目录
        file_prefix: str, 数据文件前缀
        batch_id: int, 批次ID(从0开始，与存储时的文件批次ID相对应)
        batch_size: int, 每批次的样本(行)数量
        file_suffix: str, 数据文件后缀
    Returns:
        generator: 依据batch_size返回相应数量的样本，其格式为：features(样本特征, numpy.ndarray), labels(样本标记, numpy.ndarray)
    Example:
        在外层调用时，形如:
        for epoch in range(epochs):
            n_batches = 5  Args:
            for batch_i in range(n_batches):
                for batch_features, batch_labels in img_load_features_labels_in_batch(batch_i, batch_size, 'data/cnn_train_'):
                    train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
                print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')  Args:
                print_stats(sess, batch_features, batch_labels, cost, accuracy)
        不使用for循环时，读入所有数据

    """
    filename = os.path.join(os.getcwd(), file_dir, file_prefix + str(batch_id) + file_suffix)
    features, labels = pickle.load(open(filename, mode='rb'))
    return img_batch_features_labels(features, labels, batch_size)


def img_generate_batch_data_from_file(file_dir, file_prefix, batch_size, file_suffix='.p'):
    """
    从指定目录中分批读取指定前缀的所有图片样本数据，供分批训练、验证使用

    Args:
        file_dir: str, 数据文件所在目录
        file_prefix: str, 数据文件前缀
        batch_size: int, 每批数量
        file_suffix: str, 数据文件后缀
    Returns:
        generator: 形如(inputs, targets)的元组, 返回指定数量的样本
    """
    file_dir = os.path.join(os.getcwd(), file_dir)
    while 1:
        file_count = fou.get_prefix_file_count(file_dir, file_prefix)
        for file_bid in range(file_count):
            filename = os.path.join(file_dir, file_prefix + str(file_bid) + file_suffix)
            features, labels = pickle.load(open(filename, mode='rb'))
            for start in range(0, len(features), batch_size):
                end = min(start + batch_size, len(features))
                yield (features[start:end], labels[start:end])


def splice_two_image_left_right(img_left, img_right):
    """
    2张大小及通道数量相同的图片，图片数组格式为(height,width,channel)， 左右各取一半，拼接成新图片

    Args:
        img_left: numpy.array, 左侧图片数组
        img_right: numpy.array, 右侧图片数组
    Returns:
        numpy.array: 拼接后的图片数组
    """
    half_cols = len(img_left[1]) // 2
    img_left[:, half_cols:, :] = img_right[:, half_cols:, :]
    return img_left










