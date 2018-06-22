#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @license : Copyright(C)2018, Yu Q. Studio, All rights reserved.
# @Author  : Yu Q.
# @Time    : 2018/3/20 17:13
# @Software: PyCharm
"""
    侦测走神司机:数据预处理及项目相关工具模块
"""

import numpy as np
import pandas as pd
import os
import math
import random
import cv2 as cv
from tqdm import tqdm
from six.moves import range
from sklearn.model_selection import train_test_split
from keras.applications import Xception, xception
from keras.models import load_model
from keras.utils import multi_gpu_model

import general_utils as gu
import video_image_processing_utils as vip
import file_operation_utils as fou

# csv文件名称
CSV_NAME = 'driver_imgs_list.csv'
# 分类列表
TOTAL_LABELS = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
# 组合后的分类名称/图像名称
CLASSNAME_IMGNAME = 'classname_imgname'
# 数据存放的根目录
DATA_ROOT_DIR = 'data'
# 训练集图像存放目录
IMG_TRAIN_DIR = 'train'
# 测试集图像存放目录
IMG_TEST_DIR = 'test'
# 数据预处理存放目录
DATA_PREPROCESS = 'data_preprocess'
# 测试集数据文件前缀
IMG_TEST_DATA_PREFIX = 'img_test_'
# 结果存放主目录
RESULTS_DIR = 'results'
# compile后保存模型的前缀
M_COMPILE_PREFIX = 'm_compile_'
# Checkpoint后保存模型的前缀
M_CHECKPOINT_PREFIX = 'm_checkpoint_'
# 可视化日志目录
TENSORBOARD = 'tensorboard'
# 预测结果目录
SUMMARIES_DIR = 'summaries'

def get_kaggle_data(file_url, data_dir, zip_name, file_name):
    """
    下载数据并解压

    Args:
        file_url: str, 下载地址
        data_dir: str, 数据存放目录
        zip_name: str, 待解压的zip文件名称
        file_name: str, 解压后的某个文件名称，以此判断数据文件是否已经存在，避免重复下载、解压
    Returns:
        无
    """
    cwd = os.getcwd()
    data_path = os.path.join(cwd, data_dir)

    # 判断driver_imgs_list.csv.zip
    file_n = os.path.join(data_path, file_name)
    if os.path.isfile(file_n):
        print("{} already exist.".format(file_n))
    else:
        zip_file_n = os.path.join(data_path, zip_name)
        if os.path.isfile(zip_file_n):
            # 解压
            fou.un_zip(zip_file_n, des_dir=data_path)
        else:
            pass
            # 下载
            # fou.download_login_big_data_file(file_url, username, psw, data_path)


def unzip_kaggle_data(data_dir, zip_name, file_name):
    """
    下载数据并解压

    Args:
        data_dir: str, 数据存放目录
        zip_name: str, 待解压的zip文件名称
        file_name: str, 解压后的某个文件名称，以此判断数据文件是否已经存在，避免重复下载、解压
    Returns:
        无
    """
    cwd = os.getcwd()
    data_path = os.path.join(cwd, data_dir)

    # 判断driver_imgs_list.csv.zip
    file_n = os.path.join(data_path, file_name)
    if os.path.isfile(file_n):
        print("{} already exist.".format(file_n))
    else:
        zip_file_n = os.path.join(data_path, zip_name)
        if os.path.isfile(zip_file_n):
            # 解压
            fou.un_zip(zip_file_n, des_dir=data_path)
        else:
            pass
            # 下载
            # fou.download_login_big_data_file(file_url, username, psw, data_path)

def read_csv_file():
    """
    读取csv文件，并组合分类名称和图像名称

    Returns:
        pandas.core.frame.DataFrame:csv文件读取内容，新加列存储组合后的分类名称和图像名称
    """
    csv_data = pd.read_csv(os.path.join(DATA_ROOT_DIR, CSV_NAME))
    class_name_img = []
    for i in range(len(csv_data['img'])):
        class_name_img.append(os.path.join(csv_data['classname'][i], csv_data['img'][i]))
    csv_data[CLASSNAME_IMGNAME] = class_name_img
    return csv_data

def load_csv_data_and_split_train_validation_set(test_size=0.2):
    """
    载入cvs文件数据，并将图像文件名称划分为训练集、验证集

    Args:
        test_size: float, 测试集所占比例
    Returns:
        list,list,list,list: 训练集图像文件名称列表,训练集图像分类列表,验证集图像文件名称列表,验证集图像分类列表
    """
    csv_data = read_csv_file()
    csv_data = csv_data.sample(frac=1, random_state=4)

    # debug
    gu.p('len(csv_data)', len(csv_data))
    print(csv_data.head(5))

    x_train_names, x_val_names, y_train, y_val = train_test_split(csv_data[CLASSNAME_IMGNAME], csv_data['classname'], test_size=test_size, random_state=4,
                                                      stratify=csv_data['classname'])
    return list(x_train_names), list(x_val_names), list(y_train), list(y_val)

def generate_features_labels_array(img_names, labels, target_img_shape):
    """
    依据图像名称列表、标记列表，载入与之相应的预处理后的图像特征数组(4D)，one-hot编码后的标记数组(2D)

    Args:
        img_names: list, 图像名称列表
        labels: list, 标记列表
        target_img_shape: (int, int, int), 图像shape:(height, width, channel)
    Returns:
        numpy.array, numpy.array:图像数组(4D), 标记数组(2D)
    """
    X = vip.img_read_to_array(os.path.join(os.getcwd(), DATA_ROOT_DIR, IMG_TRAIN_DIR), img_names, target_img_shape, xception.preprocess_input)
    y = vip.label_one_hot_encode(labels, TOTAL_LABELS)
    return X, y

def get_transfer_learning_sample_by_driverID(driver_id):
    """
    依据驾驶员ID号获取数据集

    Args:
        driver_id: str, 驾驶员ID号
    Returns:
        list, list: 图像文件名称列表, 图像分类列表
    """
    csv_data = read_csv_file()
    grouped_imgs = csv_data.groupby('subject')
    driver_id_df = grouped_imgs.get_group(driver_id)
    return driver_id_df[CLASSNAME_IMGNAME], driver_id_df['classname']

def get_transfer_learning_sample_by_driverID_list(driver_id_list):
    """
    依据驾驶员ID列表获取数据集

    Args:
        driver_id_list: str, 驾驶员ID列表
    Returns:
        list, list: 图像文件名称列表, 图像分类列表
    """
    csv_data = read_csv_file()
    grouped_imgs = csv_data.groupby('subject')
    img_names = []
    img_labels = []
    for driver_id in driver_id_list:
        driver_id_df = grouped_imgs.get_group(driver_id)
        img_names.extend(driver_id_df[CLASSNAME_IMGNAME])
        img_labels.extend(driver_id_df['classname'])
    return img_names, img_labels

def train_val_names_divided_by_driverID(train_rate=0.8, val_driver_count=0):
    """
    依据驾驶者ID，划分训练集、验证集图像名称列表

    Args:
        train_rate:float, 训练集所占比例
        val_driver_count:int, 验证集驾驶者ID数量，默认为0，验证集包括除训练集之外的所有驾驶者
    Rreturns:
        list, list, list, list: 训练集特征, 训练集标记, 验证集特征, 验证集标记
    """
    csv_data = read_csv_file()
    driver_ids = set(csv_data['subject'])
    train_driver_ids = random.sample(driver_ids, math.ceil(len(driver_ids) * train_rate))
    val_driver_ids = driver_ids.difference(train_driver_ids)
    if val_driver_count > 0:
        val_driver_ids = val_driver_ids[:val_driver_count]

    # 训练集
    X_train_names = []
    y_train = []
    for train_id in train_driver_ids:
        X, y = get_transfer_learning_sample_by_driverID(train_id)
        X_train_names.extend(X)
        y_train.extend(y)

    X_val_names = []
    y_val = []
    for val_id in val_driver_ids:
        X, y = get_transfer_learning_sample_by_driverID(val_id)
        X_val_names.extend(X)
        y_val.extend(y)
    return X_train_names, y_train, X_val_names, y_val

def splice_img_left_right(img_names, labels, target_img_shape, same_class_same_driver=False):
    """
    依据图片名称列表及对应的标记，将指定目录中的图片进行拼接；拼接后按照图片预处理函数变换图像数组元素，对标记进行one-hot编码；
    拼接后的图像数组、标记数组元素数量是原相应数组的2倍。

    Args:
        img_names: list, 图片名称列表
        labels: list, 图片分类标记
        target_img_shape: (int, int, int), 图像数组shape:(height, width, channel)
        same_class_same_driver: bool,为真选择同一类别同一驾驶者图像进行拼接，为假选择同一类别图像进行拼接，默认为假
    Returns:
        numpy.array, numpy.array: 图像数组(4D)，标记数组(2D)
    """
    X_enlarge = np.empty((len(img_names) * 2, *target_img_shape), dtype=np.float32)
    y_enlarge_labels = list(range(len(img_names) * 2))
    csv_data = read_csv_file()
    i = 0
    for left_img_name in tqdm(img_names, desc='Splicing image', unit='files'):
        driver_id = csv_data[csv_data['classname_imgname'] == left_img_name]['subject']
        driver_id = driver_id.iloc[0]
        if same_class_same_driver:
            right_img_names = list(csv_data[(csv_data['subject'] == driver_id) & (csv_data['classname'] == labels[i])]['classname_imgname'])
        else:
            right_img_names = list(csv_data[csv_data['classname'] == labels[i]]['classname_imgname'])

        right_img_names.remove(left_img_name)
        right_img_name = random.sample(right_img_names, 1)[0]
        left_img_arr = vip.read_image_file_preprocess(os.path.join(os.getcwd(), DATA_ROOT_DIR, IMG_TRAIN_DIR, left_img_name), target_img_shape, preprocess_fun=xception.preprocess_input)
        right_img_arr = vip.read_image_file_preprocess(os.path.join(os.getcwd(), DATA_ROOT_DIR, IMG_TRAIN_DIR, right_img_name), target_img_shape, preprocess_fun=xception.preprocess_input)

        X_enlarge[2 * i] = left_img_arr
        X_enlarge[2 * i + 1] = vip.splice_two_image_left_right(left_img_arr, right_img_arr)
        y_enlarge_labels[2 * i] = labels[i]
        y_enlarge_labels[2 * i + 1] = labels[i]

        i += 1
    return X_enlarge, vip.label_one_hot_encode(y_enlarge_labels, TOTAL_LABELS)

def splice_img_left_right_and_save(target_img_shape, same_class_same_driver=False):
    """
    左右拼接训练集中的所有图片，并将其存储于拼接图片所在的目录; 同时生成包含拼接图片分类的csv文件

    Args:
        target_img_shape: (int, int, int), 拼接图像shape:(height, width, channel)
        same_class_same_driver: bool, 为真表示拼接的图像来自同一驾驶者同一类别，为假表示拼接的图片来自同一类别
    Returns:
        无
    """
    csv_data = read_csv_file()
    labels = list(csv_data['classname'])
    i = 0
    for left_img_name in tqdm(csv_data['classname_imgname'], desc='Splicing and saving image', unit='files'):
        driver_id = csv_data[csv_data['classname_imgname'] == left_img_name]['subject']
        driver_id = driver_id.iloc[0]
        if same_class_same_driver:
            right_img_names = list(csv_data[(csv_data['subject'] == driver_id) & (csv_data['classname'] == labels[i])]['classname_imgname'])
        else:
            right_img_names = list(csv_data[csv_data['classname'] == labels[i]]['classname_imgname'])
        right_img_names.remove(left_img_name)
        right_img_name = random.sample(right_img_names, 1)[0]

        left_img_arr = cv.imread(os.path.join(os.getcwd(), DATA_ROOT_DIR, IMG_TRAIN_DIR, left_img_name), flags=cv.IMREAD_COLOR)
        left_img_arr = cv.resize(left_img_arr, (target_img_shape[0], target_img_shape[1]))
        right_img_arr = cv.imread(os.path.join(os.getcwd(), DATA_ROOT_DIR, IMG_TRAIN_DIR, right_img_name), flags=cv.IMREAD_COLOR)
        right_img_arr = cv.resize(right_img_arr, (target_img_shape[0], target_img_shape[1]))
        spliced_img_arr = vip.splice_two_image_left_right(left_img_arr, right_img_arr)

        spliced_img_name = left_img_name.rsplit('.')
        spliced_img_name = fou.get_file_name(spliced_img_name[0])
        spliced_img_name = spliced_img_name + '_s.jpg'
        cv.imwrite(os.path.join(os.getcwd(), DATA_ROOT_DIR, IMG_TRAIN_DIR, labels[i], spliced_img_name), spliced_img_arr)

        new_row = pd.DataFrame(csv_data.iloc[i]).T
        new_row['img'] = spliced_img_name
        new_row['classname_imgname'] = os.path.join(labels[i], spliced_img_name)
        csv_data = pd.concat([csv_data,new_row], ignore_index=True)

        i += 1
    csv_n = CSV_NAME.split('.')[0] + '_s.csv'
    csv_data.to_csv(os.path.join(os.getcwd(), DATA_ROOT_DIR, csv_n), index=False)

def save_image_by_driverID(target_path, driverID):
    """
    将指定驾驶者(ID)的所有类别图片另存至指定目录

    Args:
        path: str, 另存图片的目录
        driverID: str, 驾驶者ID
    :return: 无
    """
    csv_data = read_csv_file()
    csv_data_driverIDs = csv_data.groupby('subject').get_group(driverID)
    fou.mkdir(target_path)
    for class_name in tqdm(csv_data_driverIDs['classname_imgname'], desc='Copying image', unit='files'):
        new_name = class_name
        new_name = new_name.replace('\\','_')
        target_file = os.path.join(target_path, new_name)
        source_file = os.path.join(os.getcwd(), DATA_ROOT_DIR, IMG_TRAIN_DIR, class_name)
        open(target_file, "wb").write(open(source_file, "rb").read())
# save_image_by_driverID('G:/test', 'p081')


def load_img_data_from_file(file_dir, file_prefix, total_count, batch_size, img_shape, is_train_data=True):
    """
    从数据文件中载入图像数据

    Args:
        file_dir: str, 数据文件所在目录
        file_prefix: str, 数据文件前缀
        total_count: int, 载入数据总条数，该值应不大于存储为数据文件时的样本总条数
        batch_size: int, 载入时的每批图像数量
        img_shape: (int, int, int), 图像shape:(width, height, channel)
        is_train_data: bool, 训练集/测试集标识，默认为True(训练集)，该值取False表示测试集
    Returns:
        numpy.array, numpy.array: 训练集:样本特征, 样本标记; 测试集:样本特征,图像名称列表
    """
    X = np.empty((total_count, *img_shape), dtype=np.float32)
    if is_train_data:
        y = np.empty((total_count, len(TOTAL_LABELS)), dtype=np.int8)
    file_count = fou.get_prefix_file_count(os.path.join(DATA_ROOT_DIR, DATA_PREPROCESS, file_dir), file_prefix)
    i = 0
    for batch_id in tqdm(range(file_count),desc='Loading data in {}'.format(file_dir), unit='files'):
        for X_b, y_b in vip.img_load_features_labels_in_batch(os.path.join(DATA_ROOT_DIR, DATA_PREPROCESS, file_dir), file_prefix, batch_id, batch_size):
            # start_i = i * batch_size
            if i is 0:
                pos = 0
            if is_train_data:
                for j in range(len(y_b)):
                    if pos >= total_count:
                        break
                    X[pos] = X_b[j]
                    y[pos] = y_b[j]
                    pos = pos + 1
            else:
                if i is 0:
                    y = list(y_b)
                else:
                    y.extend(list(y_b))
                for j in range(len(X_b)):
                    if pos >= total_count:
                        for k in range(len(y_b) - j):
                            y.pop()
                        break
                    X[pos] = X_b[j]
                    pos = pos + 1
            i += 1
    return X, y


def model_predict_img_class(model, test_img_dir, target_img_shape, load_img_batch_size, predict_batch_size, gpus, verbose=1):
    """
    分批读入指定目录中的图片，利用分类模型预测图片分类，返回分类结果

    Args:
        model: class 'keras.engine.training.Model',模型实例
        test_img_dir: str, 测试图片所在目录
        target_img_shape: (int, int, int), 图像shape:(height, width, channel)
        load_img_batch_size: int, 载入测试图片时，每批数量
        predict_batch_size: int, 使用模型预测时，每批数量
        gpus: int, gpu数量
        verbose: int, 日志显示模式, 0或1
    Returns:
        list,list; 预测分类概率(2D数据), 图像名称列表(1D数据)
    """
    img_name_list = os.listdir(os.path.join(os.getcwd(), test_img_dir))
    predictions = []
    image_count = len(img_name_list)
    parallel_model = multi_gpu_model(model, gpus=gpus)
    for batch_i in np.arange(0, image_count, load_img_batch_size):
        start = batch_i
        end = min(batch_i + load_img_batch_size, image_count)
        sub_name_list = img_name_list[start:end]
        X_b = vip.img_read_to_array(os.path.join(os.getcwd(), test_img_dir), sub_name_list, target_img_shape, xception.preprocess_input)
        predictions.extend(parallel_model.predict(X_b, batch_size=predict_batch_size, verbose=verbose))
    return predictions, img_name_list

def load_multi_model_predict_img(model_dir, model_prefix, test_img_dir, target_img_shape, load_img_batch_size, predict_batch_size, model_name_list=None, gpus=4, verbose=1):
    """
    载入已保存的多个模型，分批读入指定目录中的图片，预测图片分类，返回平均分类结果

    Args:
        model_path_name: str, 模型路径名称
        test_img_dir: str, 测试图片所在目录
        target_img_shape: (int, int, int), 图像shape:(height, width, channel)
        load_img_batch_size: int, 载入测试图片时，每批数量
        predict_batch_size: int, 使用模型预测时，每批数量
        model_name_list: list, 待载gpus: int, gpu数量入的模型列表，默认为None，当该实参不为None时，model_prefix参数失效
        gpus: int, gpu数量
        verbose: int, 日志显示模式, 0或1
    Returns:
        list,list; 预测分类概率(2D数据), 图像名称列表(1D数据)
    """
    if model_name_list is None:
        model_name_list = fou.get_prefix_file_names(model_dir, model_prefix)
    predictions_total = []
    for model_name in model_name_list:
        print('Using model {}'.format(model_name))
        model = load_model(os.path.join(model_dir, model_name))
        predictions, img_name_list = model_predict_img_class(model, test_img_dir, target_img_shape, load_img_batch_size, predict_batch_size, gpus=gpus, verbose=verbose)
        predictions_total.append(predictions)
    return predictions_mean(predictions_total), img_name_list

def predictions_mean(predictions_total):
    """
    计算多个预测结果列表的均值

    Args:
        predictions_total: list, 预测结果列表
    Returns:
        list: 结果列表
    """
    num_predictions = len(predictions_total)
    results = np.array(predictions_total[0])
    for i in range(1, num_predictions):
        results += np.array(predictions_total[i])
    results = results / float(num_predictions)
    return results.tolist()

def mkdir_results(dir_prefix):
    """
    创建存放结果的目录

    Args:
        dir_prefix: str, 目录前缀
    Returns:
        str, str: 结果存放目录完整路径, 结果存放目录名称
    """
    dt = gu.get_current_datetime('%Y%m%d_%H%M%S')
    res_dir = '{}_{}'.format(dir_prefix, dt)
    return fou.mkdir(os.path.join(RESULTS_DIR, res_dir)), res_dir

def save_predictions_results(path_name, predictions, img_names, columns=TOTAL_LABELS, img_col_name='img'):
    """
    保存预测结果，生成csv文件

    Args:
        path_name : str, csv路径名称
        predictions:list, 预测结果(2D)
        img_names:list, 图片名称列表(1D)
    Returns:
        无
    """
    df = pd.DataFrame(predictions, columns=columns)
    df.insert(0, img_col_name, pd.Series(img_names, index=df.index))
    df.to_csv(path_name, index=False)

def sorted_unique_list(list_data, reverse=False):
    """
    将列表、元组等1维数据转换为无重复元素的有序列表，默认按照升序排列；排序去重时不改变原列表值

    Args:
        list_data: list, 列表
        reverse: bool, 默认为假，按照升序排列，为真按照降序排列
    Returns:
        list: 无重复元素的有序列表
    """
    unique_list = list(set(list_data))
    return sorted(unique_list, reverse=reverse)


