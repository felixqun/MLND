#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @license : Copyright(C)2018, Yu Q. Studio, All rights reserved. 
# @Author  : Yu Q.
# @Time    : 2018/4/29 17:13
# @File    : model_utils.py
# @Software: PyCharm
"""
    模型操作相关的工具类，包括保存模型、计算交叉熵等
"""

import os
import numpy as np
from keras.models import load_model, model_from_json, model_from_yaml
from sklearn.metrics import log_loss
from keras.utils import multi_gpu_model

import file_operation_utils as fou


def save_model_architecture(path, name, model, str_JSON=True):
    """
    保存模型结构为文件

    Args:
        path: str, 文件目录
        name: str, 文件名称
        model: <class 'keras.models.Sequential'> or <class 'keras.engine.training.Model'>, 模型实例
        str_JSON: bool, 保存格式，取True时以JSON string格式保存，否则以YAML string格式保存
    Returns:
        无
    """
    fou.mkdir(path)
    with open(os.path.join(path, name), 'w') as f:
        if str_JSON:
            str = model.to_json()
        else:
            str = model.to_yaml()
        f.write(str)

def load_model_architecture(path_name, str_JSON=True):
    """
    载入已保存的模型结构

    Args:
        path_name: str, 已保存的模型结构文件名称
        str_JSON: bool, 载入格式，取True时以JSON string格式载入，否则以YAML string格式载入
    Returns:
        <class 'keras.models.Sequential'> or <class 'keras.engine.training.Model'>, 模型实例
    """
    if str_JSON:
        model = model_from_json(open(path_name).read())
    else:
        model = model_from_yaml(open(path_name).read())
    return model

def load_model_architecture_weights(model_arc_name, model_weights_name, str_JSON=True):
    """
    依据模型结构、模型权重载入模型

    Args:
        model_arc_name: str, 模型结构路径名称
        model_weights_name: str, 模型权重路径名称
        str_JSON: bool, 载入格式，取True时以JSON string格式载入，否则以YAML string格式载入
    Returns:
        <class 'keras.models.Sequential'> or <class 'keras.engine.training.Model'>, 模型实例
    """
    # 载入模型结构
    architecture_path_name = os.path.join(os.getcwd(), model_arc_name)
    if str_JSON:
        model = model_from_json(open(architecture_path_name).read())
    else:
        model = model_from_yaml(open(architecture_path_name).read())
    # 载入模型权重
    model.load_weights(os.path.join(os.getcwd(), model_weights_name))
    return model

def get_log_loss_one_model(model_path_name, X_val, y_val, batch_size=200, show_log_loss=True):
    """
    载入已训练的单个模型，在验证集上计算交叉熵

    Args:
        model_path_name: str, 模型路径名称
        X_val: np.array, 验证集特征数组
        y_val: np.array, 验证集标记数组
        batch_size: int, 每批数量
        show_log_loss: bool, 是否打印输出计算得到的log_loss，默认为真，打印输出
    Returns:
         float: log_loss值
    """
    model = load_model(model_path_name)
    parallel_model = multi_gpu_model(model, gpus=8)
    predictions_valid = parallel_model.predict(X_val, batch_size=batch_size, verbose=1)
    log_loss_val = log_loss(y_val, predictions_valid)
    if show_log_loss:
        print('{} log_loss: {}'.format(fou.get_file_name(model_path_name), log_loss_val))
    return log_loss_val

def get_log_loss_multi_model(model_dir, model_prefix, X_val, y_val, batch_size=200, model_name_list=None, show_single_log_loss=False):
    """
    载入已训练的多个模型，在验证集上计算交叉熵的平均值

    Args:
        model_dir: str, 模型所在目录
        model_prefix: str, 模型前缀
        X_val: np.array, 验证集特征数组
        y_val: np.array, 验证集标记数组
        batch_size: int, 每批数量
        model_name_list: list, 待载入的模型列表，默认为None，当该实参不为None时，model_prefix参数失效
        show_single_log_loss: bool, 是否打印输出单个模型的log_loss，默认为假，不打印输出
    Returns:
         float: 各模型log_loss的均值
    """
    if model_name_list is None:
        model_name_list = fou.get_prefix_file_names(model_dir, model_prefix)
    log_loss_values = np.empty(len(model_name_list), dtype=np.float32)
    for i, model_name in enumerate(model_name_list):
        log_loss_values[i] = get_log_loss_one_model(os.path.join(model_dir, model_name), X_val, y_val, batch_size=batch_size, show_log_loss=show_single_log_loss)
    mean_ll = log_loss_values.mean()
    print('Mean log_loss: {}'.format(mean_ll))
    return mean_ll


def tl_model_lock_layers(model, locked_layer_start=0, locked_layer_end=0):
    """
    迁移学习时，锁定层

    Args:
        model: class 'keras.engine.training.Model',迁移学习模型实例
        locked_layer_start: int, 待锁定层的起始层序号(从0开始)，默认为0
        locked_layer_end: int, 待锁定层的终止序号(从0开始)，默认为0：锁定所有层
    Returns:
        class 'keras.engine.training.Model': 锁定层之后的模型实例
    """
    if locked_layer_end < 1:
        locked_layer_end = len(model.layers)
    for layer in model.layers[locked_layer_start:locked_layer_end]:
        layer.trainable = False
    return model

def tl_model_print_param_count(model):
    """
    迁移学习时，锁定层

    Args:
        model: class 'keras.engine.training.Model',迁移学习模型实例
        locked_layer_start: int, 待锁定层的起始层序号(从0开始)，默认为0
        locked_layer_end: int, 待锁定层的终止序号(从0开始)，默认为0：锁定所有层
    Returns:
        class 'keras.engine.training.Model': 锁定层之后的模型实例
    """
    pass

def print_model_layers_and_index(model):
    """
    显示模型的层索引及层名称

    Args:
        model: class 'keras.engine.training.Model',模型实例
    Returns:
        无
    """
    print('Layers count', len(model.layers))
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
