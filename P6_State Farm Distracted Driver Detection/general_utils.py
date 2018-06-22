#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @license : Copyright(C)2018, Yu Q. Studio, All rights reserved. 
# @Author  : Yu Q.
# @Time    : 2018/1/4 21:21
# @Software: PyCharm
# @Version : 1.1
"""
    通用工具模块，包括：
    打印、获取系统当前时间、获取变量类型、获取变量名称
"""
import time

def p(obj_name,obj=""):
    """
    调试过程中，打印显示变量名称及变量值 
    
    Args:
        obj_obj_name: str, 待打印对象名称
        obj: obj, 待打印对象
    Returns: 
    无
    """
    if isinstance(obj, (str, )) and len(obj) < 1:
        print("{0}".format(obj_name))
    else:
        print("{0}:{1}".format(obj_name, obj))


def get_type(obj, print_type=True):
    """
    获取变量的类型，需要时打印 
    
    Args:
        obj: obj, 任意数据类型
        print_type: bool, 是否需要打印显示该数据类型，默认为True(需要)
    Returns: 
        str: 数据类型描述
    """
    obj_type = type(obj)
    if print_type:
        print("Type is : {}".format(obj_type))

    return obj_type


def get_name_of_obj(globals, obj, except_word=""):
    """
    利用globals()内置函数，获取对象的名称；
    globals() 返回一个模块命名空间的字典，因此只能获取本模块的对象名称，
    获取其他模块(如主调模块)的对象名称时，需传入当前模块的globals()函数返回值

    Args:
        globals: globals()函数返回值
        obj: obj, 对象名称
        except_word: str, 不获取名称的词
    Returns:
        str: 对象名称
    """
    for k, v in list(globals.items()):
        if id(v) == id(obj):
            return k


def get_current_datetime(format="%Y-%m-%d %H:%M:%S"):
    """
    获取本地时间

    Args:
        format: str, 时间输出格式，默认为"%Y-%m-%d %H:%M:%S"格式
    Returns:
        str: 当前时间
    """
    return time.strftime(format, time.localtime())

def get_time_stamp_millisecond(ct=time.time(),format="%Y-%m-%d %H:%M:%S"):
    """
    获取包含3位毫秒的当前时间，或者将指定的时间戳float(小数点前有10位数字，形如1464650070.319240)转换为
    字符串显示的格式，形如“%Y-%m-%d %H:%M:%S.millisecond”

    Args:
        ct: float, 当前时间，或者是输入的时间
        format: str, 时间戳字符串的格式
    Returns:
        str: 指定格式的时间戳字符串
    """
    local_time = time.localtime(ct)
    data_head = time.strftime(format, local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    return time_stamp

def get_during_time(start_timestamp):
    """
    获取历史时间戳与调用本函数时的间隔时间

    Args:
        start_timestamp: float, 开始时间
    Returns:
        float: 间隔时间(s)
    """
    return time.time() - start_timestamp


