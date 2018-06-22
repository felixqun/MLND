#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @license : Copyright(C)2018, Yu Q. Studio, All rights reserved. 
# @Author  : Yu Q.
# @Time    : 2018/4/17 10:57
# @Software: PyCharm
"""
    文件操作的工具模块，用到urllib模块、requests包，包括：
    下载数据包并用进度条显示下载进度，解压数据包到指定目录
"""

import urllib
import requests
import os
import zipfile
from tqdm import tqdm

class DLProgress(tqdm):
    """
    进度条类
    """
    last_block = 0
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def download_login_big_data_file(url, username, password, download_dir, download_name='', chunk_size=1024*1024):
    """
        使用requests下载大文件，下载文件时需要先登录，保存文件到指定目录

    Args:
        url: str, 下载链接url
        username: str, 登录用户名
        password: str, 登录密码
        download_dir: str, 保存目录
        download_name: str, 保存文件名称
        chunk_size: int, 下载时每次保存的缓存块大小
    Returns:
        int: 下载文件大小
    """
    response = requests.get(url)
    login_info = {'username': username, 'password': password}
    res_data = requests.post(response.url, data=login_info, stream=True)

    total_size = int(res_data.headers['Content-Length'])
    if total_size < 1:
        return total_size

    if len(download_name) < 1:
        download_name = get_file_name(url)
    path_name = os.path.join(mkdir(download_dir), download_name)
    with DLProgress(unit='B', unit_scale=True, desc='Downloading {}'.format(download_name)) as pbar:
        chunk_num = 0
        with open(path_name, 'wb') as file:
            for chunk in res_data.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    # file.flush()
                chunk_num += 1
                pbar.hook(block_num=chunk_num, block_size=len(chunk), total_size=total_size)
            # file.close()
        pbar.close()
    return total_size



def mkdir(dir_path):
    """
    若目录不存在，创建多层目录。dir_path可以是绝对路径，如"c:/d/e"，也可以是相对路径，如"./a"、"../a/b"等，若为"/a"形式，则在主调py文件所在的根目录创建目录。

    Args:
        dir_path: str, 待创建目录
    Returns:
        str: 目录绝对路径
    """
    dir_path = dir_path.strip()
    dir_path = dir_path.rstrip("\\")
    dir_path = dir_path.rstrip("/")
    dir_path = os.path.join(os.getcwd(), dir_path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path

def download_samll_data_file(url, download_dir, desc="download data file"):
    """
    下载小文件，下载文件时无需登录，保存文件到指定目录

    Args:
        url: str, 数据包所在的url
        download_dir: str, 保存目录
    Returns:
        无
    """
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc=desc) as pbar:
        # request.request_host()
        urllib.request.urlretrieve(
            url,
            download_dir,
            pbar.hook)

def un_zip(file_name, des_dir='', zip_file_is_dir=False):
    """
    解压zip文件，解压到指定目录。压缩包的文件中若有中文名称，解压后中文名称会乱码，有待改进。

    Args:
        file_name: str, 待解压文件名称(含目录)
        des_dir: str, 解压目标目录
        zip_file_is_dir: boolean, 用于指定是否将压缩包解压至目录(目录名称为压缩文件名称)中，默认为False，不解压至目录
    Returns:
        无
    """
    file_dir = get_file_name(file_name)
    file_dir = file_dir.split('.')[0]

    if zip_file_is_dir:
        if len(des_dir) > 0:
            des_dir += '/' + file_dir
        else:
            des_dir = file_dir
    with zipfile.ZipFile(file_name) as zipf:
        for sig_name in tqdm(zipf.namelist(), desc='Extracting {}'.format(get_file_name(file_name)), unit='files'):
            zipf.extract(sig_name, path=mkdir(des_dir))
        zipf.close()

def get_file_name(path_name):
    """
    从含路径的字符串中获取文件名称

    Args:
        path_name: str, 含文件名称路径字符串
    Returns:
        str: 文件名称
    """
    list_one = path_name.split('/')[-1]
    list_two = path_name.split('\\')[-1]
    if len(list_one) < len(list_two):
        file_name = list_one
    else:
        file_name = list_two
    return file_name

def get_prefix_file_count(dir, prefix):
    """
    获取目录中指定前缀的文件数量

    Args:
        dir: str, 文件所在目录
        prefix: str, 文件前缀
    Returns:
        int: 特定前缀的文件数量
    """
    fname_list = os.listdir(os.path.join(os.getcwd(), dir))
    count = 0
    for name in fname_list:
        if name.find(prefix) is 0:
            count += 1
    return count

def get_prefix_file_names(dir, prefix):
    """
    获取目录中指定前缀的文件名称列表

    Args:
        dir: str, 文件所在目录
        prefix: str, 文件前缀
    Returns:
        list: 特定前缀的文件名称列表
    """
    fname_list = os.listdir(os.path.join(os.getcwd(), dir))
    f_names = []
    for name in fname_list:
        if name.find(prefix) is 0:
            f_names.append(name)
    return f_names



