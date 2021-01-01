# Google utils: https://cloud.google.com/storage/docs/reference/libraries
# Google Cloud相关函数，但是现在把Google Cloud的下载地址改成GitHub了

import os
import platform
import subprocess
import time
from pathlib import Path

import torch


def gsutil_getsize(url=''):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output('gsutil du %s' % url, shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # bytes


def attempt_download(weights):
    '''尝试下载单个权重文件。
    参数：
        weights: 权重文件的本地路径
    '''
    # Attempt to download pretrained weights if not found locally
    # 转化为字符串，去掉两边空白和单引号
    weights = str(weights).strip().replace("'", '')
    # 获取文件名的小写
    file = Path(weights).name.lower()

    # 提示信息
    msg = weights + ' missing, try downloading from https://github.com/ultralytics/yolov3/releases/'
    # 可下载的模型
    models = ['yolov3.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt']  # available models
    # 是否有其他下载地址
    redundant = False  # offer second download option

    # 如果尝试下载的模型可用，并且在本地不存在
    if file in models and not os.path.isfile(weights):
        # Google Drive
        # d = {'yolov5s.pt': '1R5T6rIyy3lLwgFXNms8whc-387H0tMQO',
        #      'yolov5m.pt': '1vobuEExpWQVpXExsJ2w-Mbf3HJjWkQJr',
        #      'yolov5l.pt': '1hrlqD1Wdei7UT4OgT785BEk1JwnSvNEV',
        #      'yolov5x.pt': '1mM8aZJlWTxOg7BZJvNUMrTnA2AbeCVzS'}
        # r = gdrive_download(id=d[file], name=weights) if file in d else 1
        # if r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6:  # check
        #    return

        # 从GitHub下载
        try:  # GitHub
            url = 'https://github.com/ultralytics/yolov3/releases/download/v1.0/' + file
            print('Downloading %s to %s...' % (url, weights))
            # 从GitHub上加载一个带有预训练权重的模型
            torch.hub.download_url_to_file(url, weights)
            # 下载完检查文件是否存在
            assert os.path.exists(weights) and os.path.getsize(weights) > 1E6  # check
        except Exception as e:  # GCP
            # 下载错误
            print('Download error: %s' % e)
            assert redundant, 'No secondary mirror'
            url = 'https://storage.googleapis.com/ultralytics/yolov3/ckpt/' + file
            print('Downloading %s to %s...' % (url, weights))
            r = os.system('curl -L %s -o %s' % (url, weights))  # torch.hub.download_url_to_file(url, weights)
        finally:
            # 下载失败
            if not (os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # check
                # 清除残余文件
                os.remove(weights) if os.path.exists(weights) else None  # remove partial downloads
                print('ERROR: Download failure: %s' % msg)
            print('')
            return


def gdrive_download(id='1n_oKgR81BJtqk75b00eAjdv03qVCQn2f', name='coco128.zip'):
    # Downloads a file from Google Drive. from utils.google_utils import *; gdrive_download()
    t = time.time()

    print('Downloading https://drive.google.com/uc?export=download&id=%s as %s... ' % (id, name), end='')
    os.remove(name) if os.path.exists(name) else None  # remove existing
    os.remove('cookie') if os.path.exists('cookie') else None

    # Attempt file download
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system('curl -c ./cookie -s -L "drive.google.com/uc?export=download&id=%s" > %s ' % (id, out))
    if os.path.exists('cookie'):  # large file
        s = 'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm=%s&id=%s" -o %s' % (get_token(), id, name)
    else:  # small file
        s = 'curl -s -L -o %s "drive.google.com/uc?export=download&id=%s"' % (name, id)
    r = os.system(s)  # execute, capture return
    os.remove('cookie') if os.path.exists('cookie') else None

    # Error check
    if r != 0:
        os.remove(name) if os.path.exists(name) else None  # remove partial
        print('Download error ')  # raise Exception('Download error')
        return r

    # Unzip if archive
    if name.endswith('.zip'):
        print('unzipping... ', end='')
        os.system('unzip -q %s' % name)  # unzip
        os.remove(name)  # remove zip to free space

    print('Done (%.1fs)' % (time.time() - t))
    return r


def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""

# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     # Uploads a file to a bucket
#     # https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
#
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#
#     blob.upload_from_filename(source_file_name)
#
#     print('File {} uploaded to {}.'.format(
#         source_file_name,
#         destination_blob_name))
#
#
# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     # Uploads a blob from a bucket
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#
#     blob.download_to_filename(destination_file_name)
#
#     print('Blob {} downloaded to {}.'.format(
#         source_blob_name,
#         destination_file_name))
