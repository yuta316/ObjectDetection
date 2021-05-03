import os.path as osp
from numpy import random
# XMLをファイルやテキストから読み込んだり、加工したり、保存したりするためのライブラリ
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from ..utils.transforms import *

def make_datapath_list(rootpath):
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')

    train_id_names = osp.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Main/val.txt')

    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
      file_id = line.strip()
      img_path = (imgpath_template % file_id)
      anno_path = (annopath_template % file_id)
      train_img_list.append(img_path)
      train_anno_list.append(anno_path)

    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  
        img_path = (imgpath_template % file_id)  
        anno_path = (annopath_template % file_id) 
        val_img_list.append(img_path) 
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list

class Anno_xml2list(object):
  def __init__(self, classes):
    self.classes = classes
  
  def __call__(self, xml_path, width, height):
    #物体のアノテーション
    ret = []

    xml = ET.parse(xml_path).getroot()
    for obj in xml.iter('object'):
      difficult = int(obj.find('difficult').text)
      if difficult == 1:
        continue
      
      #1物体に関するアノテーション
      bndbox = []
      name = obj.find('name').text.lower().strip()
      bbox = obj.find('bndbox')

      #正規化
      pts = ['xmin', 'ymin', 'xmax', 'ymax']
      for pt in pts:
        cur_pixel = int(bbox.find(pt).text) -1
        if pt == 'xmin' or pt == 'xmax':  
          cur_pixel /= width
        else:
          cur_pixel /= height
        bndbox.append(cur_pixel)
      
      label_index = self.classes.index(name)
      bndbox.append(label_index)
      #resは[xmin, ymin, xmax, ymax, label_ind]
      ret+=[bndbox]
  
    return np.array(ret)

"""
train_transform = Compose([
        FromIntToFloat(),
        ToAbsolute(),
        RandomCrop(),
        RandomMirror(),
        ToPercent(),
        Resize(224),
        SubtractMeans((104, 117, 123) )
])
test_transform = Compose([
        FromIntToFloat(),
        Resize(500),
        ToAbsolute(),                                   
        CenterCrop(400),
        ToPercent(),
        SubtractMeans((104, 117, 123) )                             
])
"""