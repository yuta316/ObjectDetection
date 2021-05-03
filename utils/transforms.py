from numpy import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class Compose(object):
    """
    今回は画像のみでなくアノテーションも拡大などの前処理が入るため
    特別なComposeクラスを作成。
    """
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img, boxes=None, labels=None):
        for transform in self.transforms:
            img, boxes, labels = transform(img, boxes, labels)
        return img, boxes, labels

class FromIntToFloat(object):
    """
    画像のint形式からfloatに変更するためのクラス
    """
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

class ToAbsolute(object):
    """
    アノテーションデータの規格化を解除
    """
    def __call__(self, img, boxes=None, labels=None):
        h, w, c = img.shape
        boxes[:, 0] *= w
        boxes[:, 2] *= w
        boxes[:, 1] *= h
        boxes[:, 3] *= h

        return img, boxes, labels

class RandomCrop(object):
    """
    画像からランダムに矩形を抜き出す
    """
        
    def __call__(self, img, boxes=None, labels=None):
        height ,width,_ = img.shape
            
        for _ in range(50):
            current_img = img
            w = random.uniform(0.3*width, width)
            h = random.uniform(0.3*height, height)
            if h/w < 0.5 or h/w >2:
                continue
            print(width-w)
            left = random.uniform(width-w)
            top = random.uniform(height-h)
            rect = np.array([int(left), int(top), int(left+w), int(top+h)])
            current_img = current_img[rect[1]:rect[3], rect[0]:rect[2],:]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            #マスク作成
            m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
            m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
            mask = m1 * m2
            if not mask.any():
                continue

            current_boxes = boxes[mask, :].copy()
            current_labels = labels[mask]

            current_boxes[:, :2] = np.maximum(current_boxes[:, :2],rect[:2])
            current_boxes[:, :2] -= rect[:2]

            current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],rect[2:])
            current_boxes[:, 2:] -= rect[:2]

            return current_img, current_boxes, current_labels

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, boxes=None, labels=None):
        height ,width,_ = img.shape
        length = self.size//2
        left = width//2-length
        top = height//2-length
        rect = np.array([int(left), int(top), int(left+self.size), int(top+self.size)])

        img = img[rect[1]:rect[3], rect[0]:rect[2],:]

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        #マスク作成
        m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
        m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
        mask = m1 * m2

        current_boxes = boxes[mask, :].copy()
        current_labels = labels[mask]

        current_boxes[:, :2] = np.maximum(current_boxes[:, :2],rect[:2])
        current_boxes[:, :2] -= rect[:2]

        current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],rect[2:])
        current_boxes[:, 2:] -= rect[:2]

        return img, current_boxes, current_labels

class RandomMirror(object):
    def __call__(self, img, boxes, classes):
        _, w, _ = img.shape
        if random.randint(2):
            img = img[:,::-1]
            boxes = boxes.copy()
            boxes[:,0::2] = width - boxes[:,2::-2]

        return img, boxes, classes

class ToPercent(object):
    def __call__(self, img, boxes=None, labels=None):
        h,w,c = img.shape
        boxes[:, 0] /= w
        boxes[:, 2] /= w
        boxes[:, 1] /= h
        boxes[:, 3] /= h
        return img, boxes, labels

class Resize(object):
    def __init__(self, size=224):
        self.size = size
    def __call__(self, img, boxes=None, labels=None):
        img = cv2.resize(img, (self.size, self.size))

        return img, boxes, labels

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)
    
    def __call__(self, img, boxes=None, labels=None):
        img = img.astype(np.float32)
        img -= self.mean
        return img.astype(np.float32), boxes, labels
