#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from lxml.etree import Element, SubElement, tostring, ElementTree

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

from PIL import Image

classes = ["Block", "RegionLeft", "RegionRight", "RegionUp", "RegionDown", "RegionIcon", "RegionTitle",
            "Table", "List", "Tree", "EditText", "Spinner", "ComboBox", "CheckBox", "RadioButton", "Slider", "ScrollBar", "Icon", "Switch"]  # 类别

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open('./label_xml/%s.xml' % (image_id), encoding='UTF-8')
    img_fp = ('./images/%s.png' % (image_id))
    out_file = open('./label_txt/%s.txt' % (image_id), 'w')  # 生成txt格式文件
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    if w < 10 or h < 10:
        image = Image.open(img_fp)
        w, h = image.size

    for obj in root.iter('object'):
        cls = obj.find('name').text
        # print(cls)
        if cls not in classes:
            print(str(image_id) + " " + cls)
        else:
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

xml_path = os.path.join(CURRENT_DIR, './label_xml/')
xml_cnt = 0
# xml list
img_xmls = os.listdir(xml_path)
for img_xml in img_xmls:
    label_name = img_xml.split('.')[0]
    xml_cnt = xml_cnt + 1
    print(label_name)
    convert_annotation(label_name)

print(xml_cnt)
