# Farfetech case study
#
# This script resizes the images to specified size.
#
# Author: Kai Chen
# Date: Apr, 2018
#

import cv2
import numpy as np
import os, sys

dir_path = "data/images/"
dirs = os.listdir(dir_path)

#img_width, img_height = 100, 100
img_width, img_height = 80, 80
dest_dir_path = "data/images_{}_{}/".format(img_width, img_height)
if not os.path.exists(dest_dir_path):
    os.makedirs(dest_dir_path)


for file_name in dirs:
   file_path = os.path.join(dir_path, file_name)
   # print(file_path)
   # The idea is taken from
   # https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
   img = cv2.imread(file_path)
   res = cv2.resize(img, dsize=(img_width, img_height), interpolation=cv2.INTER_CUBIC)
   cv2.imwrite("{}{}".format(dest_dir_path, file_name), res)


