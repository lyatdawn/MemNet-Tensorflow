# -*- coding:utf-8 -*-
import os
import cv2
import glob

if __name__ == '__main__':
    data_dir = "/home/ly/DATASETS/image_denoising/test_sets/Set12"

    save_path = "../datasets/Set12_Quality10"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for image_file in glob.glob(os.path.join(data_dir, "*.png")):
        image = cv2.imread(image_file, 0) # Gray image

        save_image_path = os.path.join(save_path, image_file.split("/")[7][0:2] + ".jpg") # 01.jpg
        cv2.imwrite(save_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 10]) # quality 10. 
        # default is 95.
        # cv2.imwrite(save_image_path, image)