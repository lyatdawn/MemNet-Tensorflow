# -*- coding:utf-8 -*-
"""
将VOC2007和VOC2012图像数据集, 利用cv2.imwrite()函数保存图像时, quality factor为10, 20.
"""
import os
import cv2
import glob

if __name__ == '__main__':
    data_dir = "/home/ly/caffe-ssd/data/VOC0712"
    save_dir = "/home/ly/git/MemNet/datasets/VOC0712"

    for filename in ["VOC2007", "VOC2012"]:
        image_path = glob.glob(os.path.join(data_dir, filename, "JPEGImages/*.jpg"))
        # print(image_path[0].split("/")[8]) # Get the name of images.

        for i in range(len(image_path)):
            image = cv2.imread(image_path[i], 0) # Gray image

            save_path = os.path.join(save_dir, filename)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_image_path = os.path.join(save_path, image_path[i].split("/")[8])
            # cv2.imwrite(save_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 10]) # quality 10. 
            # default is 95.
            cv2.imwrite(save_image_path, image)
