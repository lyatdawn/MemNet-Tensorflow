# -*- coding:utf-8 -*-
"""
Compute PSNR and SSIM with Set12.
"""
import os
import glob
import cv2
import numpy as np
from skimage.measure import compare_psnr, compare_ssim

if __name__ == '__main__':
    data_set12 = glob.glob(os.path.join("../datasets/Set12", "*.jpg"))
    data_set2_quality10 = glob.glob(os.path.join("../datasets/Set12_Quality10", "*.jpg"))
    data_set12_recovery = glob.glob(os.path.join("../datasets/Set12_Recovery", "*.jpg"))

    compress_avg_psnr = 0.
    deblocking_avg_psnr = 0.
    compress_avg_ssim = 0.
    deblocking_avg_ssim = 0.
    for i in range(len(data_set12)):
        # reszie 256 * 256.
        img_set12 = cv2.resize(cv2.imread(str(data_set12[i]), 0), (256, 256))
        img_set12_q10 = cv2.resize(cv2.imread(str(data_set2_quality10[i]), 0), (256, 256))
        img_set12_recovery = cv2.resize(cv2.imread(str(data_set12_recovery[i]), 0), (256, 256))

        # label, noisy_image
        psnr_compress = compare_psnr(img_set12, img_set12_q10, data_range=255)
        print(psnr_compress)
        compress_avg_psnr += psnr_compress
        psnr_deblocking = compare_psnr(img_set12, img_set12_recovery, data_range=255)
        print(psnr_deblocking)
        deblocking_avg_psnr += psnr_deblocking

        ssim_compress = compare_ssim(img_set12, img_set12_q10)
        print(ssim_compress)
        compress_avg_ssim += ssim_compress
        ssim_deblocking = compare_ssim(img_set12, img_set12_recovery)
        print(ssim_deblocking)
        deblocking_avg_ssim += ssim_deblocking

    print("Average compress PSNR is: {}".format(compress_avg_psnr / len(data_set12)))
    print("Average compress SSIM is: {}".format(compress_avg_ssim / len(data_set12)))

    print("Average deblocking PSNR is: {}".format(deblocking_avg_psnr / len(data_set12)))
    print("Average deblocking SSIM is: {}".format(deblocking_avg_ssim / len(data_set12)))