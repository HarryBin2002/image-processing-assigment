import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os

def read_img(img_path):
    img = cv2.imread(img_path, 0)
    if img is None:
        raise ValueError(f"Image at {img_path} could not be loaded. Check the file path.")
    return img


def padding_img(img, filter_size=3):
    pad_size = filter_size // 2
    padded_img = np.pad(img, pad_size, mode='edge')
    return padded_img


def mean_filter(img, filter_size=3):
    padded_img = padding_img(img, filter_size)
    smoothed_img = np.zeros_like(img)
    pad_size = filter_size // 2

    for i in range(pad_size, padded_img.shape[0] - pad_size):
        for j in range(pad_size, padded_img.shape[1] - pad_size):
            smoothed_img[i - pad_size, j - pad_size] = np.mean(
                padded_img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1])

    return smoothed_img


def median_filter(img, filter_size=3):
    padded_img = padding_img(img, filter_size)
    smoothed_img = np.zeros_like(img)
    pad_size = filter_size // 2

    for i in range(pad_size, padded_img.shape[0] - pad_size):
        for j in range(pad_size, padded_img.shape[1] - pad_size):
            smoothed_img[i - pad_size, j - pad_size] = np.median(
                padded_img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1])

    return smoothed_img


def psnr(gt_img, smooth_img):
    mse = np.mean((gt_img - smooth_img) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_score = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr_score


def show_res(before_img, after_img):
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.show()


if __name__ == '__main__':
    img_noise = "noise.png"  # <-- Specify the correct path
    img_gt = "ori_img.png"  # <-- Specify the correct path
    try:
        img = read_img(img_noise)
    except ValueError as e:
        print(e)
        exit()

    filter_size = 3

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    show_res(img, mean_smoothed_img)
    print('PSNR score of mean filter: ', psnr(read_img(img_gt), mean_smoothed_img))

    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img)
    print('PSNR score of median filter: ', psnr(read_img(img_gt), median_smoothed_img))
