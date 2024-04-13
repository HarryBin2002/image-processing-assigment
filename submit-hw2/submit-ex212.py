import numpy as np
from skimage import io as io_url
import matplotlib.pyplot as plt
import cv2

def DFT_slow(data):
    N = len(data)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        X[k] = sum(data[n] * np.exp(-2j * np.pi * k * n / N) for n in range(N))
    return X

def DFT_2D(gray_img):
    H, W = gray_img.shape
    # DFT row-wise
    row_fft = np.array([DFT_slow(gray_img[i, :]) for i in range(H)])
    # DFT column-wise on the result of the row-wise DFT
    row_col_fft = np.array([DFT_slow(row_fft[:, j]) for j in range(W)]).T
    return row_fft, row_col_fft

def show_img(origin, row_fft, row_col_fft):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))
    axs[0].imshow(origin, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(np.log(np.abs(np.fft.fftshift(row_fft))), cmap='gray')
    axs[1].set_title('Row-wise FFT')
    axs[1].axis('off')
    axs[2].imshow((np.log(np.abs(np.fft.fftshift(row_col_fft)))), cmap='gray')
    axs[2].set_title('Column-wise FFT')
    axs[2].axis('off')
    plt.show()

if __name__ == '__main__':
    # Example 1: Test 1D DFT
    x = np.random.random(1024)
    print(np.allclose(DFT_slow(x), np.fft.fft(x)))

    # Example 2: Process and show 2D FFT of an image
    img = io_url.imread('https://img2.zergnet.com/2309662_300.jpg')
    gray_img = np.mean(img, -1)
    row_fft, row_col_fft = DFT_2D(gray_img)
    show_img(gray_img, row_fft, row_col_fft)
