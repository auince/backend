
import numpy as np
import cv2
import numpy as np

class Dehazer:
    def __init__(self):
        # 初始化模型或参数
        pass

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        执行去雾处理
        :param image: 输入图像 (H, W, C) RGB
        :return: 处理后的图像 (H, W, C) RGB
        """
        image_dehazed = dehaze(image)
        return image_dehazed

def dark_channel(img, kernel_size=5):
    """计算暗通道"""
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dark = cv2.erode(min_channel, kernel)
    return dark

def estimate_atmospheric_light(img, dark, top_percent=0.001):
    """估计大气光 A"""
    h, w = dark.shape
    num_pixels = h * w
    num_top = max(int(num_pixels * top_percent), 1)

    dark_vec = dark.reshape(num_pixels)
    img_vec = img.reshape(num_pixels, 3)

    indices = np.argsort(dark_vec)[-num_top:]
    A = np.mean(img_vec[indices], axis=0)
    return A

def estimate_transmission(img, A, kernel_size=15, omega=0.9):
    """估计透射率"""
    norm_img = img / A
    dark_norm = dark_channel(norm_img, kernel_size)
    transmission = 1 - omega * dark_norm
    return transmission

def recover_image(img, transmission, A, t0=0.05):
    """恢复去雾图像"""
    transmission = np.maximum(transmission, t0)
    J = (img - A) / transmission[:, :, np.newaxis] + A
    return np.clip(J, 0, 255)

def dehaze(image):
    img = image
    if img is None:
        raise ValueError("去雾插件无法读取输入图像")

    img = img.astype(np.float64)

    dark = dark_channel(img)
    A = estimate_atmospheric_light(img, dark)
    transmission = estimate_transmission(img, A)
    result = recover_image(img, transmission, A)

    result = result.astype(np.uint8)
    return result
