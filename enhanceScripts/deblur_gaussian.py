import numpy as np
import cv2

class Sharpener:
    def __init__(self, method='unsharp', sigma=1.0, amount=1.5):
        """
        图像锐化/去模糊处理器
        :param method: 'unsharp' (非锐化掩模, 推荐) 或 'laplacian' (拉普拉斯)
        :param sigma: 高斯模糊半径 (unsharp方法用), 越大去模糊越强但噪点越多
        :param amount: 锐化强度 (1.0-3.0), 越大边缘越锐利
        """
        self.method = method
        self.sigma = sigma
        self.amount = amount

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        执行锐化处理
        :param image: 输入图像 (H, W, C) RGB, uint8
        :return: 处理后的图像 (H, W, C) RGB, uint8
        """
        if self.method == 'unsharp':
            return unsharp_mask(image, sigma=self.sigma, amount=self.amount)
        elif self.method == 'laplacian':
            return laplacian_sharpen(image, strength=self.amount)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

def unsharp_mask(img: np.ndarray, sigma: float = 1.0, amount: float = 1.5, threshold: int = 0) -> np.ndarray:
    """
    非锐化掩模 (USM) - 工业标准锐化算法
    原理: 原图 - 高斯模糊 = 高频细节, 原图 + amount * 细节 = 锐化图
    """
    if img is None:
        raise ValueError("锐化插件无法读取输入图像")
    
    img = img.astype(np.float64)
    
    # 1. 高斯模糊获取低频信息
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    # 2. 计算高频细节 (原图 - 模糊)
    detail = img - blurred
    
    # 3. 可选: 阈值去噪 (小于threshold的细节视为噪声不加回)
    if threshold > 0:
        mask = np.abs(detail) < threshold
        detail[mask] = 0
    
    # 4. 将细节增强后加回原图
    sharpened = img + amount * detail
    
    # 5. 裁剪到有效范围并转回uint8
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def laplacian_sharpen(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    拉普拉斯锐化 - 快速但可能放大噪声
    strength: 锐化强度 (0.5-2.0)
    """
    if img is None:
        raise ValueError("锐化插件无法读取输入图像")
    
    img = img.astype(np.float64)
    
    # 拉普拉斯核 (8邻域)
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    
    # 分别对每个通道应用卷积
    laplacian = np.zeros_like(img)
    for i in range(3):
        laplacian[:, :, i] = cv2.filter2D(img[:, :, i], cv2.CV_64F, kernel)
    
    # 拉普拉斯锐化: I' = I - strength * Laplacian(I)
    sharpened = img - strength * laplacian
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)

# === 使用示例 ===
if __name__ == "__main__":
    # 示例用法
    sharpener = Sharpener(method='unsharp', sigma=1.2, amount=1.8)
    # image_sharpened = sharpener.process(image_input)
    
    # 或者直接用函数
    # result = unsharp_mask(image, sigma=1.0, amount=1.5)