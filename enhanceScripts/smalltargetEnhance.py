import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import time

# ==========================================
# 1. 环境依赖与 NPU 适配
# ==========================================
try:
    import torch_npu
    print("Successfully imported torch_npu.")
except ImportError:
    pass  # 这里的 pass 是为了在非 NPU 环境下也能运行，后续会检测 device

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from torch_npu.contrib import transfer_to_npu
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


# ==========================================
# 2. 模型结构定义 (EDSR)
# ==========================================
class EDSR(nn.Module):
    def __init__(self, n_resblocks, n_feats, scale, rgb_range, n_colors, res_scale, conv=default_conv):
        super(EDSR, self).__init__()

        kernel_size = 3 
        act = nn.ReLU(True)
        
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # Head
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # Body
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # Tail
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x 

# ==========================================
# 3. 核心功能类: SmallTargetEnhancer
# ==========================================
class SmallTargetEnhancer:
    def __init__(self, model_path: str, scale: int = 4, device_str: str = None):
        """
        初始化弱小目标增强器 (基于 EDSR 超分)
        :param model_path: 权重文件路径 (.pt/.pth)
        :param scale: 放大倍数 (默认 4)
        :param device_str: 指定设备 ('npu', 'cuda', 'cpu')，默认自动检测
        """
        # --- 1. 设备初始化 ---
        if device_str:
            self.device = torch.device(device_str)
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            self.device = torch.device("npu:0")
            print(f"[SmallTargetEnhancer] Running on NPU: {torch.npu.get_device_name(0)}")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("[SmallTargetEnhancer] Running on GPU")
        else:
            self.device = torch.device("cpu")
            print("[SmallTargetEnhancer] Running on CPU")

        # --- 2. 模型参数配置 ---
        self.scale = scale
        # 默认 EDSR 参数 (如果使用的是 Baseline 模型，需调整为 n_resblocks=16, n_feats=64)
        self.n_resblocks = 32
        self.n_feats = 256
        self.rgb_range = 255
        self.n_colors = 3
        self.res_scale = 0.1

        # --- 3. 加载模型 ---
        self.model = EDSR(
            n_resblocks=self.n_resblocks,
            n_feats=self.n_feats,
            scale=self.scale,
            rgb_range=self.rgb_range,
            n_colors=self.n_colors,
            res_scale=self.res_scale
        ).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}")

        try:
            print(f"[SmallTargetEnhancer] Loading weights from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 处理 checkpoint 字典结构差异
            state_dict = checkpoint
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            print("[SmallTargetEnhancer] Model loaded successfully.")
        except Exception as e:
            print(f"[SmallTargetEnhancer] Error loading weights: {e}")
            raise e

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        执行弱小目标增强 (切片推理版，防止 OOM)
        :param image: 输入图像 (H, W, 3) RGB, uint8
        :return: 处理后的图像 (H*s, W*s, 3) RGB, uint8
        """
        # ==========================================
        # 参数配置 (根据显存大小调整)
        # ==========================================
        # patch_size: 每次处理的切片大小。降低此值可减少显存占用 (建议 256, 400, 或 512)
        # 910B/GPU 显存紧张时，建议设为 192 或 256
        patch_size = 256 
        padding = 10  # 边缘重叠区域，防止拼接处出现接缝
        scale = self.scale
        
        h, w, c = image.shape
        
        # 初始化输出画布 (H*scale, W*scale, C)
        out_h, out_w = h * scale, w * scale
        output_image = np.zeros((out_h, out_w, c), dtype=np.uint8)

        # ==========================================
        # 切片循环
        # ==========================================
        # 使用 tqdm 显示进度 (可选)
        # from tqdm import tqdm 
        
        # 这里的步长是 patch_size - (2 * padding)，保证中间有效区域无缝连接
        step = patch_size - 2 * padding
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                # 1. 计算当前切片的输入坐标 (考虑 padding)
                y_start = max(0, y - padding)
                x_start = max(0, x - padding)
                y_end = min(h, y + step + padding) # 此时是 loose end，后面会根据 patch_size 截断
                x_end = min(w, x + step + padding)
                
                # 提取 Patch
                patch = image[y_start:y_end, x_start:x_end, :]
                
                # 2. 推理当前 Patch
                # Numpy -> Tensor
                patch_tensor = torch.from_numpy(patch.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # EDSR 不需要除以 255
                    res_tensor = self.model(patch_tensor)
                
                # Tensor -> Numpy
                res_patch = res_tensor.squeeze(0).clamp(0, 255).round().cpu().detach().numpy()
                res_patch = res_patch.transpose(1, 2, 0).astype(np.uint8) # (H_s, W_s, C)
                
                # 3. 计算有效区域并填充回大图
                # 输入图中 patch 相对于 (y, x) 的有效偏移
                input_offset_y = y - y_start
                input_offset_x = x - x_start
                
                # 在输出图中的坐标
                out_y_start = y * scale
                out_x_start = x * scale
                
                # 截取结果中的有效部分 (去除 padding 带来的边缘)
                # 逻辑：如果是第一块/最后一块，保留边界；否则去除 padding
                
                # 有效区域在 Result Patch 内部的起始点
                res_valid_y_start = input_offset_y * scale
                res_valid_x_start = input_offset_x * scale
                
                # 需要填充到 Output Image 的高度和宽度
                # 取决于输入 step 的实际长度
                h_take = min(step, h - y) * scale
                w_take = min(step, w - x) * scale
                
                # 执行填充
                output_image[out_y_start : out_y_start + h_take, 
                             out_x_start : out_x_start + w_take] = \
                    res_patch[res_valid_y_start : res_valid_y_start + h_take,
                              res_valid_x_start : res_valid_x_start + w_take]

                # 显存清理 (关键)
                del patch_tensor, res_tensor
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return output_image
# ==========================================
# 调用示例
# ==========================================
if __name__ == "__main__":
    # 模拟路径
    model_path = "/root/EDSR-PyTorch/experiment/EDSR_x4.pt"
    input_img_path = "/root/EDSR-PyTorch/test/0853x4.png"
    output_img_path = "result_small_target.png"

    # 1. 初始化
    # 如果没有权重文件，此处会抛出异常
    if os.path.exists(model_path):
        enhancer = SmallTargetEnhancer(model_path=model_path, scale=4)

        # 2. 读取图片 (模拟 RGB 输入)
        if os.path.exists(input_img_path):
            img_bgr = cv2.imread(input_img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # (H, W, 3)

            # 3. 执行增强
            print(f"Input shape: {img_rgb.shape}")
            result_rgb = enhancer.process(img_rgb)
            print(f"Output shape: {result_rgb.shape}")

            # 4. 保存
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_img_path, result_bgr)
            print(f"Saved to {output_img_path}")
    else:
        print("Please provide a valid model path to run the example.")