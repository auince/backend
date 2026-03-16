
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Vision Analysis Backend"
    API_PREFIX: str = "/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8010
    
    # --- 模型路径配置 ---
    # 假设所有模型都存放在 weights/ 目录下
    
    # 1. 纯分类任务 (如 SAR, HRRP)
    MODEL_PATH_SAR: str = "/root/autodl-tmp/backend/models/yolo11n-cls.pt"
    MODEL_PATH_HRRP: str = "/root/autodl-tmp/backend/models/yolo11n-cls.pt"
    
    # 2. 检测任务 (直接检测)
    MODEL_PATH_STRUCTURE: str = "/root/autodl-tmp/backend/models/yolo11n.pt"
    MODEL_PATH_REMOTE: str = "/root/autodl-tmp/backend/models/yolo11n.pt"
    MODEL_PATH_PORT: str = "/root/autodl-tmp/backend/models/yolo26n-obb.pt"
    MODEL_PATH_OVER_HORIZON: str = "/root/autodl-tmp/backend/models/yolo11n.pt"
    
    # 3. 两阶段任务 (检测 -> 裁剪 -> 分类)
    # 第一阶段：通用检测器 (负责定位)
    MODEL_PATH_INFRARED_DET: str = "/root/autodl-tmp/backend/models/yolo11n.pt"
    MODEL_PATH_VISIBLE_DET: str = "/root/autodl-tmp/backend/models/yolo11n.pt"
    
    # 第二阶段：细粒度分类器 (负责识别)
    MODEL_PATH_INFRARED_CLS: str = "/root/autodl-tmp/backend/models/yolo11n-cls.pt"
    MODEL_PATH_VISIBLE_CLS: str = "/root/autodl-tmp/backend/models/yolo11n-cls.pt"
    
    # 5. SAM 模型 (用于 Prompted Segmentation)
    MODEL_PATH_SAM: str = "/root/autodl-tmp/model/sam3.pt" # 或者 mobile_sam.pt
    
    
    # 6. 增强模型
    MODEL_PATH_DEHAZE_PREDCTOR: str = "/root/autodl-tmp/backend/models/dehaze_predictor.pth"
    MODEL_PATH_DEHAZE_CRITIC: str = "/root/autodl-tmp/backend/models/dehaze_critic.pth"

    MODEL_PATH_DEBLUR: str = "/root/autodl-tmp/backend/models/motion_deblurring.pth"
    DEBLUR_SIGMA: float = 1.2 #sigma: 高斯模糊半径 (unsharp方法用), 越大去模糊越强但噪点越多
    DEBLUR_AMOUNT: float =1.8 #amount: 锐化强度 (1.0-3.0), 越大边缘越锐利
    DEBLUR_METHOD: str ='unsharp' # 'unsharp' 或 'laplacian
    MODEL_PATH_SMALLTARGET: str = "/root/autodl-tmp/backend/models/smalltarget2.pth"

    class Config:
        env_file = ".env"
    
settings = Settings()
