
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

# --- 通用 ---
class BaseResponse(BaseModel):
    code: int = 200
    message: str = "success"

# --- 1. 图像分析 (Analysis) ---
class VisionAnalysisRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    task_type: str = Field(..., description="Task type e.g. 'infrared_classification', 'structure_detection'")
    conf: float = Field(0.25, description="Confidence threshold")
    iou: float = Field(0.01, description="IoU threshold")
    agnostic_nms: Optional[bool] = Field(True, description="Class-agnostic NMS")
    
class DetectionResult(BaseModel):
    label: str
    confidence: float
    bbox: Optional[List[float]] = None # [x, y, w, h]
    mask: Optional[str] = None
    obb: Optional[List[float]] = None # [x1, y1, x2, y2, x3, y3, x4, y4]

class VisionAnalysisResponse(BaseModel):
    results: List[DetectionResult]
    annotated_image: Optional[str] = None # 可选：返回画了框的图

# --- 2. 图像增强 (Enhance) ---
class EnhanceRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    method: str = Field(..., description="Enhance method: defog, denoise, deblur, tiny_target_enhance")

class EnhanceResponse(BaseModel):
    enhanced_image: str

# --- 3. 融合评价 (Fusion) ---
class FusionRequest(BaseModel):
    vis_img: str
    ir_img: str
    conf: float = Field(0.25, description="Confidence threshold")
    iou: float = Field(0.01, description="IoU threshold")
    agnostic_nms: Optional[bool] = Field(True, description="Class-agnostic NMS")

class SingleTargetResult(BaseModel):
    bbox: List[float] # [x, y, w, h]
    top5: List[Dict[str, Union[str, float]]] # [{"class": "plane", "conf": 0.99}, ...]

class FusionResult(BaseModel):
    vis_targets: List[SingleTargetResult]
    ir_targets: List[SingleTargetResult]

class FusionResponse(BaseModel):
    scores: Optional[Dict[str, float]] = None
    fused_image: Optional[str] = None
    fusion_result: Optional[FusionResult] = None

# --- 4. 地理计算 (Geo) ---
class UAVParams(BaseModel):
    # 必选参数
    alt: float = Field(..., description="飞机高度 (m)")
    lat: float = Field(..., description="飞机纬度")
    lon: float = Field(..., description="飞机经度")
    pitch: float = Field(..., description="飞机俯仰角")
    
    # 新增参数 (参照 TargetLocate.py 需要的输入)
    # 给定默认值以兼容旧代码，或者要求前端必传
    roll: float = Field(0.0, description="飞机滚转角")
    heading: float = Field(0.0, description="飞机航向角")
    payload_azimuth: float = Field(0.0, description="载荷方位角")
    payload_pitch: float = Field(0.0, description="载荷俯仰角")
    
    # 相机参数 (用于像素坐标转角度)
    fov: float = Field(60.0, description="视场角 (deg)")

class GeoRequest(BaseModel):
    image: str
    uav_params: UAVParams
    conf: float = Field(0.25, description="Confidence threshold")
    iou: float = Field(0.01, description="IoU threshold")

class GeoTarget(BaseModel):
    label: str
    confidence: float
    bbox: List[float] # [x, y, w, h]
    target_lat: float
    target_lon: float
    address: Optional[str] = None

class GeoResponse(BaseModel):
    targets: List[GeoTarget] 

# --- 5. 大模型 (LLM) ---
class LLMRequest(BaseModel):
    image: str
    prompt: str

# --- 6. 跟踪 (Tracking) ---
class TrackInitRequest(BaseModel):
    video_source: str
    track_type: str
