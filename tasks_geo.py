
import numpy as np
import logging
from fastapi import APIRouter, HTTPException
from schemas import GeoRequest, GeoResponse, GeoTarget
from utils import base64_to_pil
from config import settings
from tasks_analysis import ModelManager, _run_predict
from geo_utils import calculate_target_geolocation, decimal_to_dms

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/geo", tags=["Geo"])

@router.post("/calculate", response_model=GeoResponse)
async def calculate_geo(request: GeoRequest):
    """
    目标 GPS 位置解算
    
    逻辑：
    1. 使用 rt_detr 检测图像中的目标
    2. 对每个检测到的目标：
        a. 获取目标中心点像素坐标
        b. 根据无人机参数 (UAVParams) 和像素坐标，计算目标的经纬度
    3. 返回所有目标的解算结果
    """
    
    # 1. 图像解码
    try:
        pil_image = base64_to_pil(request.image)
        np_image = np.array(pil_image)
        img_height, img_width = np_image.shape[:2]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # 2. 目标检测
    target_model_path = settings.MODEL_PATH_STRUCTURE
    try:
        model = ModelManager.get_model(target_model_path)
        result = _run_predict(model, np_image, conf=request.conf, iou=request.iou)
    except Exception as e:
        logger.error(f"Failed to run detection for geo calc: {e}")
        raise HTTPException(status_code=500, detail="Detection model failed")

    # 3. 遍历所有检测目标
    geo_targets = []
    
    params = request.uav_params
    fov = params.fov
    
    if result.boxes and len(result.boxes) > 0:
        for box in result.boxes:
            # a. 提取基本信息
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = result.names[cls_id] if result.names else str(cls_id)
            
            bbox = [x1, y1, x2 - x1, y2 - y1] # [x, y, w, h]
            
            # b. 计算中心点
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # c. 计算角度偏移
            delta_azimuth = (center_x - img_width/2) / (img_width/2) * (fov/2)
            delta_pitch = (center_y - img_height/2) / (img_height/2) * (fov/2)
            
            # d. 坐标解算
            target_lon, target_lat = calculate_target_geolocation(
                plane_lon=params.lon,
                plane_lat=params.lat,
                plane_alt=params.alt,
                plane_heading=params.heading,
                plane_pitch=params.pitch,
                plane_roll=params.roll,
                payload_azimuth=params.payload_azimuth + delta_azimuth,
                payload_pitch=params.payload_pitch - delta_pitch
            )
            
            # e. 格式化地址
            lat_dms = decimal_to_dms(target_lat, is_latitude=True)
            lon_dms = decimal_to_dms(target_lon, is_latitude=False)
            address_str = f"Lat: {lat_dms}, Lon: {lon_dms}"
            
            geo_targets.append(GeoTarget(
                label=label,
                confidence=conf,
                bbox=bbox,
                target_lat=target_lat,
                target_lon=target_lon,
                address=address_str
            ))
            
    else:
        logger.warning("No target detected in image.")
        # 如果没有检测到目标，是否需要返回空列表？根据 response_model=GeoResponse(targets=[])，是的。
        # 也可以保留原逻辑：用图片中心计算一个默认坐标？
        # 用户要求"每个被检出的框都计算"，如果没有框，自然没有计算结果。
        # 所以返回空列表是合理的。
        pass

    return GeoResponse(targets=geo_targets)
