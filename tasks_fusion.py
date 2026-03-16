
import numpy as np
import logging
from fastapi import APIRouter, HTTPException
from schemas import FusionRequest, FusionResponse, FusionResult, SingleTargetResult
from utils import base64_to_pil
from config import settings
from tasks_analysis import ModelManager
from ultralytics import rt_detr

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fusion", tags=["Fusion"])

def process_single_modality(np_image, det_model_path, cls_model_path, conf=0.25, iou=0.45):
    """
    Process image: Detect -> For each object -> Crop -> Classify -> Top5
    Returns: List[SingleTargetResult]
    """
    results = []
    try:
        # 1. Load Models
        det_model = ModelManager.get_model(det_model_path)
        cls_model = ModelManager.get_model(cls_model_path)
        
        # 2. Detection
        det_results = det_model.predict(np_image, conf=conf, iou=iou, verbose=False)
        det_result = det_results[0]
        
        if not det_result.boxes:
            return []
            
        h, w, _ = np_image.shape
        
        # 3. Iterate over all detected boxes
        for box in det_result.boxes:
            # Extract bbox
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Boundary checks
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            current_bbox = [x1, y1, x2 - x1, y2 - y1] # xywh
            
            # Crop
            crop_img = np_image[y1:y2, x1:x2]
            
            # Classify
            cls_results = cls_model.predict(crop_img, verbose=False)
            cls_result = cls_results[0]
            
            top5_list = []
            if cls_result.probs:
                top5_indices = cls_result.probs.top5
                top5_confs = cls_result.probs.top5conf
                
                if not isinstance(top5_indices, list):
                    top5_indices = top5_indices.tolist()
                if hasattr(top5_confs, 'tolist'):
                    top5_confs = top5_confs.tolist()
                    
                for idx, conf_val in zip(top5_indices, top5_confs):
                    name = cls_model.names[idx]
                    top5_list.append({"class": name, "conf": float(conf_val)})
            else:
                 # Fallback if detection model is used as classifier or something goes wrong
                 cls_id = int(box.cls[0])
                 name = det_model.names[cls_id]
                 conf_val = float(box.conf[0])
                 top5_list.append({"class": name, "conf": conf_val})
            
            results.append(SingleTargetResult(bbox=current_bbox, top5=top5_list))
            
        return results
        
    except Exception as e:
        logger.error(f"Error processing modality: {e}")
        return []

@router.post("/evaluate", response_model=FusionResponse)
async def evaluate_fusion(request: FusionRequest):
    """
    多源融合评估
    
    1. 对可见光图像进行检测+裁剪+分类
    2. 对红外图像进行检测+裁剪+分类
    3. 返回所有目标检测框及分类结果
    """
    # 1. 解码图像
    try:
        vis_pil = base64_to_pil(request.vis_img)
        ir_pil = base64_to_pil(request.ir_img)
        
        vis_np = np.array(vis_pil)
        ir_np = np.array(ir_pil)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")
        
    # 2. 处理可见光
    vis_targets = process_single_modality(
        vis_np, 
        settings.MODEL_PATH_VISIBLE_DET, 
        settings.MODEL_PATH_VISIBLE_CLS,
        conf=request.conf,
        iou=request.iou
    )
    
    # 3. 处理红外
    ir_targets = process_single_modality(
        ir_np, 
        settings.MODEL_PATH_INFRARED_DET, 
        settings.MODEL_PATH_INFRARED_CLS,
        conf=request.conf,
        iou=request.iou
    )
    
    fusion_res = FusionResult(
        vis_targets=vis_targets,
        ir_targets=ir_targets
    )
    
    return FusionResponse(
        scores=None, # 这里暂时不计算融合指标，如果需要可以添加
        fused_image=None,
        fusion_result=fusion_res
    )
