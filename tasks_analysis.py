
import os
import numpy as np
from fastapi import APIRouter, HTTPException
from schemas import VisionAnalysisRequest, VisionAnalysisResponse, DetectionResult
from utils import base64_to_pil, pil_to_base64
from config import settings
from ultralytics import rt_detr, SAM
import logging
from PIL import Image
import cv2
import random

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vision", tags=["Analysis"])

class ModelManager:
    """
    模型管理器，负责按需加载和缓存模型
    """
    _models = {}

    @classmethod
    def get_model(cls, model_path: str, model_type: str = "rt_detr"):
        # 检查缓存
        if model_path in cls._models:
            return cls._models[model_path]
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            logger.warning(f"Model weights not found at {model_path}, using 'model11n.pt' as fallback or failing.")
            if model_type == "SAM":
                 # 尝试使用 mobile_sam.pt 或报错
                 pass
            raise HTTPException(status_code=500, detail=f"Model weights not found at {model_path}")

        # 加载模型
        try:
            logger.info(f"Loading {model_type} model from {model_path}...")
            if model_type == "SAM":
                model = SAM(model_path)
            else:
                model = rt_detr(model_path)
                
            cls._models[model_path] = model
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    @classmethod
    def load_all_models(cls):
        """
        预加载所有配置的模型
        """
        logger.info("Pre-loading all models...")
        
        # 收集所有需要加载的路径
        paths_to_load = [
            (settings.MODEL_PATH_SAR, "rt_detr"),
            (settings.MODEL_PATH_HRRP, "rt_detr"),
            (settings.MODEL_PATH_STRUCTURE, "rt_detr"),
            (settings.MODEL_PATH_REMOTE, "rt_detr"),
            (settings.MODEL_PATH_PORT, "rt_detr"),
            (settings.MODEL_PATH_OVER_HORIZON, "rt_detr"),
            (settings.MODEL_PATH_INFRARED_DET, "rt_detr"),
            (settings.MODEL_PATH_VISIBLE_DET, "rt_detr"),
            (settings.MODEL_PATH_INFRARED_CLS, "rt_detr"),
            (settings.MODEL_PATH_VISIBLE_CLS, "rt_detr"),
            (settings.MODEL_PATH_SAM, "SAM"),
        ]
        
        # 去重
        unique_paths = set()
        for path, m_type in paths_to_load:
            if path and path not in unique_paths:
                unique_paths.add(path)
                try:
                    # 如果文件不存在，get_model 会抛出异常或 warning
                    # 这里我们 catch 住异常，不阻碍其他模型加载
                    if os.path.exists(path):
                        cls.get_model(path, model_type=m_type)
                    else:
                        logger.warning(f"Skipping pre-load for {path}: File not found")
                except Exception as e:
                    logger.error(f"Failed to pre-load model {path}: {e}")

def _run_predict(model: rt_detr, image_input, conf=0.25, iou=0.001,agnostic_nms=True):
    """
    通用 rt_detr 预测封装
    image_input: PIL.Image 或 np.ndarray
    """
    results = model.predict(image_input, conf=conf,imgsz=model.overrides.get('imgsz', 640), iou=iou,agnostic_nms=agnostic_nms, verbose=True)
    print(results[0])
    return results[0]
    # return results

def _overlay_masks(image: np.ndarray, masks: list, alpha: float = 0.5) -> np.ndarray:
    """
    将 mask 列表叠加到图像上
    image: (H, W, 3) np.uint8
    masks: list of (H, W) bool or uint8
    """
    annotated = image.copy()
    for i, mask in enumerate(masks):
        # 生成随机颜色
        color = [random.randint(0, 255) for _ in range(3)]
        
        # 确保 mask 是 boolean 或 0/1
        if isinstance(mask, np.ndarray):
            if mask.dtype == bool:
                mask_bool = mask
            else:
                # 假设是 0-1 float 或 0-255 int，使用阈值
                mask_bool = mask > 0.5
        else:
            continue
            
        # 叠加颜色
        # 注意: image[mask_bool] 返回的是 (N, 3) 数组
        # 我们需要混合颜色
        roi = annotated[mask_bool]
        if roi.size > 0:
            blended = roi.astype(float) * (1 - alpha) + np.array(color) * alpha
            annotated[mask_bool] = blended.astype(np.uint8)
            
    return annotated

@router.post("/analyze", response_model=VisionAnalysisResponse)
async def analyze_vision(request: VisionAnalysisRequest):
    """
    处理视觉分析任务：分类、检测、分割
    """
    # 1. 解码图片
    try:
        pil_image = base64_to_pil(request.image)
        # 转换为 numpy 数组 (H, W, C) - RGB
        np_image = np.array(pil_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    task_type = request.task_type
    results = []
    annotated_image_b64 = None

    try:
        # === A. 两阶段任务 (检测 -> 裁剪 -> 分类) ===
        if task_type in ["ir_cls", "vis_cls"]:
            # 1. 确定模型路径
            if task_type == "ir_cls":
                det_path = settings.MODEL_PATH_INFRARED_DET
                cls_path = settings.MODEL_PATH_INFRARED_CLS
            else: # vis_cls
                det_path = settings.MODEL_PATH_VISIBLE_DET
                cls_path = settings.MODEL_PATH_VISIBLE_CLS
            
            # 2. 获取模型
            det_model = ModelManager.get_model(det_path)
            cls_model = ModelManager.get_model(cls_path)
            
            # 3. 第一阶段：检测
            # 输入 np_image
            det_result = _run_predict(det_model, np_image, conf=request.conf, iou=request.iou,agnostic_nms = request.agnostic_nms)
            if det_result.boxes:
                for box in det_result.boxes:
                    # 获取检测框坐标 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 边界检查
                    h, w, _ = np_image.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    # 4. 裁剪目标
                    crop_img = np_image[y1:y2, x1:x2]
                    
                    # 5. 第二阶段：分类
                    # 对裁剪后的图片进行分类
                    cls_result = _run_predict(cls_model, crop_img, conf=request.conf, iou=request.iou,agnostic_nms = request.agnostic_nms)
                    
                    final_label = "Unknown"
                    final_conf = 0.0
                    
                    if cls_result.probs:
                        top1_idx = cls_result.probs.top1
                        final_label = cls_model.names[top1_idx]
                        final_conf = float(cls_result.probs.top1conf)
                    else:
                        # 如果分类器没有返回概率（可能是检测模型误用），回退到检测器的类别
                        # 或者视为分类失败
                        cls_id = int(box.cls[0])
                        final_label = det_model.names[cls_id]
                        final_conf = float(box.conf[0])

                    # 构造结果
                    # 前端需要的 bbox 是 [x, y, w, h]
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    results.append(DetectionResult(
                        label=final_label,
                        confidence=final_conf,
                        bbox=bbox
                    ))
        
    # === B. 单阶段任务 (常规检测/分类) ===
        else:
            # 1. 映射任务到模型路径
            model_path = None
            if task_type == "sar_cls": model_path = settings.MODEL_PATH_SAR
            elif task_type == "hrrp_classification": model_path = settings.MODEL_PATH_HRRP
            elif task_type == "strc_det": model_path = settings.MODEL_PATH_STRUCTURE
            elif task_type == "rs_cls": model_path = settings.MODEL_PATH_REMOTE
            elif task_type == "port_det": model_path = settings.MODEL_PATH_PORT
            elif task_type == "over_hor_det": model_path = settings.MODEL_PATH_OVER_HORIZON
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported task type: {task_type}")
            
            model = ModelManager.get_model(model_path)
            
            # 2. 推理（统一使用 result 变量接收单个结果对象）
            result = _run_predict(model, np_image, conf=request.conf, iou=request.iou, agnostic_nms=request.agnostic_nms)
            
            # 3. 解析结果（根据结果类型自动判断）
            # 优先级：OBB > 水平框 > 分类
            if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
                # --- 处理 OBB (旋转框) 结果，如 port_det ---
                for i in range(len(result.obb)):
                    cls_id = int(result.obb.cls[i])
                    class_name = model.names[cls_id]
                    confidence = float(result.obb.conf[i])
                    
                    # 获取 4 点坐标 [x1,y1,x2,y2,x3,y3,x4,y4] 并展平为列表
                    xyxyxyxy = result.obb.xyxyxyxy[i].cpu().numpy().flatten().tolist()
                    
                    results.append(DetectionResult(
                        label=class_name, 
                        confidence=confidence, 
                        obb=xyxyxyxy  # 8个浮点数的列表
                    ))
                
            elif result.boxes:
                # --- 处理水平框检测结果 ---
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h]
                    results.append(DetectionResult(label=class_name, confidence=confidence, bbox=bbox))
                
                    
            elif result.probs:
                # --- 处理分类结果 ---
                top1_index = result.probs.top1
                class_name = model.names[top1_index]
                confidence = float(result.probs.top1conf)
                results.append(DetectionResult(label=class_name, confidence=confidence))
                # 分类任务不返回标注图像（或返回原图）
                annotated_image_b64 = None

    except HTTPException as he:
        # Mock 兜底逻辑保持不变
        if he.status_code == 500 and "Model weights not found" in he.detail:
            logger.warning(f"Fallback to mock data for {task_type}")
            results, annotated_image_b64 = _get_mock_results(task_type, request.image)
        else:
            raise he
    except Exception as e:
        logger.error(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return VisionAnalysisResponse(results=results, annotated_image=annotated_image_b64)

def _get_mock_results(task_type: str, original_image_b64: str = None):
    """Mock 数据"""
    annotated_img = original_image_b64
    results = []
    
    if "classification" in task_type or "cls" in task_type:
        # 即使是两阶段，如果没模型，也返回带 bbox 的 mock 数据
        if task_type in ["ir_cls", "vis_cls"]:
             results = [
                DetectionResult(label="Mock_TwoStage_Target", confidence=0.95, bbox=[50, 50, 100, 100]),
            ]
        else:
            results = [
                DetectionResult(label="Mock_Class_A", confidence=0.92)
            ]
    elif "segment" in task_type:
        results = [DetectionResult(label="Mock_Segment", confidence=0.95, bbox=[100, 100, 200, 200])]
        # 在这里可以画一个假的 mask 图
        # 简单起见返回原图
    else:
        results = [DetectionResult(label="Mock_Target", confidence=0.98, bbox=[100, 100, 150, 100])]
        
    return results, annotated_img
