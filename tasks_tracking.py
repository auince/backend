
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import cv2
import numpy as np
import asyncio
import json
import logging
from ultralytics import rt_detr
from config import settings
from tasks_analysis import ModelManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vision", tags=["Tracking"])

def get_model_path_by_task(task_type: str) -> str:
    """根据任务类型返回对应的模型路径"""
    # 这里可以根据实际需求复用检测模型
    if task_type == "structure": return settings.MODEL_PATH_STRUCTURE
    if task_type == "remote": return settings.MODEL_PATH_REMOTE
    if task_type == "port": return settings.MODEL_PATH_PORT
    if task_type == "over_horizon": return settings.MODEL_PATH_OVER_HORIZON
    if task_type == "ir_cls": return settings.MODEL_PATH_INFRARED_DET
    if task_type == "vis_cls": return settings.MODEL_PATH_VISIBLE_DET
    
    # 默认回退到一个通用模型 (例如结构检测或 visible)
    # 或者抛出异常
    logger.warning(f"Unknown tracking task type: {task_type}, using structure model as default.")
    return settings.MODEL_PATH_STRUCTURE

@router.websocket("/track")
async def websocket_endpoint(
    websocket: WebSocket, 
    video_path: str = Query(...), 
    task_type: str = Query("structure"),
    conf: float = Query(0.25),
    iou: float = Query(0.45)
):
    """
    处理视频流式跟踪
    参数:
    - video_path: 视频文件的绝对路径
    - task_type: 任务类型，用于选择模型 (structure, remote, port, over_horizon, infrared, visible)
    - conf: 置信度阈值
    - iou: IoU 阈值
    
    返回 (JSON Stream):
    {
        "frame_id": int,
        "results": [
            {
                "id": int,          # 跟踪 ID
                "label": str,       # 类别名
                "confidence": float,# 置信度
                "bbox": [x, y, w, h]# xywh
            },
            ...
        ],
        "status": "tracking" | "finished" | "error"
    }
    """
    await websocket.accept()
    logger.info(f"Tracking started for video: {video_path} with task: {task_type}")
    
    cap = None
    try:
        # 1. 验证视频路径
        if not os.path.exists(video_path):
             await websocket.send_json({"status": "error", "message": f"Video file not found: {video_path}"})
             await websocket.close()
             return

        # 2. 加载模型
        model_path = get_model_path_by_task(task_type)
        model = ModelManager.get_model(model_path)
        
        # 3. 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            await websocket.send_json({"status": "error", "message": "Failed to open video"})
            await websocket.close()
            return
            
        frame_id = 0
        
        # 4. 逐帧跟踪
        # stream=True 返回生成器
        # persist=True 保持跨帧 ID
        results_generator = model.track(
            source=video_path, 
            persist=True, 
            stream=True, 
            verbose=False,
            conf=conf,
            iou=iou,
            agnostic_nms=True
        )
        
        for result in results_generator:
            # 检查客户端连接状态 (可选，通过 try-catch WebSocketDisconnect 实现)
            frame_results = []
            
            # 解析跟踪结果
            if result.boxes:
                # result.boxes 包含 xyxy, conf, cls, id (如果跟踪成功)
                for box in result.boxes:
                    # 获取跟踪 ID (可能为 None 如果检测到但没跟上)
                    track_id = int(box.id[0]) if box.id is not None else -1
                    
                    # 获取类别和置信度
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    w = x2 - x1
                    h = y2 - y1
                    bbox = [x1, y1, w, h]
                    
                    frame_results.append({
                        "id": track_id,
                        "label": label,
                        "confidence": conf,
                        "bbox": bbox
                    })
            
            # 发送结果给客户端
            response_data = {
                "frame_id": frame_id,
                "results": frame_results,
                "status": "tracking"
            }
            
            await websocket.send_json(response_data)
            
            # 模拟帧率控制 (可选，防止发送太快前端处理不过来)
            await asyncio.sleep(0.001) 
            
            frame_id += 1

        # 视频处理结束
        await websocket.send_json({"status": "finished", "frame_id": frame_id})
        await websocket.close()

    except WebSocketDisconnect:
        logger.info("Client disconnected during tracking")
    except Exception as e:
        logger.error(f"Tracking error: {e}")
        # 尝试发送错误消息 (如果连接还存在)
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
            await websocket.close()
        except:
            pass
    finally:
        if cap:
            cap.release()

import os
