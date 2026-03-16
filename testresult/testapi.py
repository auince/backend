import requests
import base64
import json
import os
import cv2
import numpy as np
import asyncio
import websockets
from PIL import Image, ImageDraw
import io

# --- 配置部分 ---
BASE_URL = "http://localhost:8010/v1"
WS_URL = "ws://localhost:8010/v1"

# 测试数据路径 (用户可修改此处)
TEST_DATA_DIR = "/root/autodl-tmp/backend/testdata"
TEST_RESULT_DIR = "/root/autodl-tmp/backend/testresult"

# 确保输入文件存在，否则会自动生成
TEST_IMAGE_PATH = os.path.join(TEST_DATA_DIR, "/root/autodl-tmp/backend/testdata/ScreenShot_2026-01-24_175151_665.png")
TEST_VIDEO_PATH = os.path.join(TEST_DATA_DIR, "/root/autodl-tmp/backend/testdata/example.mp4")

# 确保输出目录存在
os.makedirs(TEST_RESULT_DIR, exist_ok=True)
os.makedirs(TEST_DATA_DIR, exist_ok=True)

def create_dummy_data():
    """生成测试用的图片和视频文件，如果不存在的话"""
    # 1. 生成测试图片
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Generating dummy image at {TEST_IMAGE_PATH}...")
        img = Image.new('RGB', (640, 480), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        # 画一些矩形模拟目标
        draw.rectangle([100, 100, 200, 200], fill=(255, 0, 0)) # Red box
        draw.rectangle([300, 150, 400, 300], fill=(0, 255, 0)) # Green box
        img.save(TEST_IMAGE_PATH)

    # 2. 生成测试视频
    if not os.path.exists(TEST_VIDEO_PATH):
        print(f"Generating dummy video at {TEST_VIDEO_PATH}...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(TEST_VIDEO_PATH, fourcc, 20.0, (640, 480))
        for i in range(60): # 3 seconds
            frame = np.full((480, 640, 3), 200, dtype=np.uint8)
            # 移动的方块
            x = 100 + i * 5
            cv2.rectangle(frame, (x, 200), (x+50, 250), (0, 0, 255), -1)
            out.write(frame)
        out.release()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def decode_save_image(base64_str, output_path):
    if not base64_str:
        print(f"Warning: No image data to save for {output_path}")
        return
    img_data = base64.b64decode(base64_str)
    with open(output_path, "wb") as f:
        f.write(img_data)
    print(f"Saved image to {output_path}")

def draw_bboxes_and_save(original_image_path, results, output_path):
    """在原图上绘制 bbox 并保存"""
    img = cv2.imread(original_image_path)
    if img is None:
        print(f"Error reading {original_image_path}")
        return

    for res in results:
        bbox = res.get('bbox')
        label = res.get('label', 'Unknown')
        conf = res.get('confidence', 0.0)
        
        if bbox:
            x, y, w, h = [int(v) for v in bbox]
            # cv2.rectangle uses (x1, y1), (x2, y2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, img)
    print(f"Saved annotated result to {output_path}")

# --- 测试函数 ---

def test_vis_cls():
    print("\n[TEST] Vision Classification (vis_cls)")
    url = f"{BASE_URL}/vision/analyze"
    img_b64 = encode_image(TEST_IMAGE_PATH)
    payload = {"image": img_b64, "task_type": "vis_cls", "conf": 0.1, "iou": 0, 'agnostic_nms':True}
    
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # print(data)

        results = data.get("results", [])
        print(f"Detected {len(results)} objects.")
        
        # 结果可视化
        output_path = os.path.join(TEST_RESULT_DIR, "vis_cls_result.jpg")
        draw_bboxes_and_save(TEST_IMAGE_PATH, results, output_path)
        
    except Exception as e:
        print(f"FAILED: {e}")

def test_ir_cls():
    print("\n[TEST] Infrared Classification (ir_cls)")
    url = f"{BASE_URL}/vision/analyze"
    img_b64 = encode_image(TEST_IMAGE_PATH) # 这里应该用红外图，暂时混用测试图
    payload = {"image": img_b64, "task_type": "ir_cls", "conf": 0.1, "iou": 0.001, 'agnostic_nms':True}
    
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # print(data)

        results = data.get("results", [])
        print(f"Detected {len(results)} objects.")
        
        output_path = os.path.join(TEST_RESULT_DIR, "ir_cls_result.jpg")
        draw_bboxes_and_save(TEST_IMAGE_PATH, results, output_path)
        
    except Exception as e:
        print(f"FAILED: {e}")

def test_segmentation():
    print("\n[TEST] Segmentation (Detection + SAM)")
    url = f"{BASE_URL}/vision/analyze"
    img_b64 = encode_image(TEST_IMAGE_PATH)
    payload = {"image": img_b64, "task_type": "segmentation", "conf": 0.25, "iou": 0.45}
    
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        
        # Segmentation 应该返回 annotated_image
        annotated_b64 = data.get("annotated_image")
        if annotated_b64:
            output_path = os.path.join(TEST_RESULT_DIR, "segmentation_result.jpg")
            decode_save_image(annotated_b64, output_path)
        else:
            print("No annotated image returned for segmentation.")
            
    except Exception as e:
        print(f"FAILED: {e}")

def test_enhance(method_name):
    print(f"\n[TEST] Enhancement ({method_name})")
    url = f"{BASE_URL}/vision/enhance"
    img_b64 = encode_image(TEST_IMAGE_PATH)
    payload = {"image": img_b64, "method": method_name}
    
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        
        enhanced_b64 = data.get("enhanced_image")
        if enhanced_b64:
            output_path = os.path.join(TEST_RESULT_DIR, f"enhance_{method_name}_result.jpg")
            decode_save_image(enhanced_b64, output_path)
            
    except Exception as e:
        print(f"FAILED: {e}")

def test_fusion():
    print("\n[TEST] Fusion Evaluation")
    url = f"{BASE_URL}/fusion/evaluate"
    img_b64 = encode_image(TEST_IMAGE_PATH)
    # 使用同一张图模拟可见光和红外
    payload = {"vis_img": img_b64, "ir_img": img_b64, "conf": 0.25, "iou": 0.45}
    
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        
        fusion_result = data.get("fusion_result")
        fused_image_b64 = data.get("fused_image")
        
        print(f"Fusion Result: {json.dumps(fusion_result, indent=2)}")
        
        if fused_image_b64:
            output_path = os.path.join(TEST_RESULT_DIR, "fusion_result.jpg")
            decode_save_image(fused_image_b64, output_path)
        
        # 保存 Fusion Result 到 JSON
        json_output_path = os.path.join(TEST_RESULT_DIR, "fusion_result.json")
        with open(json_output_path, "w") as f:
            json.dump(fusion_result, f, indent=2)
        print(f"Saved fusion result to {json_output_path}")

    except Exception as e:
        print(f"FAILED: {e}")

def test_geo():
    print("\n[TEST] Geo Calculation (Multi-Target)")
    url = f"{BASE_URL}/geo/calculate"
    img_b64 = encode_image(TEST_IMAGE_PATH)
    
    # 模拟无人机参数
    uav_params = {
        "alt": 100.0,
        "lat": 39.9042,
        "lon": 116.4074,
        "pitch": -10.0,
        "roll": 0.0,
        "heading": 90.0,
        "payload_azimuth": 0.0,
        "payload_pitch": -45.0,
        "fov": 60.0
    }
    
    payload = {"image": img_b64, "uav_params": uav_params, "conf": 0.25, "iou": 0.45}
    
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        
        targets = data.get("targets", [])
        print(f"Geo Result: Found {len(targets)} targets")
        
        # 保存结果到 json
        json_output_path = os.path.join(TEST_RESULT_DIR, "geo_result.json")
        with open(json_output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved geo result to {json_output_path}")
        
        # 可视化结果并保存
        if targets:
            img = cv2.imread(TEST_IMAGE_PATH)
            if img is None:
                print(f"Error reading {TEST_IMAGE_PATH} for visualization")
            else:
                for t in targets:
                    bbox = t.get("bbox")
                    label = t.get("label")
                    lat = t.get("target_lat")
                    lon = t.get("target_lon")
                    
                    if bbox:
                        x, y, w, h = [int(v) for v in bbox]
                        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        text = f"{label} Lat:{lat:.4f} Lon:{lon:.4f}"
                        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                img_output_path = os.path.join(TEST_RESULT_DIR, "geo_result_visualized.jpg")
                cv2.imwrite(img_output_path, img)
                print(f"Saved visualized geo result to {img_output_path}")
        
    except Exception as e:
        print(f"FAILED: {e}")

async def test_tracking():
    print("\n[TEST] Tracking (WebSocket)")
    
    # 确保使用绝对路径，因为 Server 需要读取文件
    abs_video_path = os.path.abspath(TEST_VIDEO_PATH)
    if not os.path.exists(abs_video_path):
        print("Video file not found, skipping tracking test.")
        return

    # 准备保存结果视频
    cap = cv2.VideoCapture(abs_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_video_path = os.path.join(TEST_RESULT_DIR, "tracking_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    ws_uri = f"{WS_URL}/vision/track?video_path={abs_video_path}&task_type=vis_cls&conf=0.25&iou=0.45" # 这里的 task_type 决定用哪个模型
    
    try:
        async with websockets.connect(ws_uri) as websocket:
            print("Connected to WebSocket, receiving frames...")
            
            frame_idx = 0
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(message)
                    status = data.get("status")
                    
                    if status == "finished" or status == "error":
                        print(f"Tracking finished with status: {status}")
                        break
                    
                    if status == "tracking":
                        server_frame_id = data.get("frame_id")
                        results = data.get("results", [])
                        
                        # 同步读取本地视频帧
                        # 假设 Server 是顺序处理的，我们这里也顺序读取
                        # 注意：如果 Server 跳帧，这里可能会错位。
                        # 更稳健的方法是读取到对应的 frame_id
                        
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # 绘制
                        for res in results:
                            bbox = res.get('bbox') # [x, y, w, h]
                            if bbox:
                                x, y, w, h = [int(v) for v in bbox]
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                                label = f"{res.get('id', '?')}-{res.get('label', 'obj')}"
                                cv2.putText(frame, label, (x, y-5), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        out.write(frame)
                        frame_idx += 1
                        if frame_idx % 10 == 0:
                            print(f"Processed frame {frame_idx}")

                except asyncio.TimeoutError:
                    print("Timeout waiting for WebSocket message")
                    break
                    
    except Exception as e:
        print(f"WebSocket Tracking FAILED: {e}")
    finally:
        cap.release()
        out.release()
        print(f"Saved tracking video to {output_video_path}")

if __name__ == "__main__":
    # create_dummy_data()
    
    test_vis_cls()
    test_ir_cls()
    test_segmentation()
    
    test_enhance("defog")
    test_enhance("defog_darkchannel")
    test_enhance("denoise")
    test_enhance("deblur")
    test_enhance("tiny_target_enhance")
    
    test_fusion()
    test_geo()
    
    # 运行异步的 tracking 测试
    # asyncio.run(test_tracking())