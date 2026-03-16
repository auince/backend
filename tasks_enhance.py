
import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException
from schemas import EnhanceRequest, EnhanceResponse
from utils import base64_to_pil, pil_to_base64
from enhanceScripts import EnhanceManager

router = APIRouter(prefix="/vision", tags=["Enhancement"])

@router.post("/enhance", response_model=EnhanceResponse)
async def enhance_image(request: EnhanceRequest):
    """
    处理图像增强任务
    method: defog, denoise, deblur, tiny_target_enhance
    """
    # 1. 解码图片
    try:
        pil_image = base64_to_pil(request.image)
        # 转换为 numpy 数组 (H, W, C) - RGB
        np_image = np.array(pil_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # 2. 根据 method 调用对应的增强脚本
    method = request.method
    enhanced_np = np_image # 默认不做处理
    
    try:
        if method == "defog":
            processor = EnhanceManager.get_dehazer()
            enhanced_np = processor.process(np_image)
            
        elif method == "defog_darkchannel":
            processor = EnhanceManager.get_dehazer_darkchannel()
            enhanced_np = processor.process(np_image)
            
        elif method == "denoise":
            processor = EnhanceManager.get_deblur_denoiser()
            enhanced_np = processor.process(np_image)

        elif method == "deblur":
            processor = EnhanceManager.get_deblur_denoiser()
            enhanced_np = processor.process(np_image)
            
        elif method == "tiny_target_enhance":
            processor = EnhanceManager.get_small_enhancer()
            enhanced_np = processor.process(np_image)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported enhance method: {method}")

        # 3. 结果转回 Base64
        # 确保数据类型正确 (uint8)
        enhanced_np = enhanced_np.astype(np.uint8)
        enhanced_pil = Image.fromarray(enhanced_np)
        enhanced_b64 = pil_to_base64(enhanced_pil)
        
        return EnhanceResponse(enhanced_image=enhanced_b64)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")
