import sys
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加 backend 到 sys.path
sys.path.append(os.path.abspath("/data1/LCH/Frames/gradio/backend"))

def test_model_loading():
    logger.info("Testing model loading logic...")
    
    try:
        from tasks_analysis import ModelManager
        from enhanceScripts import EnhanceManager
        from config import settings
        
        logger.info(f"Checking config paths:")
        logger.info(f"Deblur: {settings.MODEL_PATH_DEBLUR}")
        # logger.info(f"Dehaze: {settings.MODEL_PATH_DEHAZE}")
        logger.info(f"SmallTarget: {settings.MODEL_PATH_SMALLTARGET}")
        
        # 测试模型加载逻辑 (只会尝试加载存在的)
        logger.info("--- Loading Analysis Models ---")
        ModelManager.load_all_models()
        
        # 测试 Enhance 模型加载逻辑
        logger.info("--- Loading Enhance Models ---")
        EnhanceManager.load_all_models()
        
        logger.info("Test completed successfully (check logs for specific model status).")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_loading()
