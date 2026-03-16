
from config import settings
import logging

logger = logging.getLogger(__name__)

class EnhanceManager:
    _dehazer = None
    _deblur_denoiser = None
    _small_enhancer = None

    @classmethod
    def load_all_models(cls):
        """预加载所有增强模型"""
        logger.info("Pre-loading all Enhancement models...")
        try:
            cls.get_dehazer()
        except Exception as e:
            logger.error(f"Failed to load Dehazer: {e}")
            
        try:
            cls.get_deblur_denoiser()
        except Exception as e:
            logger.error(f"Failed to load DeblurDenoiser: {e}")
            
        try:
            cls.get_deblur_gaussian()
        except Exception as e:
            logger.error(f"Failed to load DeblurDenoiser: {e}")
            
        try:
            cls.get_small_enhancer()
        except Exception as e:
            logger.error(f"Failed to load SmallTargetEnhancer: {e}")

    @classmethod
    def get_dehazer(cls):
        if cls._deblur_denoiser is None:
            from enhanceScripts.dehaze import DehazePredictor
            # 使用 config 中的路径，默认使用 dehaze 路径
            predictor_path = settings.MODEL_PATH_DEHAZE_PREDCTOR
            critic_path = settings.MODEL_PATH_DEHAZE_CRITIC

            # 注意：如果文件不存在，DeblurDenoiser 会抛出异常
            # 我们在这里捕获它，或者让它抛出以便 load_all_models 记录日志
            cls._dehazer = DehazePredictor(predictor_path=predictor_path,critic_path=critic_path)
        return cls._dehazer
    @classmethod
    def get_dehazer_darkchannel(cls):
        if cls._deblur_denoiser is None:
            from enhanceScripts.dehaze_darkchannel import Dehazer
            # 注意：如果文件不存在，DeblurDenoiser 会抛出异常
            # 我们在这里捕获它，或者让它抛出以便 load_all_models 记录日志
            cls._dehazer = Dehazer()
        return cls._dehazer

    @classmethod
    def get_deblur_denoiser(cls):
        if cls._deblur_denoiser is None:
            from enhanceScripts.deblur_gaussian import Sharpener
            deblur_method = settings.DEBLUR_METHOD if hasattr(settings, 'DEBLUR_METHOD') else 'unsharp'
            deblur_sigma = settings.DEBLUR_SIGMA if hasattr(settings, 'DEBLUR_SIGMA') else 1.2
            deblur_amount = settings.DEBLUR_AMOUNT if hasattr(settings, 'DEBLUR_AMOUNT') else 1.8
            cls._deblur_denoiser = Sharpener(method=deblur_method, sigma=deblur_sigma, amount=deblur_amount)
            # from enhanceScripts.deblur_denoise import RestormerPredictor
            # # 使用 config 中的路径，默认使用 deblur 路径
            # model_path = settings.MODEL_PATH_DEBLUR
            # # 注意：如果文件不存在，DeblurDenoiser 会抛出异常
            # # 我们在这里捕获它，或者让它抛出以便 load_all_models 记录日志
            # cls._deblur_denoiser = RestormerPredictor(model_path=model_path)
        return cls._deblur_denoiser
    # @classmethod
    # def get_deblur_gaussian(cls):
    #     if cls._deblur_denoiser is None:
    #         from enhanceScripts.deblur_gaussian import Sharpener
    #         # 使用 config 中的路径，默认使用 deblur 路径
    #         # 注意：如果文件不存在，DeblurDenoiser 会抛出异常
    #         # 我们在这里捕获它，或者让它抛出以便 load_all_models 记录日志
    #         deblur_method = settings.DEBLUR_METHOD if hasattr(settings, 'DEBLUR_METHOD') else 'unsharp'
    #         deblur_sigma = settings.DEBLUR_SIGMA if hasattr(settings, 'DEBLUR_SIGMA') else 1.2
    #         deblur_amount = settings.DEBLUR_AMOUNT if hasattr(settings, 'DEBLUR_AMOUNT') else 1.8
    #         cls._deblur_denoiser = Sharpener(method=deblur_method, sigma=deblur_sigma, amount=deblur_amount)
    #     return cls._deblur_denoiser
    @classmethod
    def get_small_enhancer(cls):
        if cls._small_enhancer is None:
            from enhanceScripts.smalltargetEnhance import SmallTargetEnhancer
            model_path = settings.MODEL_PATH_SMALLTARGET
            cls._small_enhancer = SmallTargetEnhancer(model_path=model_path)
        return cls._small_enhancer
