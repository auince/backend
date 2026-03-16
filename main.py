
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from tasks_analysis import router as analysis_router
from tasks_tracking import router as tracking_router
from tasks_enhance import router as enhance_router
from tasks_fusion import router as fusion_router
from tasks_geo import router as geo_router
from tasks_llm import router as llm_router

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.on_event("startup")
async def startup_event():
    # 预加载模型
    from tasks_analysis import ModelManager
    from enhanceScripts import EnhanceManager
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("Starting up: Loading all models...")
    
    ModelManager.load_all_models()
    EnhanceManager.load_all_models()
    
    logger.info("All models loaded (or attempted).")

# CORS 设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(analysis_router,prefix="/v1")
app.include_router(tracking_router,prefix="/v1")
app.include_router(enhance_router,prefix="/v1")
app.include_router(fusion_router,prefix="/v1")
app.include_router(geo_router,prefix="/v1")

app.include_router(llm_router)

@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.APP_NAME}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
