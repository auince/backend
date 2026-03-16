
# 视觉分析系统后端 (Vision Analysis Backend)

这是基于 FastAPI 设计的扁平化后端架构，用于支持 Gradio 前端或其他客户端。

## 目录结构

```
backend/
├── main.py              # 程序入口，App初始化与路由挂载
├── config.py            # 配置管理
├── schemas.py           # 数据模型 (Request/Response 定义)
├── utils.py             # 通用工具函数
├── tasks_analysis.py    # 视觉分析 (分类、检测、分割) 接口
├── tasks_tracking.py    # 视频跟踪 (WebSocket) 接口
├── tasks_enhance.py     # 图像增强 (去雾、去噪等) 接口
├── tasks_fusion.py      # 多模态融合评价接口
├── tasks_geo.py         # 地理位置解算接口
├── tasks_llm.py         # 大模型视觉交互接口
└── requirements.txt     # 依赖列表
```

## 快速开始

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **运行服务**
   ```bash
   python main.py
   # 或者
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **API 文档**
   启动后访问: `http://localhost:8000/docs` 查看交互式 API 文档。

## 架构说明

本架构采用扁平化设计，避免深层嵌套。
- **main.py**: 负责整合所有模块。
- **tasks_*.py**: 按照业务领域划分，每个文件包含该领域的 API 路由和简单的业务逻辑。
- **schemas.py**: 集中管理所有 API 的输入输出格式，确保前后端契约一致。

## 对接前端

前端 `ApiClient` 需要将 `base_url` 指向本服务的地址 (例如 `http://localhost:8000/v1` 或 `http://localhost:8000`)。
目前 `tasks_*.py` 中包含的是 Mock 数据，请在对应位置 (`TODO`) 填入实际的模型推理代码。
# backend
