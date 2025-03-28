import sys
from fastchat.serve.controller import Controller
from fastchat.serve.model_worker import ModelWorker
from fastchat.serve.openai_api_server import app, CORSMiddleware, app_settings

import sys
import uuid
import uvicorn
from fastapi import FastAPI


# 创建不同的 FastAPI 应用实例
controller_app = FastAPI(title="FastChat Controller Service")
worker_app = FastAPI(title="FastChat Model Worker Service")
api_app = FastAPI(title="FastChat OpenAI API Service")


def start_controller():
    """启动 FastChat Controller 服务

    分布式系统设计中常见的一种优化策略，决定如何分配任务给不同的model_worker（或服务器）
     - LOTTERY：这种方法系统会随机选择一个worker。不考虑worker的当前负载或任何其他因素。
     - SHORTEST_QUEUE：这种方法会选择当前队列长度最短的worker，也就是当前负载最小的工人。
    """

    controller = Controller(dispatch_method="shortest_queue")
    # sys.modules 是一个字典，它存储了已经加载的模块。每个键是一个模块名，每个值是一个模块对象。
    # 这种机制使得Python在导入模块时可以检查模块是否已经在sys.modules中，如果是，就直接使用已经加载的模块，避免重复加载。
    sys.modules["fastchat.serve.controller"].controller = controller
    controller_app.title = "FastChat Controller"
    controller_app._controller = controller
    uvicorn.run(controller_app, host="127.0.0.1", port=20001)


def start_model_worker():
    """启动 Model Worker 服务"""
    worker_id = str(uuid.uuid4())[:8]
    worker = ModelWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:20002",
        worker_id=worker_id,
        limit_worker_concurrency=5,
        # no_register=False,
        no_register=True,
        model_path="/root/autodl-tmp/data/model/ZhipuAI/chatglm3-6b",
        num_gpus=4,
        model_names=["chatglm3-6b"],
        device="cuda",
        max_gpu_memory="22GiB",
    )
    worker_app.title = f"FastChat LLM Server ChaGLM3-6b"
    worker_app._worker = worker
    uvicorn.run(worker_app, host="127.0.0.1", port=20002)


def start_openai_api_server():
    """启动 OpenAI API 服务"""
    api_app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,  # 允许前端请求携带认证信息（如 cookies
        allow_origins=["*"],  # 允许所有域名的请求，星号表示不限制任何域。
        allow_methods=["*"],  # 允许所有的 HTTP 方法。
        allow_headers=["*"],  # 允许所有的 HTTP 头
    )
    app_settings.controller_address = "http://127.0.0.1:20001"
    app_settings.api_keys = []
    api_app.title = "FastChat OpenAI API Server"
    uvicorn.run(api_app, host="127.0.0.1", port=8000)



def start_services_sequentially():
    try:
        print("Starting Controller...")
        start_controller()
    except Exception as e:
        print(f"Failed to start Controller: {e}")
        return

    try:
        print("Starting Model Worker...")
        start_model_worker()
    except Exception as e:
        print(f"Failed to start Model Worker: {e}")
        return

    try:
        print("Starting OpenAI API Server...")
        start_openai_api_server()
    except Exception as e:
        print(f"Failed to start OpenAI API Server: {e}")
        return


if __name__ == '__main__':
    # # 启动控制器
    # start_controller()

    # # 启动 Model Worker
    # start_model_worker()

    # # 启动 OpenAI API Server
    # start_openai_api_server()

    start_services_sequentially()