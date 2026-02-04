#这是Gr00t专用的server文件
'''端口逻辑：
eval_main.py
  |
  | POST http://127.0.0.1:9000/act
  v
server_Gr00t.py   (uvicorn 监听 9000)
  |
  | Gr00tHTTPClient.post → http://127.0.0.1:8000/act
  v
Gr00t 原生模型服务 (监听 8000)
'''
'''启动方式：
uvicorn scripts.eval.server_Gr00t:app \
    --host 127.0.0.1 \
    --port 9000
'''

from fastapi import FastAPI
import numpy as np
from PIL import Image
import json_numpy
from internnav.evaluator.gr00t_http_client import Gr00tHTTPClient   # 🔴 Gr00t 模型调用
import torch
    
app = FastAPI()

# ===== 🔴 Gr00t 模型加载（模型特有）=====
#注：这里正常启动模型即可，如果在本地有直接加载即可，这里是gr00t需要一个新端口运行推理服务
#注意eval_main.py中的url要指向本服务启动时的端口("http://127.0.0.1:9000/act")，而不是这个端口，这个端口只是用于原本的gr00t推理服务的，更换模型不需要这个端口
gr00t_client = Gr00tHTTPClient(
    url="http://127.0.0.1:8000/act"  # 真正的 Gr00t 服务
)

# ===== 构建Gr00t 模型输入 =====
def build_gr00t_obs(
    obs: dict,
):
    """
    obs: build_traj_request() 生成的 dict
    """
    rgb = obs["rgb"]
    gps = obs["gps"]
    yaw = obs["yaw"]
    camera_height = obs["camera_height"]
    instruction = obs["instruction"]

    image = Image.fromarray(rgb).convert("RGB")
    rgb_256 = np.array(image.resize((256, 256)))

    drone_state = np.array(
        [gps[0], gps[1], camera_height, yaw],
        dtype=np.float32
    ).reshape(1, 4)

    return {
        "video.ego_view": rgb_256.reshape(1, 256, 256, 3),
        "state.drone": drone_state,
        "annotation.human.action.task_description": [instruction],
    }

#接受请求并完成实际推理的主要逻辑
@app.post("/act")
def act(req: dict):
    """
    req = {
        "observation": build_traj_request(...) 的输出
    }
    """
    obs = req["observation"]

    # ===== 🔴 Gr00t 私有 preprocessing =====
    #构造输入
    gr00t_obs = build_gr00t_obs(obs)

    # 🔴 调用“真正的 Gr00t 服务”
    gr00t_output = gr00t_client.get_action(gr00t_obs)
    if isinstance(gr00t_output, dict) and "action.delta_pose" in gr00t_output:
        dp_actions = gr00t_output["action.delta_pose"]
    else:
        dp_actions = gr00t_output
    #到这里为止拿到了gr00t模型的输出

    # print("gr00t_output",type(gr00t_output))
    # print("dp_actions",type(dp_actions))
    # print("dp_actions的内容",dp_actions)
    # print("dp_actions_out",type(dp_actions_out))

    response = {
        "action": dp_actions
    }

    return json_numpy.dumps(response)


