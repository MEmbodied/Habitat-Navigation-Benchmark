import os
import json
import tqdm
import numpy as np
from PIL import Image
import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_baselines.config.default import get_config as get_habitat_config

# 配置参数
CONFIG_PATH = "scripts/eval/configs/our_benchmark_config_sub_r2r.yaml" # 确认路径正确
OUTPUT_DIR = "/data/sjh/InternNav/output"
SPLIT = "val_unseen"

def to_list(obj):
    """安全地将 numpy/quaternion/嵌套 list 转换为 json 可读的标准格式"""
    if obj is None:
        return []
    # 1. 处理 Habitat 特有的四元数对象
    if hasattr(obj, 'components'): 
        return [float(x) for x in obj.components]
    # 2. 处理 Numpy 数组
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # 3. 递归处理嵌套的 List（完美解决 reference_path 报错）
    if isinstance(obj, list):
        return [to_list(item) for item in obj]
    # 4. 基础数字直接返回
    if isinstance(obj, (int, float)):
        return float(obj)
    return obj

def collect():
    # 1. 初始化 Habitat 环境
    config = get_habitat_config(CONFIG_PATH)
    with habitat.config.read_write(config):
        config.habitat.dataset.split = SPLIT
    
    env = habitat.Env(config=config)
    
    # 2. 初始化专家策略 (不需要模型，它会自动寻找最短路径)
    # goal_radius 设为 0.25 米，这是 VLN-CE 常用的成功判定范围
    follower = ShortestPathFollower(env.sim, goal_radius=0.25, return_one_hot=False)

    print(f"开始采集数据集: {SPLIT}, 总计 {len(env.episodes)} 个任务")

    summary_data = []

    for episode in tqdm.tqdm(env.episodes):
        scene_id = episode.scene_id.split("/")[-2]
        episode_id = episode.episode_id
        
        # 文件夹命名：场景名_ID
        folder_name = f"{scene_id}_{episode_id}"
        save_path = os.path.join(OUTPUT_DIR, folder_name)
        images_path = os.path.join(save_path, "images")
        os.makedirs(images_path, exist_ok=True)

        env.current_episode = episode
        observations = env.reset()
        
        step = 0
        frames_metadata = [] # 记录每步的 3D 位姿

        while True:
            # A. 记录图片
            img_filename = f"frame_{step:04d}.png"
            Image.fromarray(observations["rgb"]).save(os.path.join(images_path, img_filename))

            # B. 记录位姿 (由 2D GPS/Compass 升级为 3D Pos/4D Rot)
            state = env.sim.get_agent_state()
            frames_metadata.append({
                "step": step,
                "position": to_list(state.position), # 3D 绝对坐标
                "rotation": to_list(state.rotation)  # 4D 四元数 [x, y, z, w]
            })

            # C. 动作逻辑
            best_action = follower.get_next_action(episode.goals[0].position)
            if best_action is None or best_action == 0 or step > 400:
                break
            observations = env.step(best_action)
            step += 1
        
        total_frames = len(frames_metadata)

        # D. 合并并保存 Episode 完整数据 (去除 tokens 和 path)
        episode_data = {
            "episode_id": str(episode_id),
            "scene_id": episode.scene_id,
            "start_position": frames_metadata[0]["position"],
            "start_rotation": frames_metadata[0]["rotation"],
            "goals": [
                {"position": to_list(g.position), "radius": float(g.radius)} 
                for g in episode.goals
            ],
            "instruction": {
                "instruction_text": episode.instruction.instruction_text
            },
            "image_count": total_frames,
            "frames_poses": frames_metadata # 每一帧的详细坐标供以后按需使用
        }
        
        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(episode_data, f, indent=2)

        # E. 记录到全局汇总表
        summary_data.append({
            "folder_name": folder_name,
            "image_count": step + 1
        })

    env.close()
    with open(os.path.join(OUTPUT_DIR, "dataset_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"采集完成！数据保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    collect()