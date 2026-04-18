import json
import os
import re
import math

# ================= 配置路径 =================
# 你用 col_image.py 采集并保存数据的根目录
OUTPUT_DIR = "/du/sjh/benchmark_data_rxr" 
JSONL_FILE = "./data/subtasks/rxr.jsonl"
OUTPUT_JSON = "./data/subtasks/rxr_subtasks.json"
# ============================================

def convert():
    new_episodes = []
    
    print(f"正在读取 {JSONL_FILE} ...")
    
    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
                
            task = json.loads(line)
            
            # 安全防线：如果大模型生成失败的数据，直接跳过
            if task.get("status") != "success":
                continue
                
            # 1. 拿取对应的文件夹名 (也就是场景ID_任务ID)
            folder_name = task["custom_id"] 
            
            # 2. 【核心修复】将 LLM 输出的字符串解析为 JSON 字典
            raw_output = task.get("model_output", "").strip()
            
            # 自动剔除大模型爱带的 markdown 代码块标记 (```json 和 ```)
            if raw_output.startswith("```json"):
                raw_output = raw_output[7:]
            elif raw_output.startswith("```"):
                raw_output = raw_output[3:]
            if raw_output.endswith("```"):
                raw_output = raw_output[:-3]
                
            raw_output = raw_output.strip()

            # 1. 修复 "reason" 后面漏掉的逗号
            raw_output = re.sub(r'"\s*\n(\s*"subtask_instruction":)', r'",\n\1', raw_output)
            # 2. 修复 "scene_summary" 后面漏掉的逗号
            raw_output = re.sub(r'"\s*\n(\s*"segments":)', r'",\n\1', raw_output)
            
            try:
                model_output = json.loads(raw_output)
            except json.JSONDecodeError as e:
                print(f"\n==========================================")
                print(f"[警告] 行 {line_idx} 的 model_output 解析失败！跳过。")
                print(f"报错原因: {e}")
                print(f"大模型输出的罪魁祸首长这样：")
                print(f"{task.get('model_output')}")
                print(f"==========================================\n")
                continue
            
            # 3. 获取对应的位姿元数据
            meta_path = os.path.join(OUTPUT_DIR, folder_name, "metadata.json")
            if not os.path.exists(meta_path):
                print(f"[警告] 找不到 {folder_name} 的本地采集元数据，跳过此条。")
                continue
                
            with open(meta_path, 'r', encoding='utf-8') as mf:
                meta = json.load(mf)
                
            poses = meta['frames_poses']
            
            # 4. 遍历这个视频里所有的被切分的子段落 (segments)
            segments = model_output.get("segments", [])
            for seg_idx, segment in enumerate(segments):
                start_idx = segment['start_frame']
                end_idx = segment['end_frame']
                instruction = segment['subtask_instruction']
                
                # 越界保护 (防止大模型胡言乱语生成了超出视频总帧数的数字)
                start_idx = min(max(start_idx, 0), len(poses) - 1)
                end_idx = min(max(end_idx, 0), len(poses) - 1)
                
                start_pose = poses[start_idx]
                end_pose = poses[end_idx]

                w, x, y, z = start_pose["rotation"]
                correct_start_rotation = [x, y, z, w]

                sx, sy, sz = start_pose["position"]
                ex, ey, ez = end_pose["position"]
                # 计算起点到终点的欧氏距离
                travel_dist = math.sqrt((ex - sx)**2 + (ey - sy)**2 + (ez - sz)**2)
                
                # 如果位移不到 5 厘米 (0.05米)，说明是原地转头任务，直接跳过！
                if travel_dist < 0.05:
                    # print(f"跳过原地任务: 距离 {travel_dist:.3f}m")
                    continue

                sub_poses = poses[start_idx : end_idx + 1]
                reference_path = [p["position"] for p in sub_poses]
                
                raw_scene_id = meta['scene_id']
                clean_scene_id = "mp3d/" + raw_scene_id.split("mp3d/")[-1] if "mp3d/" in raw_scene_id else raw_scene_id
                
                # 构建 Habitat 官方 Benchmark 格式
                new_episode = {
                    "episode_id": f"{meta['episode_id']}_sub{seg_idx}",
                    "trajectory_id": len(new_episodes) + 1,
                    "scene_id": clean_scene_id,
                    "start_position": start_pose["position"],
                    "start_rotation": correct_start_rotation,
                    "goals": [
                        {
                            "position": end_pose["position"],
                            "radius": 0.25
                        }
                    ],
                    "reference_path": reference_path,
                    "instruction": {
                        "instruction_text": instruction,
                        "instruction_tokens": []
                    }
                }
                new_episodes.append(new_episode)

    # 5. 打包并写入
    final_dataset = {
        "episodes": new_episodes,
        "instruction_vocab": {"word_list": []} # 保留防报错祖传字段
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)

    print(f"==========================================")
    print(f"转换完美成功！")
    print(f"共从 JSONL 中提取并生成了 {len(new_episodes)} 个 Benchmark 子任务测试用例。")
    print(f"已保存至：{OUTPUT_JSON}")

if __name__ == "__main__":
    convert()