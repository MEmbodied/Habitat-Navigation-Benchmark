#本文件仅用于连接，主要职责为将evaluator 里产生的observation dict，通过 HTTP 发给外部 Trajectory Server，并把返回结果还原成 Python 对象
import requests
from internnav.evaluator.final_habitat_vln_evaluator import BaseTrajectoryClient
import json_numpy
import numpy as np
import torch
json_numpy.patch()

#更换模型只需要添加并调用一个新的类即可，以下面这个为例
class Gr00tTrajectoryClient(BaseTrajectoryClient):
    def __init__(self, url):
        self.url = url

    def reset(self, instruction: str, **kwargs):
        pass

    def query(self, obs: dict, **kwargs) -> list[int]:
        #包一层约定协议
        payload = {"observation": obs}
        # 1. 使用 HTTP 发送
        resp = requests.post(
            self.url,
            data=json_numpy.dumps(payload),  
            headers={"Content-Type": "application/json"},
            timeout=5.0,
        )
        resp.raise_for_status()

        # 2. 还原返回值
        result = json_numpy.loads(resp.text)

        if isinstance(result, str):
            result = json_numpy.loads(result)
            
        
        # 3. 获取 delta poses (这是 Server 现在返回的内容)
        dp_actions_np = result["dp_actions"]
        
        # 转换回 tensor (因为 traj_to_actions_Gr00t 可能预期 tensor 输入，或者保持 numpy)
        # 假设 traj_to_actions_Gr00t 能处理 Tensor:
        dp_actions = torch.from_numpy(dp_actions_np)

        # 4. 在此处进行 "轨迹 -> 离散动作" 的转换
        # 这样逻辑就回到了 Client 端
        actions = traj_to_actions_Gr00t(dp_actions)

        if not actions:
            # 如果actions_list 为空，正在追加默认动作 [1] 以维持运行。
            actions = [1]
        
        return {
            "actions": actions,
            "stop": result.get("stop", False) # 顺便把 server 传回来的 stop 也带上
        }

def traj_to_actions_Gr00t(dp_actions,use_discrate_action=True):
    def reconstruct_xy_from_delta(delta_xyt):
        """
        Input:
            delta_xyt: [B, T, 3], dx, dy are position increments in global coordinates, dθ is heading difference (not used for position)
            start_xy: [B, 2] starting point
        Output:
            xy: [B, T+1, 2] reconstructed global trajectory
        """
        start_xy = np.zeros((len(delta_xyt), 2))
        delta_xy = delta_xyt[:, :, :2]  # Take dx, dy parts
        cumsum_xy = np.cumsum(delta_xy, axis=1)  # [B, T, 2]

        B = delta_xyt.shape[0]
        T = delta_xyt.shape[1]
        xy = np.zeros((B, T + 1, 2))
        xy[:, 0] = start_xy
        xy[:, 1:] = start_xy[:, None, :] + cumsum_xy

        return xy

    def trajectory_to_discrete_actions_close_to_goal(trajectory, step_size=0.25, turn_angle_deg=15, lookahead=4):
        actions = []
        yaw = 0.0
        pos = trajectory[0]
        turn_angle_rad = np.deg2rad(turn_angle_deg)
        traj = trajectory
        goal = trajectory[-1]

        def normalize_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        while np.linalg.norm(pos - goal) > 0.2:
            # Find the nearest trajectory point index to current position
            dists = np.linalg.norm(traj - pos, axis=1)
            nearest_idx = np.argmin(dists)
            # Look ahead a bit (not exceeding trajectory end)
            target_idx = min(nearest_idx + lookahead, len(traj) - 1)
            target = traj[target_idx]
            # Target direction
            target_dir = target - pos
            if np.linalg.norm(target_dir) < 1e-6:
                break
            target_yaw = np.arctan2(target_dir[1], target_dir[0])
            # Difference between current yaw and target yaw
            delta_yaw = normalize_angle(target_yaw - yaw)
            n_turns = int(round(delta_yaw / turn_angle_rad))
            if n_turns > 0:
                actions += [2] * n_turns
            elif n_turns < 0:
                actions += [3] * (-n_turns)
            yaw = normalize_angle(yaw + n_turns * turn_angle_rad)

            # Move forward one step
            next_pos = pos + step_size * np.array([np.cos(yaw), np.sin(yaw)])

            # If moving forward one step makes us farther from goal, stop
            if np.linalg.norm(next_pos - goal) > np.linalg.norm(pos - goal):
                break

            actions.append(1)
            pos = next_pos

        return actions

    # unnormalize
    dp_actions[:, :, :2] /= 4.0
    all_trajectory = reconstruct_xy_from_delta(dp_actions.float().cpu().numpy())
    trajectory = np.mean(all_trajectory, axis=0)
    if use_discrate_action:
        actions = trajectory_to_discrete_actions_close_to_goal(trajectory)
        return actions
    else:
        return trajectory