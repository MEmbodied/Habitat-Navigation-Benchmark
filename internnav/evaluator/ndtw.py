import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
import logging

@registry.register_measure
class NDTW(Measure):
    """
    Normalized Dynamic Time Warping (NDTW) 评测指标
    用于评估智能体走过的轨迹与基准参考轨迹 (Reference Path) 的保真度。
    """
    cls_uuid: str = "ndtw"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        # DTW 的容忍系数，通常和 success_distance 保持一致（默认 3.0 米）
        self._success_distance = getattr(config, "success_distance", 3.0)
        super().__init__(**kwargs)

    def _get_uuid(self, *args, **kwargs) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args, **kwargs):
        # 每一个新的 episode 开始时，重置机器人的走过路径
        self._previous_position = self._sim.get_agent_state().position
        self._agent_path = [self._previous_position]
        self.update_metric(episode=episode, *args, **kwargs)

    def update_metric(self, episode, *args, **kwargs):
        # 每一帧更新时，记录机器人当前坐标
        current_position = self._sim.get_agent_state().position
        self._agent_path.append(current_position)
        self._previous_position = current_position

        # 提取我们在 convert_to_benchmark_rxr.py 里存入的 reference_path
        if not hasattr(episode, "reference_path") or not episode.reference_path:
            logging.warning("Episode missing reference_path! NDTW set to 0.0")
            self._metric = 0.0
            return

        reference_path = np.array(episode.reference_path)
        agent_path = np.array(self._agent_path)

        # 核心算法：使用 FastDTW 计算两条空间轨迹的欧氏距离
        dtw_distance, _ = fastdtw(agent_path, reference_path, dist=euclidean)

        # 计算 NDTW: exp(-DTW / (参考路径长度 * 容忍系数))
        # 公式: NDTW = exp(-DTW / (c * |R|))
        c = self._success_distance
        n_dtw = np.exp(-dtw_distance / (len(reference_path) * c))

        # 将最终得分赋给 Habitat 的 metric 变量
        self._metric = n_dtw