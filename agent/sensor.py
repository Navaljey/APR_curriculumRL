
import numpy as np
import torch

class RaySensor:
    def __init__(self, max_range=3):
        self.max_range = max_range
        
        # 센서 방향 (26방향: 상하좌우전후 + 대각선)
        # 3x3x3 그리드에서 (0,0,0) 제외한 모든 방향
        self.sensor_directions = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    if x == 0 and y == 0 and z == 0:
                        continue
                    self.sensor_directions.append(np.array([x, y, z]))
                    
    def get_observation_size(self):
        # 26개 방향 거리 + 현재 위치(3) + 목표 위치(3) + 거리 차이(3) = 35
        # 하지만 main.py에서는 21개라고 주석에 써있었다가 나중에 계산됨.
        # 실제로는 (센서 개수) + (상대 좌표) 등으로 구성
        # 여기서는 센서값(26) + 상대목표위치(3) = 29 로 구현하거나
        # main.py의 주석은 예전 버전일 수 있으므로 논리에 맞게 구현.
        return len(self.sensor_directions) + 3

    def create_full_observation(self, agent_pos, target_pos, env):
        """
        전체 관찰값 생성
        [센서값들(0~1), 목표방향벡터(정규화)]
        """
        # 1. 장애물 거리 감지
        sensor_vals = self._scan_surroundings(agent_pos, env)
        
        # 2. 목표 방향 벡터
        target_vec = target_pos - agent_pos
        dist = np.linalg.norm(target_vec)
        if dist > 0:
            target_vec = target_vec / dist
        else:
            target_vec = np.zeros(3)
            
        # 결합
        obs = np.concatenate([sensor_vals, target_vec])
        return obs

    def _scan_surroundings(self, agent_pos, env):
        """주변 장애물 탐지 (Raycasting)"""
        distances = []
        
        for direction in self.sensor_directions:
            dist = self._cast_ray(agent_pos, direction, env)
            # 정규화 (0~1)
            distances.append(dist / self.max_range)
            
        return np.array(distances)

    def _cast_ray(self, start_pos, direction, env):
        """단일 방향 레이캐스팅 (부피 고려)"""
        for step in range(1, self.max_range + 1):
            check_pos = start_pos + direction * step
            
            # 환경의 충돌 감지 로직(부피 고려)을 그대로 사용
            # 주의: _is_collision은 '해당 위치로 이동했을 때'의 충돌을 검사함.
            # 따라서 check_pos에 agent가 위치한다고 가정하고 검사.
            if env._is_collision(check_pos):
                return step - 1
                
        return self.max_range