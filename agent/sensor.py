
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
        """단일 방향 레이캐스팅"""
        for step in range(1, self.max_range + 1):
            check_pos = start_pos + direction * step
            
            # 충돌 체크 (obstacle_manager나 env 메서드 사용 가능하지만 직접 접근)
            # 경계 벗어남
            if (check_pos < 0).any() or (check_pos >= env.grid_size).any():
                return step - 1 # 바로 앞이 벽이면 0
                
            # 장애물
            x, y, z = check_pos
            if env.grid[x, y, z] == 1: # 1: Obstacle
                return step - 1
                
        return self.max_range