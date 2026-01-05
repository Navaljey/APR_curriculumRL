
import numpy as np

class EnvironmentConfig:
    def __init__(self):
        # 기본 설정
        self.sensor_range = 3  # 센서 감지 거리
        
        # 커리큘럼 단계별 설정
        # (Grid Size X, Grid Size Y, Grid Size Z)
        self.stage_sizes = {
            0: (10, 10, 10),
            1: (15, 15, 15),
            2: (20, 20, 20),
            3: (30, 30, 30)
        }
        
        # 단계별 장애물 밀도 (전체 노드 대비 비율)
        self.stage_obstacles = {
            0: 0.05,  # 5%
            1: 0.10,  # 10%
            2: 0.15,  # 15%
            3: 0.20   # 20%
        }
        
    def get_stage_size(self, stage_idx):
        """해당 스테이지의 환경 크기 반환"""
        return self.stage_sizes.get(stage_idx, (10, 10, 10))
    
    def get_obstacle_density(self, stage_idx):
        """해당 스테이지의 장애물 밀도 반환"""
        return self.stage_obstacles.get(stage_idx, 0.05)
    
    def print_stage_info(self, stage_idx):
        size = self.get_stage_size(stage_idx)
        print(f"=== 스테이지 {stage_idx} 설정 ===")
        print(f"환경 크기: {size}")
        print(f"장애물 밀도: {self.get_obstacle_density(stage_idx)*100}%")
        print("==========================\n")

    def create_random_obstacles(self, env_size, num_obstacles):
        """랜덤한 위치에 장애물 생성"""
        obstacles = []
        for _ in range(num_obstacles):
            # 랜덤 위치
            pos = np.random.randint([0, 0, 0], env_size)
            # 랜덤 크기 (최대 2x2x2)
            size = np.random.randint([1, 1, 1], [3, 3, 3])
            
            obstacles.append({
                'position': pos,
                'size': size
            })
        return obstacles

    def create_random_start_end(self, env_size):
        """랜덤한 시작점과 목표점 생성 (서로 일정 거리 이상 떨어지도록)"""
        while True:
            start_pos = np.random.randint([0, 0, 0], env_size)
            target_pos = np.random.randint([0, 0, 0], env_size)
            
            # 맨해튼 거리 계산
            dist = np.sum(np.abs(start_pos - target_pos))
            
            # 최소 거리 조건 (환경 크기에 비례)
            min_dist = max(env_size) * 0.5
            
            if dist >= min_dist:
                return start_pos, target_pos