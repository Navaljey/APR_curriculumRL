
import numpy as np

class NodeEnvironment:
    def __init__(self, x_size, y_size, z_size):
        self.grid_size = np.array([x_size, y_size, z_size])
        
        # 3D 그리드 초기화 (0: 빈 공간, 1: 장애물, 2: 파이프)
        self.grid = np.zeros((x_size, y_size, z_size), dtype=int)
        
        self.agent_pos = None
        self.target_pos = None
        self.path_history = []
        
        # 행동 정의 (상, 하, 좌, 우, 전, 후)
        # x축, y축, z축
        self.actions = [
            np.array([1, 0, 0]),   # x+
            np.array([-1, 0, 0]),  # x-
            np.array([0, 1, 0]),   # y+
            np.array([0, -1, 0]),  # y-
            np.array([0, 0, 1]),   # z+
            np.array([0, 0, -1])   # z-
        ]
        
    def reset(self, start_pos, target_pos):
        """환경 초기화"""
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.agent_pos = np.array(start_pos)
        self.target_pos = np.array(target_pos)
        
        # 경로 기록 초기화
        self.path_history = [self.agent_pos.copy()]
        
        # 그리드에 시작점 표시
        self._update_grid(self.agent_pos, 2)
        
        return self.agent_pos

    def add_obstacle(self, pos, size):
        """장애물 추가"""
        x, y, z = pos
        dx, dy, dz = size
        
        # 범위 체크 및 클리핑
        x_end = min(x + dx, self.grid_size[0])
        y_end = min(y + dy, self.grid_size[1])
        z_end = min(z + dz, self.grid_size[2])
        
        self.grid[x:x_end, y:y_end, z:z_end] = 1

    def step(self, action_idx):
        """
        행동 실행
        Returns: next_state, reward, done, info
        Note: 보상은 reward_calculator에서 계산하므로 여기서는 기본값만 반환
        """
        direction = self.actions[action_idx]
        next_pos = self.agent_pos + direction
        
        # 충돌 검사
        if self._is_collision(next_pos):
            # 충돌 시 위치 이동 없음
            return self.agent_pos, 0, True, {'reason': 'collision'}
        
        # 위치 이동
        self.agent_pos = next_pos
        self.path_history.append(self.agent_pos.copy())
        self._update_grid(self.agent_pos, 2)
        
        # 목표 도달 검사
        if np.array_equal(self.agent_pos, self.target_pos):
            return self.agent_pos, 0, True, {'reason': 'goal_reached'}
            
        return self.agent_pos, 0, False, {}

    def _is_collision(self, pos):
        """충돌 여부 확인 (경계 및 장애물)"""
        # 1. 경계 벗어남 확인
        if (pos < 0).any() or (pos >= self.grid_size).any():
            return True
            
        # 2. 장애물 또는 이미 방문한 파이프 확인
        # grid 값이 0이 아니면 충돌 (1: 장애물, 2: 파이프)
        x, y, z = pos
        if self.grid[x, y, z] != 0:
            # 목표 지점인 경우는 예외 (충돌 아님)
            if np.array_equal(pos, self.target_pos):
                return False
            return True
            
        return False

    def _update_grid(self, pos, value):
        x, y, z = pos
        self.grid[x, y, z] = value
        
    def get_environment_info(self):
        """환경 정보 반환"""
        total = np.prod(self.grid_size)
        obstacles = np.sum(self.grid == 1)
        pipes = np.sum(self.grid == 2)
        free = total - obstacles - pipes
        
        return {
            'total_nodes': total,
            'obstacle_nodes': obstacles,
            'pipe_nodes': pipes,
            'free_nodes': free
        }