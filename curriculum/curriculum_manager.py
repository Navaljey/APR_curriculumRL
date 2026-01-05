
import numpy as np

class CurriculumManager:
    def __init__(self, env_config, hyperparams):
        self.env_config = env_config
        self.params = hyperparams
        
        self.current_stage = 0
        self.max_stage = max(env_config.stage_sizes.keys())
        
        # 성과 기록
        self.success_history = []
        self.history_window = 50  # 최근 50개 에피소드 기준
        
        # 다음 스테이지로 넘어갈 성공률 임계값
        self.promotion_threshold = 0.85
        
    def add_result(self, success):
        """에피소드 결과 기록 (True/False)"""
        self.success_history.append(1 if success else 0)
        if len(self.success_history) > self.history_window:
            self.success_history.pop(0)
            
    def get_success_rate(self):
        if not self.success_history:
            return 0.0
        return sum(self.success_history) / len(self.success_history)
        
    def update_stage(self):
        """성과에 따라 스테이지 업데이트 확인"""
        if len(self.success_history) < self.history_window:
            return False
            
        rate = self.get_success_rate()
        
        if rate >= self.promotion_threshold and self.current_stage < self.max_stage:
            print(f"\n[Curriculum] 스테이지 승급! {self.current_stage} -> {self.current_stage + 1}")
            print(f" (최근 성공률: {rate*100:.1f}%)\n")
            self.current_stage += 1
            self.success_history = [] # 기록 초기화 (새 환경 적응 위해)
            return True
            
        return False
        
    def get_current_config(self):
        """현재 스테이지에 맞는 환경 설정 반환"""
        stage_idx = self.current_stage
        
        size = self.env_config.get_stage_size(stage_idx)
        obstacles_ratio = self.env_config.get_obstacle_density(stage_idx)
        
        total_nodes = np.prod(size)
        num_obstacles = int(total_nodes * obstacles_ratio)
        
        return {
            'size': size,
            'num_obstacles': num_obstacles
        }