
import numpy as np

class RewardCalculator:
    def __init__(self, hyperparams):
        self.params = hyperparams
        self.total_reward = 0
        self.step_count = 0
        
    def reset(self):
        self.total_reward = 0
        self.step_count = 0

    def calculate_reward(self, prev_pos, curr_pos, target_pos, done, info, env):
        """보상 계산"""
        reward = 0
        
        # 1. 거리 기반 보상
        prev_dist = np.linalg.norm(target_pos - prev_pos)
        curr_dist = np.linalg.norm(target_pos - curr_pos)
        
        # 거리가 줄어들었으면 양수, 늘어났으면 음수
        dist_improvement = prev_dist - curr_dist
        reward += dist_improvement * self.params.w_dist
        
        # 2. 스텝 페널티 (빠른 경로 유도)
        reward += self.params.w_step
        
        # 3. 이벤트 기반 보상
        if done:
            if info.get('reason') == 'goal_reached':
                reward += self.params.w_goal
            elif info.get('reason') == 'collision':
                reward += self.params.w_collision
                
        self.total_reward += reward
        self.step_count += 1
        
        return reward

    def calculate_episode_metrics(self, path):
        """에피소드 경로 평가 지표"""
        if not path:
            return {}
            
        path_len = len(path)
        # 굽은 횟수 계산 (방향 전환 횟수)
        bends = 0
        if path_len > 2:
            for i in range(1, path_len - 1):
                vec1 = path[i] - path[i-1]
                vec2 = path[i+1] - path[i]
                if not np.array_equal(vec1, vec2):
                    bends += 1
                    
        return {
            'length': path_len,
            'bends': bends,
            'total_reward': self.total_reward
        }
        
    def print_metrics(self, metrics):
        if not metrics:
            return
        print(f"   - 경로 길이: {metrics['length']}")
        print(f"   - 굴절 횟수: {metrics['bends']}")