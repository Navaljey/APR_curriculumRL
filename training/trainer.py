
import numpy as np
import os
import torch
from environment.node_environment import NodeEnvironment
from utils.visualization import plot_environment, plot_environment_interactive

class Trainer:
    def __init__(self, curriculum_manager, sensor, reward_calc, ppo_agent, hyperparams):
        self.curriculum = curriculum_manager
        self.sensor = sensor
        self.reward_calc = reward_calc
        self.agent = ppo_agent
        self.params = hyperparams
        
    def train(self, max_episodes, save_interval=100):
        print(f"학습 시작: 총 {max_episodes} 에피소드")
        
        for episode in range(1, max_episodes + 1):
            # 1. 현재 커리큘럼 설정 가져오기
            config = self.curriculum.get_current_config()
            env_size = config['size']
            
            # 2. 환경 생성 및 초기화
            env = NodeEnvironment(*env_size)
            
            # 장애물 배치
            obstacles = self.curriculum.env_config.create_random_obstacles(env_size, config['num_obstacles'])
            for obs in obstacles:
                env.add_obstacle(obs['position'], obs['size'])
            
            # 3. 에피소드 초기화
            start_pos, target_pos = self.curriculum.env_config.create_random_start_end(env_size)
            state_pos = env.reset(start_pos, target_pos)
            self.reward_calc.reset()
            
            done = False
            total_reward = 0
            steps = 0
            
            # 4. 에피소드 실행
            while not done:
                # 관찰값 생성
                obs = self.sensor.create_full_observation(env.agent_pos, env.target_pos, env)
                
                # 행동 선택
                action, log_prob = self.agent.select_action(obs, training=True)
                
                # 실행
                prev_pos = env.agent_pos.copy()
                next_pos, _, done, info = env.step(action)
                
                # 보상 계산
                reward = self.reward_calc.calculate_reward(prev_pos, next_pos, env.target_pos, done, info, env)
                
                # 트랜지션 저장 (S, A, R, S', P, D)
                # S' 생성 (학습위해 필요)
                next_obs = self.sensor.create_full_observation(next_pos, env.target_pos, env)
                
                self.agent.store_transition((obs, action, reward, next_obs, log_prob, done))
                
                total_reward += reward
                steps += 1
                
                # 너무 긴 에피소드 강제 종료
                if steps >= 200:
                    done = True
            
            # 5. 모델 업데이트
            loss = self.agent.update()
            
            # 6. 커리큘럼 업데이트
            success = (info.get('reason') == 'goal_reached')
            self.curriculum.add_result(success)
            self.curriculum.update_stage()
            
            # 7. 로그 및 저장
            if episode % 10 == 0:
                success_rate = self.curriculum.get_success_rate()
                print(f"Ep {episode}/{max_episodes} | Stage: {self.curriculum.current_stage} | "
                      f"Reward: {total_reward:.1f} | Steps: {steps} | "
                      f"Success: {success_rate*100:.1f}% | Loss: {loss:.4f}")
                      
            if episode % save_interval == 0:
                path = f"models/ppo_ep{episode}.pt"
                self.agent.save(path)
                print(f"모델 저장됨: {path}")

    def evaluate(self, num_episodes=10, visualize=True):
        print(f"\n평가 시작: 총 {num_episodes} 에피소드")
        
        # 결과 저장을 위한 디렉토리 생성
        if visualize:
            os.makedirs('results', exist_ok=True)
            
        success_count = 0
        total_rewards = []
        
        for ep in range(num_episodes):
            config = self.curriculum.get_current_config()
            env = NodeEnvironment(*config['size'])
            
            # 장애물 배치
            obstacles = self.curriculum.env_config.create_random_obstacles(config['size'], config['num_obstacles'])
            for obs in obstacles:
                env.add_obstacle(obs['position'], obs['size'])
                
            start_pos, target_pos = self.curriculum.env_config.create_random_start_end(config['size'])
            env.reset(start_pos, target_pos)
            self.reward_calc.reset()
            
            done = False
            ep_reward = 0
            path_history = [env.agent_pos.copy()]
            
            while not done:
                obs = self.sensor.create_full_observation(env.agent_pos, env.target_pos, env)
                action = self.agent.select_action(obs, training=False)
                
                prev_pos = env.agent_pos.copy()
                _, _, done, info = env.step(action)
                path_history.append(env.agent_pos.copy())
                
                reward = self.reward_calc.calculate_reward(prev_pos, env.agent_pos, env.target_pos, done, info, env)
                ep_reward += reward
                
            if info.get('reason') == 'goal_reached':
                success_count += 1
                
            total_rewards.append(ep_reward)
            print(f"Ep {ep+1}: Reward {ep_reward:.1f} | Result: {info.get('reason')}")
            
            if visualize:
                save_path = f"results/eval_ep{ep+1}_{info.get('reason')}.png"
                plot_environment(env, path=path_history, title=f"Episode {ep+1} Result: {info.get('reason')}", save_path=save_path)
                plot_environment_interactive(env, path=path_history, title=f"Episode {ep+1} Result: {info.get('reason')}", save_path=save_path)
            
        print("\n=== 평가 결과 ===")
        print(f"성공률: {success_count/num_episodes*100:.1f}%")
        print(f"평균 보상: {np.mean(total_rewards):.1f}")
