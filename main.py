"""
메인 실행 파일
- 전체 시스템을 연결하고 실행합니다
- 커리큘럼 학습을 진행하고 결과를 평가합니다
"""

import numpy as np
import sys
import time
import os
import torch
import argparse

# 설정 파일 임포트
from config.hyperparameters import HyperParameters
from config.environment_config import EnvironmentConfig

# 환경 관련
from environment.node_environment import NodeEnvironment

# 에이전트 관련
from agent.sensor import RaySensor
from agent.reward_calculator import RewardCalculator
from agent.ppo_agent import PPOAgent

# 커리큘럼 관련
from curriculum.curriculum_manager import CurriculumManager

# 학습 관련
from training.trainer import Trainer

def main():
    """메인 실행 함수"""
    
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='선박 배관 자동 경로 설정 (PPO)')
    parser.add_argument('--mode', type=str, default='test', 
                       choices=['train', 'test', 'eval'],
                       help='실행 모드: train(학습), test(시스템 테스트), eval(평가)')
    parser.add_argument('--episodes', type=int, default=10000,
                       help='학습 에피소드 수')
    parser.add_argument('--load', type=str, default=None,
                       help='로드할 모델 경로')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='사용할 디바이스')
    parser.add_argument('--visualize', action='store_true',
                       help='평가 시 시각화 여부')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(" 선박 배관 자동 경로 설정 시스템 (강화학습 기반)")
    print("="*60 + "\n")
    
    # GPU 사용 가능 여부 확인
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
        args.device = 'cpu'
    
    print(f"사용 디바이스: {args.device.upper()}")
    
    # 모델 저장 폴더 생성
    os.makedirs('models', exist_ok=True)
    
    # 1. 설정 로드
    print("\n1. 설정 초기화 중...")
    hyperparams = HyperParameters()
    env_config = EnvironmentConfig()
    
    # 파라미터 출력
    hyperparams.print_parameters()
    
    # 2. 커리큘럼 매니저 생성
    print("\n2. 커리큘럼 학습 시스템 초기화 중...")
    curriculum = CurriculumManager(env_config, hyperparams)
    
    # 3. 센서 및 보상 계산기 생성
    print("3. 센서 시스템 및 보상 계산기 초기화 중...")
    sensor = RaySensor(max_range=env_config.sensor_range)
    reward_calc = RewardCalculator(hyperparams)
    
    print(f"   - 센서 개수: {len(sensor.sensor_directions)}개")
    print(f"   - 관찰값 크기: {sensor.get_observation_size()}개")
    
    # 4. PPO 에이전트 생성
    print("4. PPO 에이전트 초기화 중...")
    input_size = sensor.get_observation_size()  # 21
    action_size = 6  # 상하좌우전후
    
    ppo_agent = PPOAgent(
        input_size=input_size,
        action_size=action_size,
        hyperparams=hyperparams,
        device=args.device
    )
    
    print(f"   - 입력 크기: {input_size}")
    print(f"   - 행동 크기: {action_size}")
    print(f"   - 신경망 파라미터: {sum(p.numel() for p in ppo_agent.network.parameters()):,}개")
    
    # 모델 로드 (있는 경우)
    if args.load:
        print(f"\n5. 모델 로드 중: {args.load}")
        ppo_agent.load(args.load)
    
    # 5. 트레이너 생성
    print("\n6. 트레이너 초기화 중...")
    trainer = Trainer(
        curriculum_manager=curriculum,
        sensor=sensor,
        reward_calc=reward_calc,
        ppo_agent=ppo_agent,
        hyperparams=hyperparams
    )
    
    print("\n" + "="*60)
    print(" 시스템 초기화 완료!")
    print("="*60 + "\n")
    
    # 실행 모드에 따라 분기
    if args.mode == 'test':
        print("테스트 모드: 시스템 동작 확인\n")
        test_system(curriculum, sensor, reward_calc, env_config, ppo_agent)
        
    elif args.mode == 'train':
        print("학습 모드: PPO 강화학습 시작\n")
        trainer.train(max_episodes=args.episodes, save_interval=100)
        
    elif args.mode == 'eval':
        if args.load is None:
            print("⚠️  평가 모드는 --load 옵션으로 모델을 지정해야 합니다.")
            print("예: python main.py --mode eval --load models/ppo_final.pt")
        else:
            print("평가 모드: 학습된 모델 평가\n")
            trainer.evaluate(num_episodes=10, visualize=args.visualize)
    
    print("\n프로그램 종료.")


def test_system(curriculum, sensor, reward_calc, env_config, ppo_agent):
    """
    시스템 동작 테스트
    간단한 에피소드를 실행하여 모든 컴포넌트가 작동하는지 확인
    """
    print("\n   테스트 에피소드 실행 중...")
    
    # 첫 번째 커리큘럼 단계 (10x10x10)
    stage_idx = 0
    env_size = env_config.get_stage_size(stage_idx)
    env_config.print_stage_info(stage_idx)
    
    # 환경 생성
    environment = NodeEnvironment(*env_size, agent_radius=env_config.agent_radius)
    
    # 장애물 추가
    obstacles = env_config.create_random_obstacles(env_size, num_obstacles=3)
    for obs in obstacles:
        environment.add_obstacle(obs['position'], obs['size'])
    
    # 시작점과 목표점 생성
    start_pos, target_pos = env_config.create_random_start_end(env_size)
    
    print(f"   시작점: ({start_pos[0]:.1f}, {start_pos[1]:.1f}, {start_pos[2]:.1f})")
    print(f"   목표점: ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})")
    
    # 환경 리셋
    environment.reset(start_pos, target_pos)
    reward_calc.reset()
    
    # PPO 에이전트로 에피소드 실행
    path = [environment.agent_pos.copy()]
    total_reward = 0
    max_steps = 100
    
    print(f"\n   PPO 에이전트로 {max_steps} 스텝 실행 중...")
    print("   (학습 전이므로 무작위에 가까운 행동을 합니다)")
    
    for step in range(max_steps):
        # 현재 관찰값 가져오기
        observation = sensor.create_full_observation(
            environment.agent_pos,
            environment.target_pos,
            environment
        )
        
        # PPO 에이전트로 행동 선택 (학습 모드)
        action = ppo_agent.select_action(observation, training=False)
        
        # 이전 위치 저장
        prev_pos = environment.agent_pos.copy()
        
        # 행동 실행
        state, basic_reward, done, info = environment.step(action)
        
        # 상세한 보상 계산
        reward = reward_calc.calculate_reward(
            prev_pos,
            environment.agent_pos,
            environment.target_pos,
            done,
            info,
            environment
        )
        
        total_reward += reward
        path.append(environment.agent_pos.copy())
        
        # 목표 도달 또는 충돌 시 종료
        if done:
            reason = info.get('reason', 'unknown')
            print(f"   에피소드 종료: {reason} (스텝: {step+1})")
            break
    
    # 경로 평가
    metrics = reward_calc.calculate_episode_metrics(path)
    reward_calc.print_metrics(metrics)
    
    print(f"   총 보상: {total_reward:.2f}")
    print(f"   달성률: {(step+1)/max_steps*100:.1f}%")
    
    # 환경 정보 출력
    env_info = environment.get_environment_info()
    print(f"\n   환경 정보:")
    print(f"   - 전체 노드: {env_info['total_nodes']:,}개")
    print(f"   - 장애물: {env_info['obstacle_nodes']:,}개")
    print(f"   - 빈 공간: {env_info['free_nodes']:,}개")
    
    print("\n   ✓ 시스템이 정상적으로 작동합니다!")
    print("\n   학습을 시작하려면:")
    print("   python main.py --mode train --episodes 10000")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)