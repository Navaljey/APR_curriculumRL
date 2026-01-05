
class HyperParameters:
    def __init__(self):
        # 학습 관련 파라미터
        self.learning_rate = 0.0003
        self.gamma = 0.99           # 할인율
        self.lmbda = 0.95           # GAE 파라미터
        self.eps_clip = 0.2         # PPO 클리핑
        self.K_epochs = 3           # 업데이트 횟수
        self.batch_size = 64        # 배치 크기
        self.buffer_size = 2048     # 리플레이 버퍼 크기
        
        # 신경망 구조
        self.hidden_dim = 256
        
        # 보상 가중치
        self.w_dist = 1.0           # 거리 감소 보상
        self.w_step = -0.01         # 스텝 페널티
        self.w_collision = -1.0     # 충돌 페널티
        self.w_goal = 10.0          # 목표 도달 보상
        
        # 엔트로피 보너스 (탐험 유도)
        self.entropy_coef = 0.01

    def print_parameters(self):
        print("\n=== 하이퍼파라미터 설정 ===")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Gamma: {self.gamma}")
        print(f"Entropy Coef: {self.entropy_coef}")
        print("===========================\n")