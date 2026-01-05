
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # 공통 피처 추출기
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor: 행동 확률 출력
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic: 상태 가치 출력
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        features = self.feature_layer(x)
        return self.actor(features), self.critic(features)

class PPOAgent:
    def __init__(self, input_size, action_size, hyperparams, device='cpu'):
        self.device = torch.device(device)
        self.params = hyperparams
        self.input_size = input_size
        self.action_size = action_size
        
        self.network = ActorCritic(input_size, action_size, hyperparams.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=hyperparams.learning_rate)
        
        # 데이터 버퍼
        self.data_buffer = []
        
    def select_action(self, state, training=True):
        """행동 선택"""
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            probs, _ = self.network(state_tensor)
            
        if training:
            # 확률 기반 샘플링
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item()
        else:
            # 탐욕적 선택
            return torch.argmax(probs).item()
            
    def store_transition(self, transition):
        # (s, a, r, s', prob, done)
        self.data_buffer.append(transition)
        
    def update(self):
        """PPO 업데이트"""
        if len(self.data_buffer) < self.params.batch_size:
            return 0.0
            
        # 데이터 변환
        states, actions, rewards, next_states, old_log_probs, dones = zip(*self.data_buffer)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # GAE (Generalized Advantage Estimation) 계산
        # 간소화를 위해 Monte Carlo Return 사용 또는 GAE 구현
        # 여기서는 간단한 Discounted Return 사용
        returns = []
        discounted_sum = 0
        for r, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done:
                discounted_sum = 0
            discounted_sum = r + (self.params.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
            
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = returns - self.network(states)[1].detach().squeeze()
        # 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        
        # PPO Epochs
        for _ in range(self.params.K_epochs):
            probs, state_values = self.network(states)
            dist = torch.distributions.Categorical(probs)
            curr_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # Ratio
            ratio = torch.exp(curr_log_probs - old_log_probs)
            
            # Surrogate Loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.params.eps_clip, 1+self.params.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values.squeeze(), returns)
            
            loss = actor_loss + 0.5 * critic_loss - self.params.entropy_coef * entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        # 버퍼 초기화
        self.data_buffer = []
        
        return total_loss / self.params.K_epochs

    def save(self, path):
        torch.save(self.network.state_dict(), path)
        
    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))