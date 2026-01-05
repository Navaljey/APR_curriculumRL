
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_environment(env, path=None, title="Environment", **kwargs):
    """3D 환경 및 경로 시각화"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 그리드 크기
    x_size, y_size, z_size = env.grid_size
    ax.set_xlim(0, x_size)
    ax.set_ylim(0, y_size)
    ax.set_zlim(0, z_size)
    
    # 장애물 시각화
    # grid 값이 1인 인덱스 추출
    obstacles = np.argwhere(env.grid == 1)
    if len(obstacles) > 0:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], 
                   c='red', marker='s', s=50, alpha=0.3, label='Obstacle')
    
    # 시작점과 목표점
    if env.path_history:
        start = env.path_history[0]
        ax.scatter(start[0], start[1], start[2], c='blue', marker='o', s=100, label='Start')
        
    if env.target_pos is not None:
        target = env.target_pos
        ax.scatter(target[0], target[1], target[2], c='green', marker='*', s=150, label='Target')
        
    # 경로 시각화
    if path is not None:
        path_coords = np.array(path)
        # 파이프(경로)를 더 두껍고 명확하게 표현
        ax.plot(path_coords[:, 0], path_coords[:, 1], path_coords[:, 2], c='cyan', linewidth=4, label='Pipe Path')
        ax.scatter(path_coords[:, 0], path_coords[:, 1], path_coords[:, 2], c='blue', marker='o', s=30, alpha=0.6)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    
    # 저장 경로가 있고 경로 데이터가 유효한지 확인
    should_save = False
    if 'save_path' in kwargs:
        if path is not None:
            should_save = True
            
    if should_save:
        try:
            plt.savefig(kwargs['save_path'])
            print(f"시각화 저장됨: {kwargs['save_path']}")
        except Exception as e:
            print(f"시각화 저장 실패: {e}")
        
    plt.show()

def plot_environment_interactive(env, path=None, title="Interactive Environment", **kwargs):
    """Plotly를 이용한 3D 상호작용 환경 시각화"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly가 설치되지 않았습니다. !pip install plotly를 실행하세요.")
        return

    x_size, y_size, z_size = env.grid_size
    
    data = []
    
    # 1. 장애물 (빨간색 큐브 형태)
    obstacles = np.argwhere(env.grid == 1)
    if len(obstacles) > 0:
        data.append(go.Scatter3d(
            x=obstacles[:, 0], y=obstacles[:, 1], z=obstacles[:, 2],
            mode='markers',
            marker=dict(size=5, color='red', symbol='square', opacity=0.5),
            name='Obstacle'
        ))
        
    # 2. 시작점 (파란색 구)
    if env.path_history:
        start = env.path_history[0]
        data.append(go.Scatter3d(
            x=[start[0]], y=[start[1]], z=[start[2]],
            mode='markers',
            marker=dict(size=10, color='blue', symbol='circle'),
            name='Start'
        ))
        
    # 3. 목표점 (초록색 다이아몬드)
    if env.target_pos is not None:
        target = env.target_pos
        data.append(go.Scatter3d(
            x=[target[0]], y=[target[1]], z=[target[2]],
            mode='markers',
            marker=dict(size=12, color='green', symbol='diamond'),
            name='Target'
        ))
        
    # 4. 경로 (두꺼운 파이프라인)
    if path is not None:
        path_coords = np.array(path)
        data.append(go.Scatter3d(
            x=path_coords[:, 0], y=path_coords[:, 1], z=path_coords[:, 2],
            mode='lines+markers',
            line=dict(color='cyan', width=8),
            marker=dict(size=4, color='blue'),
            name='Pipe Path'
        ))
        
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[0, x_size], title='X'),
            yaxis=dict(range=[0, y_size], title='Y'),
            zaxis=dict(range=[0, z_size], title='Z'),
            aspectmode='cube'
        ),
        margin=dict(r=0, l=0, b=0, t=40)
    )
    
    fig = go.Figure(data=data, layout=layout)
    
    # HTML 저장
    if 'save_path' in kwargs:
        html_path = kwargs['save_path'].replace('.png', '.html')
        fig.write_html(html_path)
        print(f"인터랙티브 시각화 저장됨: {html_path}")
        
    fig.show()

def plot_success_rates(history, title="Success Rate over Episodes"):
    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.grid(True)
    plt.show()
