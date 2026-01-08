
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


def _create_sphere_mesh(pos, radius, color='blue', resolution=6):
    """구 메쉬 생성 (조인트용)"""
    import plotly.graph_objects as go
    
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2*np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)
    
    x = pos[0] + radius * np.sin(phi) * np.cos(theta)
    y = pos[1] + radius * np.sin(phi) * np.sin(theta)
    z = pos[2] + radius * np.cos(phi)
    
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=1.0,
        name='Pipe Joint'
    )

def _create_cylinder_mesh(p1, p2, radius, color='cyan', resolution=6):
    """두 점을 잇는 원통 메쉬 생성"""
    import plotly.graph_objects as go
    
    # 벡터 계산
    v = p2 - p1
    length = np.linalg.norm(v)
    if length == 0:
        return None
        
    # 원통의 기본 좌표 생성 (Z축 정렬)
    z = np.linspace(0, length, 2)
    theta = np.linspace(0, 2*np.pi, resolution)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)
    
    # 회전 변환 (Z축 -> v 벡터 방향)
    # 기본 Z축 단위 벡터
    k = np.array([0, 0, 1])
    # 회전 축 (외적)
    if np.allclose(v/length, k):
        # 이미 Z축 정렬됨
        R = np.eye(3)
    elif np.allclose(v/length, -k):
        # -Z축 (180도 회전) -> X축 기준 180도
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        # Rodrigues' rotation formula
        u = np.cross(k, v/length)
        u = u / np.linalg.norm(u)
        cost = np.dot(k, v/length)
        sint = np.linalg.norm(np.cross(k, v/length))
        
        # Skew-symmetric matric
        K = np.array([[0, -u[2], u[1]],
                      [u[2], 0, -u[0]],
                      [-u[1], u[0], 0]])
        R = np.eye(3) + K * sint + (K @ K) * (1 - cost)
        
    # 좌표 변환 적용
    # x,y,z 그리드를 (N, M) 형태 유지하며 변환
    new_x = np.zeros_like(x_grid)
    new_y = np.zeros_like(y_grid)
    new_z = np.zeros_like(z_grid)
    
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            # 로컬 좌표
            vec = np.array([x_grid[i,j], y_grid[i,j], z_grid[i,j]])
            # 회전 + 평행이동(p1)
            new_vec = R @ vec + p1
            new_x[i,j] = new_vec[0]
            new_y[i,j] = new_vec[1]
            new_z[i,j] = new_vec[2]
            
    return go.Surface(
        x=new_x, y=new_y, z=new_z,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=1.0,
        name='Pipe Segment'
    )

def plot_environment_interactive(env, path=None, title="Interactive Environment", **kwargs):
    """Plotly를 이용한 3D 상호작용 환경 시각화 (Volumetric)"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly가 설치되지 않았습니다. !pip install plotly를 실행하세요.")
        return

    x_size, y_size, z_size = env.grid_size
    agent_radius = getattr(env, 'agent_radius', 0)
    # 실제 표현될 반경 (0 -> 0.5, 1 -> 1.5 등 3x3 박스에 꽉 차게)
    visual_radius = agent_radius + 0.4
    
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
        
    # 2. 시작점 (파란색 구 - 더 크게)
    if env.path_history:
        start = env.path_history[0]
        # 시작점 강조
        data.append(_create_sphere_mesh(start, visual_radius * 1.2, color='blue'))
        
    # 3. 목표점 (초록색 구)
    if env.target_pos is not None:
        target = env.target_pos
        data.append(_create_sphere_mesh(target, visual_radius * 1.2, color='green'))
        
    # 4. 경로 (부피가 있는 파이프)
    if path is not None:
        path_coords = np.array(path)
        
        # 각 지점(Joint)에 구 생성 - 방향이 꺾이는 곳만 생성하여 최적화
        if len(path_coords) > 0:
            # 시작점
            data.append(_create_sphere_mesh(path_coords[0], visual_radius, color='cyan', resolution=6))
            
            for i in range(1, len(path_coords) - 1):
                prev = path_coords[i-1]
                curr = path_coords[i]
                next_p = path_coords[i+1]
                
                # 벡터 계산
                v1 = curr - prev
                v2 = next_p - curr
                
                # 방향이 다를 때만 조인트 생성 (Vector equality check)
                if not np.array_equal(v1, v2):
                    data.append(_create_sphere_mesh(curr, visual_radius, color='cyan', resolution=6))
            
            # 끝점
            data.append(_create_sphere_mesh(path_coords[-1], visual_radius, color='cyan', resolution=6))
            
        # 각 구간(Segment)에 원통 생성
        for i in range(len(path_coords) - 1):
            p1 = path_coords[i]
            p2 = path_coords[i+1]
            cylinder = _create_cylinder_mesh(p1, p2, visual_radius, color='cyan', resolution=6)
            if cylinder:
                data.append(cylinder)
                
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
