import numpy as np  
import plotly.graph_objects as go
import configs

file_aca = "/mnt/c/Users/ASUS/Desktop/franka_mpc/data_acados.npz"

data_aca = np.load(file_aca, allow_pickle=True)
pos_aca = data_aca.get("pos")

file_mj = "/mnt/c/Users/ASUS/Desktop/franka_mpc/data_mj.npz"

data_mj = np.load(file_mj, allow_pickle=True)
pos_mj = data_mj.get("pos")

fig3d = go.Figure()
fig3d.add_trace(go.Scatter3d(
    x=pos_aca[:, 0], y=pos_aca[:, 1], z=pos_aca[:, 2],
    mode='lines+markers',
    marker=dict(size=2, color='blue', opacity=0.6),
    line=dict(color='blue'),
    name='End Effector Path acados'
))

fig3d.add_trace(go.Scatter3d(
    x=pos_mj[:, 0], y=pos_mj[:, 1], z=pos_mj[:, 2],
    mode='lines+markers',
    marker=dict(size=2, color='red', opacity=0.6),
    line=dict(color='red'),
    name='End Effector Path mujoco'
))




# Target

target_position = np.array(configs.target_position)
fig3d.add_trace(go.Scatter3d(
    x=[target_position[0]], y=[target_position[1]], z=[target_position[2]],
    mode='markers',
    marker=dict(color='red', size=6),
    name='Target'
))

# 添加球型障碍物
if hasattr(configs, 'Obstacle_Avoidance') and configs.Obstacle_Avoidance:
    if hasattr(configs, 'Obstacle_Position') and hasattr(configs, 'Obstacle_Scale'):
        center = np.array(configs.Obstacle_Position)
        scale = np.array(configs.Obstacle_Scale)
        radius = 1 / (np.min(scale))  
        
        # 创建球面
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig3d.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale=[[0, 'red'], [1, 'darkred']],
            opacity=0.7,
            showscale=False,
            name='Obstacle',
            hovertemplate='<b>Obstacle</b><br>Center: (%.3f, %.3f, %.3f)<br>Radius: %.3f<extra></extra>' % (*center, radius)
        ))

fig3d.update_layout(
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        aspectmode='data'
    ),
    title='End Effector Trajectory (Interactive)',
    margin=dict(l=0, r=0, b=0, t=30),
    uirevision="fixed_axes"
)

# Save interactive HTML
fig3d.write_html("trajectory_3d.html")