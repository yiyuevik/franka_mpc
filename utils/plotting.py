import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # ensure 3D plotting is available
from matplotlib.animation import FuncAnimation
import numpy as np
import plotly.graph_objects as go
import configs

def plot_trajectories(joint_positions, torques, end_effector_positions, target_position=None):
    """
    Plot the end-effector position trajectory, joint torques, and joint angles over time.
    Saves two plots: a time-series plot of positions/torques/angles ('trajectory.png') and a 3D trajectory plot ('trajectory_3d.png').
    """
    joint_positions = np.array(joint_positions)
    torques = np.array(torques)
    end_effector_positions = np.array(end_effector_positions)
    
    # Time-series plots
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    # Plot end-effector position components over time
    axs[0].plot(end_effector_positions)
    axs[0].set_ylabel("End-Effector Position (m)")
    axs[0].legend(["x", "y", "z"])
    axs[0].grid(True)
    # Plot joint torques over time
    axs[1].plot(torques)
    axs[1].set_ylabel("Joint Velocity (rad/s)")
    axs[1].legend([f"τ{i+1}" for i in range(torques.shape[1])])
    axs[1].grid(True)
    # Plot joint angles over time
    axs[2].plot(joint_positions)
    axs[2].set_ylabel("Joint Position (rad)")
    axs[2].set_xlabel("Time step")
    axs[2].legend([f"q{i+1}" for i in range(joint_positions.shape[1])])
    axs[2].grid(True)
    plt.tight_layout()
    plt.savefig("trajectory.png")
    plt.show()
    # plt.close(fig)
    
    # 3D interactive trajectory (Plotly)
    pos = end_effector_positions
    fig3d = go.Figure()

    # Path
    fig3d.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode='lines+markers',
        marker=dict(size=2, color='blue', opacity=0.6),
        line=dict(color='blue'),
        name='End Effector Path'
    ))

    # Start
    fig3d.add_trace(go.Scatter3d(
        x=[pos[0, 0]], y=[pos[0, 1]], z=[pos[0, 2]],
        mode='markers',
        marker=dict(color='green', size=6),
        name='Start'
    ))

    # End
    fig3d.add_trace(go.Scatter3d(
        x=[pos[-1, 0]], y=[pos[-1, 1]], z=[pos[-1, 2]],
        mode='markers',
        marker=dict(color='blue', size=6),
        name='End'
    ))

    # Target
    if target_position is not None:
        target_position = np.array(target_position)
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



def animate_trajectory(end_effector_positions, target_position=None, interval=50):
    """
    Create a 3D animation of the end-effector moving along its trajectory.
    Returns a matplotlib FuncAnimation object (not shown or saved by default).
    """
    pos = np.array(end_effector_positions)
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # Initialize line and point objects for animation
    line, = ax.plot([], [], [], 'b-', label='Path')
    point, = ax.plot([], [], [], 'ro', label='End Effector')
    # Plot static start and target points for reference
    ax.scatter(x[0], y[0], z[0], c='g', s=100, label='Start')
    if target_position is not None:
        target_position = np.array(target_position)
        ax.scatter(target_position[0], target_position[1], target_position[2], c='r', s=100, label='Target')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('End Effector Trajectory Animation')
    ax.legend()
    
    # Initialization function: clear the line and point
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point
    
    # Update function: draw up to the current frame
    def update(frame):
        # Update line to include trajectory up to current frame
        line.set_data(x[:frame+1], y[:frame+1])
        line.set_3d_properties(z[:frame+1])
        # Update moving point position
        point.set_data(x[frame], y[frame])
        point.set_3d_properties(z[frame])
        return line, point
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(x), init_func=init, interval=interval, blit=True)
    # Note: to display or save the animation, call plt.show() or anim.save(...) outside this function.
    return anim
