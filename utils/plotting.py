import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # ensure 3D plotting is available
from matplotlib.animation import FuncAnimation
import numpy as np

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
    axs[1].set_ylabel("Joint Torque (Nm)")
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
    
    # 3D trajectory plot for end-effector path
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the end-effector path in 3D
    ax.plot(end_effector_positions[:, 0], end_effector_positions[:, 1], end_effector_positions[:, 2], 'b-', label='End Effector Path')
    # Mark start, target, and end points
    ax.scatter(end_effector_positions[0, 0], end_effector_positions[0, 1], end_effector_positions[0, 2], c='g', s=100, label='Start')
    if target_position is not None:
        target_position = np.array(target_position)
        ax.scatter(target_position[0], target_position[1], target_position[2], c='r', s=100, label='Target')
    ax.scatter(end_effector_positions[-1, 0], end_effector_positions[-1, 1], end_effector_positions[-1, 2], c='b', s=100, label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('End Effector Trajectory')
    ax.legend()
    plt.tight_layout()
    plt.savefig("trajectory_3d.png")
    plt.show()

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
