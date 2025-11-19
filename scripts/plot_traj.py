import glob
from utils.plotting import plot_trajectories
import os, sys
import numpy as np
import configs
from utils.helpers import compute_end_effector_position
import plotly.graph_objects as go

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(REPO_ROOT) == "scripts":
    REPO_ROOT = os.path.dirname(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data/multimodality/parallel_rollout_20251105_1414/group_00")
    simX = np.load(os.path.join(data_path, "main_simX.npy"))
    print("simX shape:", simX.shape)
    simX = simX[15,:,:]

    simU = np.load(os.path.join(data_path, "main_simU.npy"))
    # print("simU shape:", simU.shape)
    simU = simU[15,:,:]
    end_effector_positions = np.load(os.path.join(data_path, "main_pos.npy"))
    target_position = configs.target_position
    plot_trajectories(simX, simU, end_effector_positions, target_position)      


    fig3d = go.Figure()

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
   
        
        X = simX
        U = simU
    
        H = U.shape[0]
        X_trim = X[:H, :]
        pos_list = [compute_end_effector_position(q) for q in X_trim] 
        pos = np.asarray(pos_list)

        # 3D interactive trajectory (Plotly)

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
        
        fig3d.write_html("trajectory_3d_b.html")

    branch_path = os.path.join(project_root, "data/multimodality/parallel_rollout_20251105_1411/group_00/branches_data/branch_step_015.npy")
    if os.path.isfile(branch_path):
        try:
            branch_entries = np.load(branch_path, allow_pickle=True)
        except Exception as e:
            print(f"Failed to load branch file {branch_path}: {e}")
        # branch_entries is expected to be a list-like of dicts with keys 'x_traj' and 'u_traj'
        fig3d = go.Figure()

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
        for j, entry in enumerate(branch_entries):
            try:
                X = np.array(entry["x_traj"])  # (H+1, nx)
                U = np.array(entry["u_traj"])  # (H, nu)
            except Exception:
                # Fallback: entry might be a numpy scalar containing a python object
                try:
                    entry = entry.item()
                    X = np.array(entry["x_traj"])  # (H+1, nx)
                    U = np.array(entry["u_traj"])  # (H, nu)
                except Exception as e:
                    print(f"Skipping malformed branch entry in {data_path}: {e}")
                    continue
            H = U.shape[0]
            X_trim = X[:H, :]
            pos_list = [compute_end_effector_position(q) for q in X_trim] 
            pos = np.asarray(pos_list)
            # plot_trajectories(X_trim, U, pos, idx = j)
             # 3D interactive trajectory (Plotly)

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
            
            fig3d.write_html("trajectory_3d_d.html")

if __name__ == "__main__":
    main()