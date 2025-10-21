import os
import numpy as np
import matplotlib.pyplot as plt

GROUP  = 0
JOINTS = [0, 1, 2, 3, 4, 5, 6] 
OUTDIR = "viz"
# ============================


def load_branch_firstu_for_step(group_dir: str, step: int):
    address = os.path.join(group_dir, f"branch_step_{step:03d}.npy")

    try:
        data = np.load(address, allow_pickle=True)
        first_us = [np.asarray(item["u_traj"])[0, :] for item in data]
        return np.stack(first_us, axis=0) if first_us else None
    except Exception as e:
        print(f"[Warn]  {os.path.basename()} load fails:{e}")
    return None

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    target_dir = os.path.join(project_root, f"data/rollout/parallel_rollout_20250929_120754")
    group_dir = os.path.join(target_dir, f"group_{GROUP:02d}")
    branch_dir = os.path.join(group_dir, "branches_data")
    out_dir = os.path.join(group_dir, OUTDIR)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Info] target group dir:{group_dir}")

    main_simU = np.load(os.path.join(group_dir, "main_simU.npy"))
    print(f"[Info] main_simU load successfully,shape {np.asarray(main_simU).shape}")
    main_first_u = main_simU[:,0,:]  # (T, nu)
    T, nu = main_first_u.shape
    print(f"[Info] main_first_u shape:{main_first_u.shape} (T, nu)")

    joints = JOINTS if JOINTS else list(range(min(7, nu)))
    print(f"[Info] joints: {joints}")

    for j in joints:
        plt.figure()
        plt.plot(np.arange(T), main_first_u[:, j], label="main first u")
        missing = 0
        for step in range(T):
            mat = load_branch_firstu_for_step(branch_dir, step)  # (n_branches, nu)
            if mat is None:
                missing += 1
                continue
            y = mat[:, j]
            x = np.full_like(y, step, dtype=float)
            plt.scatter(x, y, s=10)

        plt.xlabel("MPC step")
        plt.ylabel(f"u[{j}]")
        plt.title(f"{os.path.basename(group_dir)} – Joint {j+1} first control (with branches)")
        plt.legend()
        out_path = os.path.join(out_dir, f"{os.path.basename(group_dir)}_joint{j+1}_multimodal.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        if missing:
            print(f"[Info] j={j}: {missing}/{T} steps missing branches")
        print(f"[Save] {out_path}")


    # csv_path = os.path.join(out_dir, "main_first_u.csv")
    # np.savetxt(csv_path, main_first_u, delimiter=",")
    # print(f"[Save] {csv_path}")

if __name__ == "__main__":
    main()
