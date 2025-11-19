"""
Convert collected rollout .npy outputs into mpd-compatible .pt datasets.

This script scans directories (created by
`data_collect.py`), loads `main_simX.npy`, `main_simU.npy` and branch files under
`branches_data/`, and concatenates samples into PyTorch tensors saved as:

  - <out_dir>/u_tensor_{N}-{H}-{nu}.pt        # shape (N, H, nu)
  - <out_dir>/x_traj_tensor_{N}-{H}-{nx}.pt  # shape (N, H, nx)
  - <out_dir>/x0_tensor_{N}-{nx}.pt          # shape (N, nx)

By default this script aligns state/control as follows:
  - get_traj returns X of shape (H+1, nx) and U of shape (H, nu).
  - we take X_trim = X[:H] (timesteps 0..H-1) so that U[t] acts between X[t] and X[t+1].


This script is non-destructive and only reads existing .npy files.
"""
from __future__ import annotations
import os
import glob
import numpy as np
import torch
from typing import List, Tuple

# -------------------------
# Manual configuration

PROCESS_FOLDER = r"data/multimodality/parallel_rollout_20251105_1431"

# TRIM controls how X (length H+1) is aligned to U (length H):
#  - 'start': use X[:H] so X[t] aligns with U[t] (X[0]..X[H-1])
#  - 'next' : use X[1:H+1] so X[t+1] aligns with U[t]
TRIM = "start"
# -------------------------




def process_group_dir(group_dir: str, trim: str = "start") -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Return lists of (x_traj (H,nx), u_traj (H,nu), x0 (nx,)) for all steps and branches in this group.

    Ordering: always append ALL main steps for the group first (in step order),
    then append branch entries in file order. This matches the user's requested
    deterministic ordering; shuffling should happen later at training time.
    """
    X_list = []
    U_list = []
    X0_list = []

    # Main trajectory files
    main_X_p = os.path.join(group_dir, "main_simX.npy")
    main_U_p = os.path.join(group_dir, "main_simU.npy")
    main_X = None
    main_U = None
    if os.path.exists(main_X_p) and os.path.exists(main_U_p):
        main_X = np.load(main_X_p, allow_pickle=False)
        main_U = np.load(main_U_p, allow_pickle=False)
        # Expected shapes: main_X: (N_steps, H+1, nx), main_U: (N_steps, H, nu)
        if main_X.shape[0] != main_U.shape[0]:
            print(f"Warning: step count mismatch in {group_dir}: X {main_X.shape} vs U {main_U.shape}")
        n_steps = min(main_X.shape[0], main_U.shape[0])
    else:
        n_steps = 0

    # Append all main steps first (if present)
    if n_steps > 0:
        for i in range(n_steps):
            X = main_X[i]
            U = main_U[i]
            H = U.shape[0]
            X_trim = X[:H, :] if trim == "start" else X[1 : H + 1, :]
            X_list.append(X_trim)
            U_list.append(U)
            X0_list.append(X[0])

    # Then append branches in file order (branch_step_*.npy)
    branches_dir = os.path.join(group_dir, "branches_data")
    if os.path.isdir(branches_dir):
        for bp in sorted(glob.glob(os.path.join(branches_dir, "branch_step_*.npy"))):
            try:
                branch_entries = np.load(bp, allow_pickle=True)
            except Exception as e:
                print(f"Failed to load branch file {bp}: {e}")
                continue
            # branch_entries is expected to be a list-like of dicts with keys 'x_traj' and 'u_traj'
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
                        print(f"Skipping malformed branch entry in {bp}: {e}")
                        continue
                H = U.shape[0]
                X_trim = X[:H, :] if trim == "start" else X[1 : H + 1, :]
                X_list.append(X_trim)
                U_list.append(U)
                X0_list.append(X[0])

    return X_list, U_list, X0_list


def main():
    if not PROCESS_FOLDER:
        print("PROCESS_FOLDER is not set. Please edit this script and set PROCESS_FOLDER to the parallel_rollout folder to process.")
        return
    if not os.path.isdir(PROCESS_FOLDER):
        print(f"PROCESS_FOLDER was set to {PROCESS_FOLDER} but the folder was not found on disk.")
        return

    out_dirs = [os.path.abspath(PROCESS_FOLDER)]

    all_X = []
    all_U = []
    all_X0 = []
    for od in out_dirs:
        group_dirs = sorted([d for d in glob.glob(os.path.join(od, "group_*")) if os.path.isdir(d)])
        for gd in group_dirs:
            Xs, Us, X0s = process_group_dir(gd, trim=TRIM)
            all_X.extend(Xs)
            all_U.extend(Us)
            all_X0.extend(X0s)

    if not all_X:
        print("No samples found to convert.")
        return

    # Stack into arrays
    X_arr = np.stack(all_X, axis=0)  # (B, H, nx)
    U_arr = np.stack(all_U, axis=0)  # (B, H, nu)
    X0_arr = np.stack(all_X0, axis=0)  # (B, nx)

    B, H, nx = X_arr.shape
    _, H2, nu = U_arr.shape
    assert H == H2, f"Inconsistent H: X H={H} vs U H={H2}"

    # Save into the provided PROCESS_FOLDER
    base = os.path.abspath(PROCESS_FOLDER)
    os.makedirs(base, exist_ok=True)

    # Use underscore-separated filenames as requested: first=sample count, second=horizon
    u_name = os.path.join(base, f"u_tensor_{B}_{H}_{nu}.pt")
    x_name = os.path.join(base, f"x_tensor_{B}_{H}_{nx}.pt")
    x0_name = os.path.join(base, f"x0_tensor_{B}_{nx}.pt")

    print(f"Saving tensors: B={B}, H={H}, nx={nx}, nu={nu}")
    torch.save(torch.from_numpy(U_arr).float(), u_name)
    torch.save(torch.from_numpy(X_arr).float(), x_name)
    torch.save(torch.from_numpy(X0_arr).float(), x0_name)

    # (metadata was removed - script outputs merged tensors only)

    print("Saved:")
    print(" ", u_name)
    print(" ", x_name)
    print(" ", x0_name)


if __name__ == "__main__":
    main()
