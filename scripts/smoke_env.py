import numpy as np
from envs import PandaMPCEnv

def main():
    env = PandaMPCEnv(render_mode=None, max_sec=2.0)
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done:
        obs, rew, term, trunc, info = env.step(np.zeros(env.action_space.shape, np.float32))
        steps += 1
        done = term or trunc
    print("✅ smoke ok | steps:", steps, "| last_dist:", info.get("dist_to_goal"))

if __name__ == "__main__":
    main()