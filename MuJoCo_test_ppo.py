import os
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
import os

import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

    
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


if __name__ == "__main__":


    env_id = 'Reacher-v4' 
    num = 2
    
    which_env = f"{env_id}_{num}"    
    log_dir = f"./logs/models/PPO/{which_env}"
    
    # vec_env = gym.make(env_id, render_mode = "human")
    vec_env = gym.make(env_id, render_mode="rgb_array")
    
    vec_env = DummyVecEnv([lambda: vec_env])

    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False,
    #                    clip_obs=10.)

    # stats_path = "/tmp/gym/mujoco/Reacher-v4/vec_normalize.pkl"
    stats_path = f"{log_dir}/vec_normalize.pkl"
    vec_env = VecNormalize.load(stats_path, vec_env)

    vec_env.render_mode = "rgb_array"

    # print(vec_env.render_mode)

    load_model = PPO.load(f"{log_dir}/best_model.zip", env=vec_env)

    mean_reward, std_reward = evaluate_policy(load_model, vec_env, n_eval_episodes=10, warn=False)

    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")


    ### record and save video
    video_folder = "./logs/videos/"
    video_length = 1000

    obs = vec_env.reset()

    vec_env = VecVideoRecorder(vec_env, video_folder,
                        record_video_trigger=lambda x: x == 0, video_length=video_length,
                        name_prefix=f"PPO-agent-{which_env}")
    
    vec_env.reset()
    # vec_env.render()
    for _ in range(video_length + 1):
        action, _states = load_model.predict(obs, deterministic=True)
        # action = [vec_env.action_space.sample()]
        obs, _, _, _ = vec_env.step(action)
    
    # Save the video
    vec_env.close()

