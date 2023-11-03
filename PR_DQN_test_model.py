import os

import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq, log_dir, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                            self.best_mean_reward, mean_reward
                        )
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model at {} timesteps".format(x[-1]))
                        print("Saving new best model to {}.zip".format(self.save_path))
                    self.model.save(self.save_path)

        return True
    
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from stable_baselines3.common.evaluation import evaluate_policy


import gymnasium as gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import DQN

# Create log dir
log_dir = "/tmp/gym/1102DQN_v2_test_seed/"
os.makedirs(log_dir, exist_ok=True)



env_id = "PongNoFrameskip-v4"
video_folder = "logs/videos/"
video_length = 1000


# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
vec_env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=525, monitor_dir=log_dir)
# Frame-stacking with 4 frames
vec_env = VecFrameStack(vec_env, n_stack=4)




# Create Callback
# callback = SaveOnBestTrainingRewardCallback(check_freq=20, log_dir=log_dir, verbose=0)

load_model = DQN.load("/home/neaf2080/code/yen/RL/rl-baselines3-zoo/logs/dqn/PongNoFrameskip-v4_1/PongNoFrameskip-v4.zip")

# mean_reward, std_reward = evaluate_policy(load_model, vec_env, n_eval_episodes=10, warn=False)

# print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")


obs = vec_env.reset()


# Record the video starting at the first step
vec_env = VecVideoRecorder(vec_env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix=f"DQN-agent-{env_id}")

vec_env.reset()
for _ in range(video_length + 1):
    # action, _states = load_model.predict(obs, deterministic=True)
    action = [vec_env.action_space.sample()]
    obs, _, _, _ = vec_env.step(action)
  
# Save the video
vec_env.close()