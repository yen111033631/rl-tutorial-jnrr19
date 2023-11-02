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

# Create log dir
log_dir = "/tmp/gym/1102DQN_v2_test_seed/"
os.makedirs(log_dir, exist_ok=True)


# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
vec_env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=525, monitor_dir=log_dir)
# Frame-stacking with 4 frames
vec_env = VecFrameStack(vec_env, n_stack=4)




# Create Callback
# callback = SaveOnBestTrainingRewardCallback(check_freq=20, log_dir=log_dir, verbose=0)

load_model = DQN.load("/home/neaf2080/code/yen/RL/rl-baselines3-zoo/logs/dqn/PongNoFrameskip-v4_1/PongNoFrameskip-v4.zip")

mean_reward, std_reward = evaluate_policy(load_model, vec_env, n_eval_episodes=10, warn=False)

print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")


# print(load_model.gamma)
# model = DQN(policy = "CnnPolicy", 
#             env = vec_env, 
#             verbose = 1,
#             buffer_size = 100000, 
#             learning_rate = float(1e-4), 
#             batch_size = 32, 
#             learning_starts = 100000,
#             target_update_interval = 1000,
#             train_freq = 4,
#             gradient_steps = 1,
#             exploration_fraction = 0.1,
#             exploration_final_eps = 0.01,
#             optimize_memory_usage = False,
#             seed = 525
#            )

# print(model.learning_rate)

# model.learn(total_timesteps = 10_000_000, 
#             callback = callback)

# model = DQN(policy = "CnnPolicy", 
#             env = vec_env, 
#             verbose = 1,
#             buffer_size = 100000, 
#             learning_rate = float(1e-4), 
#             batch_size = 32, 
#             learning_starts = 100000,
#             target_update_interval = 1000,
#             train_freq = 4,
#             gradient_steps = 1,
#             exploration_fraction = 0.1,
#             exploration_final_eps = 0.01,
#             optimize_memory_usage = False,
#             seed = 525
#            )

# model.learn(total_timesteps = 10_000_000, 
#             callback = callback)
# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=False)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")