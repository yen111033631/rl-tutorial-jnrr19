import os
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
import numpy as np
import ipdb

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

def set_log_dir(model_name="PPO", env_id="Reacher-v4"):
    model_path = f"./logs/models/{model_name}"
    os.makedirs(model_path, exist_ok=True)
    env_listdir = os.listdir(model_path)

    num = 0
    for env_dir in env_listdir:
        if env_id in env_dir:
            now_num = int(env_dir[-1]) 
            num = now_num if now_num > num else num

    log_dir = f"{model_path}/{env_id}_{num + 1}"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir



if __name__ == "__main__":
        
    # Note: pybullet is not compatible yet with Gymnasium
    # you might need to use `import rl_zoo3.gym_patches`
    # and use gym (not Gymnasium) to instantiate the env
    # Alternatively, you can use the MuJoCo equivalent "HalfCheetah-v4"
    # vec_env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])

    env_id = "Reacher-v4"
    log_dir = set_log_dir(model_name="PPO",
                          env_id=env_id)

    vec_env = gym.make(env_id)
    vec_env = Monitor(vec_env, log_dir)
    vec_env = DummyVecEnv([lambda: vec_env])
    # vec_env = DummyVecEnv([lambda: gym.make("HalfCheetah-v4")])
    # Automatically normalize the input features and reward
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                    clip_obs=10.)
    
    # Create Callback
    callback = SaveOnBestTrainingRewardCallback(check_freq=20, log_dir=log_dir, verbose=0)

    model = PPO(policy = "MlpPolicy", 
                env = vec_env, 
                verbose = 1,
                batch_size = 32,
                n_steps = 512,
                gamma = 0.9,
                # learning_rate = 0.000104019,
                learning_rate = 0.0001,
                # ent_coef = 7.52585e-08,
                ent_coef = 7e-08,
                clip_range = 0.3,
                n_epochs = 5,
                gae_lambda = 1.0,
                max_grad_norm = 0.9,
                # vf_coef = 0.950368,
                vf_coef = 0.95,
                seed = 525,    
            )

    # ipdb.set_trace()
    model.learn(total_timesteps = 50_000, 
                callback = callback)

    # # Don't forget to save the VecNormalize statistics when saving the agent
    # # log_dir = "/tmp/mojoco/"
    # model.save(log_dir + env_id)
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    vec_env.save(stats_path)