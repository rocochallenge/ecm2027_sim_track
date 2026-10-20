import gymnasium as gym

gym.register(id="Local-Scene-Usd", entry_point="isaaclab.envs:ManagerBasedRLEnv", kwargs={"env_cfg_entry_point": "robofinals.modules.scenes.base:LocalScene"}, disable_env_checker=True)
