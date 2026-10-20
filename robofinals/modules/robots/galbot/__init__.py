import gymnasium as gym

gym.register(
    id="Robocasa-Robot-Galbot-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.galbot:GalbotAbsEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-Galbot-Mobile",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.galbot:GalbotMobileEnvCfg",
    },
    disable_env_checker=True,
)
