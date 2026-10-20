import gymnasium as gym

gym.register(
    id="Local-Task-ChamberServiceTask",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "robofinals_tasks.chamber_service_task:ChamberServiceTask",
    },
    disable_env_checker=True,
)
