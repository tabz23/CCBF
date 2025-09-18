from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
import importlib
import dsrl
import numpy as np
import pyrallis
import torch
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from pyrallis import field
import sys

sys.path.append('/Users/i.k.tabbara/Documents/python directory/OSRL')


from osrl.algorithms import BCQL, BCQLTrainer
from osrl.common.exp_util import load_config_and_model, seed_all


# import sys
# import importlib
# import os

# # Specify the path to your directory
# your_directory_path = '/Users/i.k.tabbara/Documents/python directory/OSRL/osrl'

# # Iterate through the modules in sys.modules
# for module_name, module in sys.modules.items():
#     try:
#         # Check if the module is inside your specific directory
#         if module_name.startswith('osrl.') and os.path.exists(module.__file__):
#             # Ensure it's from the correct directory
#             if your_directory_path in module.__file__:
#                 print(f"Reloading {module_name}...")
#                 importlib.reload(module)
#     except Exception as e:
#         print(f"Error reloading {module_name}: {e}")


@dataclass
class EvalConfig:
    path: str = "log/.../checkpoint/model.pt"
    noise_scale: List[float] = None
    eval_episodes: int = 20
    best: bool = True####I CHANGED THIS TO TRUE, PREVIOUSLY IT WAS FALSE
    device: str = "cpu"
    threads: int = 4


@pyrallis.wrap()
def eval(args: EvalConfig):

    cfg, model = load_config_and_model(args.path, args.best)
    seed_all(cfg["seed"])
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    if "Metadrive" in cfg["task"]:
        import gym
    else:
        import gymnasium as gym  # noqa

    env = wrap_env(
        env=gym.make("SafetyCarGoal2Gymnasium-v0", render_mode="human"),## ADDED THIS changed this to "SafetyCarGoal1Gymnasium-v0" from OfflineCarGoal1Gymnasium-v0 as mentioned in yaml file
        reward_scale=cfg["reward_scale"],                                   ##changed this from OfflinePointGoal1Gymnasium-v0 as in yaml file to SafetyPointGoal1Gymnasium-v0
    )
    # print("hi")
    # env = gym.make("SafetyCarGoal1-v0", render_mode="human")
    # env.reset()
    # env.render()
    env = OfflineEnvWrapper(env)
    
    env.set_target_cost(cfg["cost_limit"])

    bcql_model = BCQL(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        a_hidden_sizes=cfg["a_hidden_sizes"],
        c_hidden_sizes=cfg["c_hidden_sizes"],
        vae_hidden_sizes=cfg["vae_hidden_sizes"],
        sample_action_num=cfg["sample_action_num"],
        PID=cfg["PID"],
        gamma=cfg["gamma"],
        tau=cfg["tau"],
        lmbda=cfg["lmbda"],
        beta=cfg["beta"],
        phi=cfg["phi"],
        num_q=cfg["num_q"],
        num_qc=cfg["num_qc"],
        cost_limit=cfg["cost_limit"],
        episode_len=cfg["episode_len"],
        device=args.device,
    )
    bcql_model.load_state_dict(model["model_state"])
    bcql_model.to(args.device)
    

    trainer = BCQLTrainer(bcql_model,
                          env,
                          reward_scale=cfg["reward_scale"],
                          cost_scale=cfg["cost_scale"],
                          device=args.device)

    ret, cost, length = trainer.evaluate(args.eval_episodes)
    print("hi")
    print(ret)
    print(cost)
    print(length)
    normalized_ret, normalized_cost = env.get_normalized_score(ret, cost)
    print(
        f"Eval reward: {ret}, normalized reward: {normalized_ret}; cost: {cost}, normalized cost: {normalized_cost}; length: {length}"
    )


if __name__ == "__main__":
    eval()
    print("eval done")