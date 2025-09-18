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
import os
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from bcql_modif_cbf_list import BCQL, BCQLTrainer
from osrl.common.exp_util import load_config_and_model, seed_all


@dataclass
class EvalConfig:
    path: str = "log/.../checkpoint/model.pt"  # This will be ignored in multi-model mode
    noise_scale: List[float] = None
    eval_episodes: int = 20
    best: bool = True  # Previously changed to TRUE
    device: str = "cpu"
    threads: int = 4
    # New field for multi-model evaluation (only need model paths)
   


@pyrallis.wrap()
def eval(args: EvalConfig):
    all_metrics = []
    # Use the model_paths if provided
    
    model_paths = [
        ###idbf
        "examples/research/models/OfflineSwimmerVelocityGymnasium-v1_269/combined_model.pth",#"best_safe acc": 0.9791599869728088,"best_unsafe_acc": 0.8777299463748932
        "examples/research/models/OfflineSwimmerVelocityGymnasium-v1_481/combined_model.pth",#"best_safe acc": 0.9478048920631409,"best_unsafe_acc": 0.84089315533638
        "examples/research/models/OfflineSwimmerVelocityGymnasium-v1_576/combined_model.pth",#"best_safe acc": 0.9485815405845642,"best_unsafe_acc": 0.8504167705774307

        ###CCBF
        "examples/research/models/OfflineSwimmerVelocityGymnasium-v1_784/combined_model.pth",##    "best_safe acc": 0.764591982960701, "best_unsafe_acc": 0.974375969171524
        "examples/research/models/OfflineSwimmerVelocityGymnasium-v1_932/combined_model.pth",##    "best_safe acc": 0.7573345065116882,"best_unsafe_acc": 0.9764593034982681
        "examples/research/models/OfflineSwimmerVelocityGymnasium-v1_709/combined_model.pth",##     "best_safe acc": 0.758169686794281, "best_unsafe_acc": 0.974375969171524
        
        ###CBF
        "examples/research/models/OfflineSwimmerVelocityGymnasium-v1_933/combined_model.pth",    #"best_safe acc": 0.9614395439624787, "best_unsafe_acc": 0.9082919180393219
        "examples/research/models/OfflineSwimmerVelocityGymnasium-v1_278/combined_model.pth",    #"best_safe acc": 0.9791599869728088, "best_unsafe_acc": 0.8777299463748932
        "examples/research/models/OfflineSwimmerVelocityGymnasium-v1_619/combined_model.pth",    #"best_safe acc": 0.974046328663826,"best_unsafe_acc": 0.8911491602659225
    ]
    
    
    for i, model_path in enumerate(model_paths):
        # print(f"\n[{i+1}/{len(model_paths)}] Evaluating model: {os.path.basename(os.path.dirname(model_path))}")
    
        # Generate hyperparameter path from model path
        model_dir = os.path.dirname(model_path)
        hyperparam_paths = os.path.join(model_dir, "hyperparameters.json")
        cfg, model = load_config_and_model(args.path, args.best)
        seed_all(cfg["seed"])
        if args.device == "cpu":
            torch.set_num_threads(args.threads)

        if "Metadrive" in cfg["task"]:
            import gym
        else:
            import gymnasium as gym  # noqa

        # env = wrap_env(
        #     env=gym.make("SafetySwimmerVelocityGymnasium-v1", render_mode="human"),## ADDED THIS changed this to "SafetyCarGoal1Gymnasium-v0" from OfflineCarGoal1Gymnasium-v0 as mentioned in yaml file
        #     reward_scale=cfg["reward_scale"],                                   ##changed this from OfflinePointGoal1Gymnasium-v0 as in yaml file to SafetyPointGoal1Gymnasium-v0
        # )
        env = wrap_env(
            env=gym.make("OfflineSwimmerVelocityGymnasium-v1"),## ADDED THIS changed this to "SafetyCarGoal1Gymnasium-v0" from OfflineCarGoal1Gymnasium-v0 as mentioned in yaml file
            reward_scale=cfg["reward_scale"],                                   ##changed this from OfflinePointGoal1Gymnasium-v0 as in yaml file to SafetyPointGoal1Gymnasium-v0
        )

        # env = OfflineEnvWrapper(env)
        
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
                            device=args.device,
                            model_path=model_path,
                            hyperparams_path=hyperparam_paths
                            )

        ret, cost, length = trainer.evaluate(args.eval_episodes)

        normalized_ret, normalized_cost = env.get_normalized_score(ret, cost)
        all_metrics.append((ret, normalized_ret, cost, normalized_cost, length))
        print(
            f"Eval reward: {ret}, normalized reward: {normalized_ret}; cost: {cost}, normalized cost: {normalized_cost}; length: {length}"
        )
        # Compute statistics for groups
    group_names = ["idbf", "ccbf", "cbf"]

    for idx, group in enumerate(range(0, 9, 3)):
        group_metrics = np.array(all_metrics[group:group+3])  # Extract the subset
        mean_metrics = np.mean(group_metrics, axis=0)
        var_metrics = np.std(group_metrics, axis=0)

        group_name = group_names[idx]  # Assign group name
        
        print(f"\nGroup {group_name} (Models {group+1}-{group+3}):")
        print(f"  Avg Normalized Reward: {mean_metrics[1]:.4f}, std: {var_metrics[1]:.4f}")
        print(f"  Avg Normalized Cost: {mean_metrics[3]:.4f}, std: {var_metrics[3]:.4f}")
        print(f"  Avg Length: {mean_metrics[4]:.4f}, std: {var_metrics[4]:.4f}")

if __name__ == "__main__":

    eval()
    print("All evaluations completed")
    



'''
#python "examples/research/check/swimmer_random/eval_bcql_modif_list.py" --path "logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5" --eval_episodes 20
swimmer bcql
                                                                                                                                           
Group idbf (Models 1-3):
  Avg Normalized Reward: 0.3911, std: 0.0830
  Avg Normalized Cost: 9.3833, std: 1.6493
  Avg Length: 1000.0000, std: 0.0000

Group ccbf (Models 4-6):
  Avg Normalized Reward: 0.2935, std: 0.0269
  Avg Normalized Cost: 1.4025, std: 0.4680
  Avg Length: 1000.0000, std: 0.0000

Group cbf (Models 7-9):
  Avg Normalized Reward: 0.2006, std: 0.1040
  Avg Normalized Cost: 4.6892, std: 3.1959
  Avg Length: 1000.0000, std: 0.0000

None:
  Avg Normalized Reward: 0.2456, std: 0.084
  Avg Normalized Cost: 2.8167, std: 1.15
  Avg Length: 1000.0000, std: 0.0000
  '''
