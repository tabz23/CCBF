from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import dsrl
import numpy as np
import pyrallis
import torch
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from pyrallis import field
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from bc_modif_cbf_list import BC, BCTrainer
from osrl.common.exp_util import load_config_and_model, seed_all
# Add to imports at the top
import qpsolvers
import numpy as np
from network_ihab import CombinedCBFDynamics

import random
@dataclass
class EvalConfig:
    path: str = "log/.../checkpoint/model.pt"
    noise_scale: List[float] = None
    costs: List[float] = field(default=[1, 10, 20, 30, 40], is_mutable=True)
    eval_episodes: int = 20
    best: bool = True
    device: str = "cpu"
    threads: int = 4
    

@pyrallis.wrap()
def eval(args: EvalConfig):
    all_metrics = []
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

        # env = wrap_env(env = gym.make("SafetySwimmerVelocityGymnasium-v1",render_mode="human"))#uncomment this and go uncomment env.render in bc_modif to view render. 
        env = wrap_env(env = gym.make("OfflineSwimmerVelocityGymnasium-v1"))
        # env = OfflineEnvWrapper(env)
        env.set_target_cost(cfg["cost_limit"])

        # model & optimizer & scheduler setup
        state_dim = env.observation_space.shape[0]
        if cfg["bc_mode"] == "multi-task":
            state_dim += 1
        bc_model = BC(
            state_dim=state_dim,
            action_dim=env.action_space.shape[0],
            max_action=env.action_space.high[0],
            a_hidden_sizes=cfg["a_hidden_sizes"],
            episode_len=cfg["episode_len"],
            device=args.device,
        )
        bc_model.load_state_dict(model["model_state"])
        bc_model.to(args.device)

        trainer = BCTrainer(bc_model,
                            env,
                            bc_mode=cfg["bc_mode"],
                            cost_limit=cfg["cost_limit"],
                            device=args.device,
                            model_path=model_path,
                            hyperparams_path=hyperparam_paths)

        if cfg["bc_mode"] == "multi-task":
            for target_cost in args.costs:
                env.set_target_cost(target_cost)
                trainer.set_target_cost(target_cost)
                ret, cost, length = trainer.evaluate(args.eval_episodes)
                normalized_ret, normalized_cost = env.get_normalized_score(ret, cost)
                print(
                    f"Eval reward: {ret}, normalized reward: {normalized_ret}; target cost {target_cost}, real cost {cost}, normalized cost: {normalized_cost}"
                )
        else:
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

'''
python "/Users/i.k.tabbara/Documents/python directory/CCBF/OSRL/examples/research/check/swimmer_random/eval_bc_cbf_modif_list.py" --device="mps" --path "logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567"  --eval_episode 20 --device cpu
swimmer BC

Group idbf (Models 1-3):
  Avg Normalized Reward: 0.3609, std: 0.0095
  Avg Normalized Cost: 3.8025, std: 0.3182
  Avg Length: 1000.0000, std: 0.0000

Group ccbf (Models 4-6):
  Avg Normalized Reward: 0.3952, std: 0.0323
  Avg Normalized Cost: 0.9442, std: 0.1807
  Avg Length: 1000.0000, std: 0.0000

Group cbf (Models 7-9):
  Avg Normalized Reward: 0.4051, std: 0.0610
  Avg Normalized Cost: 4.0617, std: 2.2448
  Avg Length: 1000.0000, std: 0.0000
  
None:
  Avg Normalized Reward: 0.4360, std: 0.033
  Avg Normalized Cost: 2.2567, std: 0.644
  Avg Length: 1000.0000, std: 0.0000
  

python "examples/research/check/swimmer_random/eval_bc_cbf_modif_list.py" --device="mps" --path "logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180" --eval_episode 20 --device cpu
swimmer BC safe
Group idbf (Models 1-3):
  Avg Normalized Reward: 0.4995, std: 0.0195
  Avg Normalized Cost: 0.2825, std: 0.0871
  Avg Length: 1000.0000, std: 0.0000

Group ccbf (Models 4-6):
  Avg Normalized Reward: 0.4529, std: 0.0199
  Avg Normalized Cost: 0.0333, std: 0.0228
  Avg Length: 1000.0000, std: 0.0000

Group cbf (Models 7-9):
  Avg Normalized Reward: 0.4558, std: 0.0163
  Avg Normalized Cost: 0.1175, std: 0.0616
  Avg Length: 1000.0000, std: 0.0000
  
None:
  Avg Normalized Reward: 0.4314, std: 0.03
  Avg Normalized Cost: 0.1183, std: 0.054
  Avg Length: 1000.0000, std: 0.0000
'''



