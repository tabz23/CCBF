from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import dsrl
import numpy as np
import pyrallis
import torch
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
    ###final ones below
    ###idbf
    "examples/research/models/OfflineHopperVelocityGymnasium-v1_271/combined_model.pth",
    "examples/research/models/OfflineHopperVelocityGymnasium-v1_209/combined_model.pth",
    "examples/research/models/OfflineHopperVelocityGymnasium-v1_693/combined_model.pth",
    ###ccbf
    "examples/research/models/OfflineHopperVelocityGymnasium-v1_848/combined_model.pth",
    "examples/research/models/OfflineHopperVelocityGymnasium-v1_703/combined_model_laststep.pth",
    "examples/research/models/OfflineHopperVelocityGymnasium-v1_887/combined_model.pth",
    # ##CBF
    "examples/research/models/OfflineHopperVelocityGymnasium-v1_409/combined_model.pth",
    "examples/research/models/OfflineHopperVelocityGymnasium-v1_929/combined_model.pth",
    "examples/research/models/OfflineHopperVelocityGymnasium-v1_892/combined_model.pth",
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
        env = wrap_env(env = gym.make("OfflineHopperVelocityGymnasium-v1"))
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
python "examples/research/check/hopper_random/eval_bc_cbf_modif_list.py" --path "logs/OfflineHopperVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-0912"  --eval_episode 20 --device cpu

hopper bc 
Group idbf (Models 1-3):
  Avg Normalized Reward: 0.3266, std: 0.2379
  Avg Normalized Cost: 4.7075, std: 2.2583
  Avg Length: 388.0167, std: 277.1884

Group ccbf (Models 4-6):
  Avg Normalized Reward: 0.0377, std: 0.0042
  Avg Normalized Cost: 0.0900, std: 0.0629
  Avg Length: 74.1667, std: 5.1274

Group cbf (Models 7-9):
  Avg Normalized Reward: 0.0460, std: 0.0127
  Avg Normalized Cost: 0.2323, std: 0.1817
  Avg Length: 83.4333, std: 14.5257
  
none:
  Avg Normalized Reward: 0.0430, std: 0.02
  Avg Normalized Cost: 0.1925, std: 0.27
  Avg Length: 78.3833
  

python "examples/research/check/hopper_random/eval_bc_cbf_modif_list.py" --path "logs/OfflineHopperVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-70bc" --eval_episode 20 --device cpu

hopper bc safe
Group idbf (Models 1-3):
  Avg Normalized Reward: 0.5583, std: 0.0268
  Avg Normalized Cost: 0.4833, std: 0.2195
  Avg Length: 901.1833, std: 54.9389

Group ccbf (Models 4-6):
  Avg Normalized Reward: 0.5583, std: 0.0484
  Avg Normalized Cost: 0.0467, std: 0.0344
  Avg Length: 897.0833, std: 83.1093

Group cbf (Models 7-9)
  Avg Normalized Reward: 0.6093, std: 0.0145
  Avg Normalized Cost: 0.1425, std: 0.1006
  Avg Length: 970.1333, std: 28.0180
none:#none means just set the use cbf param to false and check the policy without cbf how it does

  Avg Normalized Reward: 0.5664, std: 0.0026
  Avg Normalized Cost: 0.0308, std: 0.020
  Avg Length: 910.7833, 
  
  '''

# python "examples/research/check/hopper_random/eval_bc_cbf_modif_list.py" --path "logs/OfflineHopperVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-70bc" --eval_episode 20 --device cpu
