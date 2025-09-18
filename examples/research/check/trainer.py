import os
import sys
import uuid

import types
from dataclasses import asdict
from typing import Any
import json
import random
import bullet_safety_gym  # noqa
import dsrl
import safety_gymnasium as gym
import numpy as np
import pyrallis
import torch
import wandb
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from torch.utils.data import DataLoader
from tqdm.auto import trange
from examples.configs.bc_configs import BC_DEFAULT_CONFIG, BCTrainConfig

from osrl.algorithms import BC, BCTrainer
from osrl.common.dataset import process_bc_dataset
from osrl.common.exp_util import auto_name, seed_all

from network_ihab import AffineDynamics, CBF, CombinedCBFDynamics
from dataset_ihab import TransitionDataset
import torch.nn.functional as F

class CombinedCBFTrainer:
    def __init__(self, model, train_dataset, val_dataset, dt=0.1, lr=1e-4, device="cpu", train_steps=10000, 
                 eval_every_n_steps=100, eval_steps=100, args=None,
                 eps_safe=0.1, eps_unsafe=0.1, eps_grad=0.1, w_safe=1, w_unsafe=1, w_grad=0.1,lambda_lip=20,w_CQL=1,
                 train_dynamics=True, dynamics_lr=1e-4,num_action_samples=10,temp=0.9,detach=False):
        
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.lr = lr
        self.device = device
        self.args = args  
        self.train_steps = train_steps
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_steps = eval_steps
        self.dt = dt
        self.train_dynamics = train_dynamics
        
        self.eps_safe = eps_safe
        self.eps_unsafe = eps_unsafe
        self.eps_grad = eps_grad
        self.w_safe = w_safe
        self.w_unsafe = w_unsafe
        self.w_grad = w_grad
        self.w_CQL=w_CQL
        self.num_action_samples=num_action_samples
        self.temp=temp
        self.dynamics_lr = dynamics_lr
        rng = random.Random()  
        self.random_value = rng.randint(100, 999)
        self.detach=detach
        self.setup_optimizer()

        
        self.lambda_lip =lambda_lip 
        
        config = {
            "learning_rate": lr,
            "train_steps": train_steps,
            "eps_safe": eps_safe,
            "eps_unsafe": eps_unsafe,
            "eps_grad": eps_grad,
            "w_safe": w_safe,
            "w_unsafe": w_unsafe,
            "w_grad": w_grad,
            "w_CQL":w_CQL,
            "dt": dt,
            "cbf_hidden_dim": model.cbf_hidden_dim,
            "dynamics_hidden_dim": model.dynamics_hidden_dim,
            "cbf_num_layers": model.cbf_num_layers,
            "dynamics_num_layers": model.dynamics_num_layers,
            "batch_size": args.batch_size if hasattr(args, 'batch_size') else None,
            "seed": args.seed if hasattr(args, 'seed') else None,
            "train_dynamics": train_dynamics,
            "dynamics_lr": dynamics_lr,
            "task":args.task,
            "lambda_lip":self.lambda_lip,
            "num_action_samples":num_action_samples,
            "temp":self.temp,
            "detach":self.detach
        }
        
        wandb.init(project="combined_cbf_dynamics_training",name=f"run_{self.args.task}_{self.random_value}", config=config)
                
    def compute_loss(self, observations, next_observations, actions, costs, training_bool=None):
        # Determine safe and unsafe states
        safe_mask = (costs <= 0).reshape(-1, 1)
        unsafe_mask = (costs > 0).reshape(-1, 1)

        # Calculate CBF values for current states
        B = self.model.forward_cbf(next_observations).reshape(-1, 1)

        # Safe loss computation
        loss_safe_vector = self.w_safe * F.relu(self.eps_safe - B) * safe_mask
        num_safe_elements = safe_mask.sum()
        loss_safe = loss_safe_vector.sum() / (num_safe_elements + 1e-8)

        # Unsafe loss computation
        loss_unsafe_vector = self.w_unsafe * F.relu(self.eps_unsafe + B) * unsafe_mask
        num_unsafe_elements = unsafe_mask.sum()
        loss_unsafe = loss_unsafe_vector.sum() / (num_unsafe_elements + 1e-8)

        # Compute B values
        B_curr = self.model.forward_cbf(observations).reshape(-1, 1) 
        B_next = self.model.forward_cbf(next_observations).reshape(-1, 1) 
        loss_lip = self.lambda_lip * torch.mean(torch.abs(B_next - B_curr)) 
  
        wandb.log({"loss_lip":loss_lip})
        
        loss_grad = self.compute_gradient_loss(observations, actions, safe_mask) ##safe mask means next state is safe 
        avg_random_cbf = 0.0  #

        loss_cql, logsumexp_h, avg_random_cbf=self.compute_CQL_loss(observations,next_observations,actions,safe_mask) 
        wandb.log({"logsumexp_h": logsumexp_h.mean().item() / self.temp})
        if (self.w_CQL==0):
            loss_cql=torch.tensor(0.0)
            
        
        # Dynamics loss computation if we're also training dynamics
        dynamics_loss = torch.tensor(0.0)
        if self.train_dynamics:
            predicted_next_observations = self.model.forward_next_state(observations, actions)
            loss_fn = torch.nn.MSELoss()
            dynamics_loss = loss_fn(predicted_next_observations, next_observations)

        # Total loss
        cbf_loss = loss_safe + loss_unsafe + loss_grad + loss_lip + loss_cql##added cql loss
        total_loss = cbf_loss + (dynamics_loss if self.train_dynamics else 0.0) 
        
        # Backward pass during training
        if training_bool:
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

        avg_safe_B = (B * safe_mask).sum() / (safe_mask.sum() + 1e-8)
        avg_unsafe_B = (B * unsafe_mask).sum() / (unsafe_mask.sum() + 1e-8)

        safe_acc = ((B >= 0) * safe_mask).sum() / (num_safe_elements + 1e-8)
        unsafe_acc = ((B < 0) * unsafe_mask).sum() / (num_unsafe_elements + 1e-8)
    
        if self.train_dynamics:
            return loss_safe.item(), loss_unsafe.item(), loss_grad.item(),loss_cql.item(), dynamics_loss.item(), avg_safe_B.item(), avg_unsafe_B.item(), safe_acc.item(), unsafe_acc.item(), avg_random_cbf  
        else: 
            return loss_safe.item(), loss_unsafe.item(), loss_grad.item(),loss_cql.item(), avg_safe_B.item(), avg_unsafe_B.item(), safe_acc.item(), unsafe_acc.item(), avg_random_cbf 

 
    def compute_next_states(self, observation, action):
        with torch.no_grad():
            return self.model.forward_next_state(observation,action)
    def sample_random_actions(self, batch_size):
        return 2 * torch.rand(batch_size, self.args.num_action, device=self.device) - 1 
    
    def compute_CQL_loss(self, observations, next_observations, actions, safe_mask):
        observations_safe = observations[safe_mask.reshape(-1,)]

        next_observations_safe = next_observations[safe_mask.reshape(-1,)]
        next_observation_h = self.model.forward_cbf(next_observations_safe)
        all_random_next_h = []
        for _ in range(self.num_action_samples):
            random_actions = self.sample_random_actions(observations_safe.shape[0])
            random_next_states = self.compute_next_states(observations_safe, random_actions)
            random_next_h = self.model.forward_cbf(random_next_states)
            all_random_next_h.append(random_next_h.squeeze())
            
        avg_random_cbf = 0.0 
        if all_random_next_h:
            stacked_h_values = torch.stack(all_random_next_h, dim=1)
            avg_random_cbf = torch.mean(stacked_h_values).item()  #
            combined_h_values = torch.cat([stacked_h_values, next_observation_h.squeeze().unsqueeze(1)], dim=1)
            logsumexp_h = self.temp * torch.logsumexp(combined_h_values/self.temp, dim=1)
           
            if not self.detach:
                cql_actions_term = logsumexp_h - next_observation_h.squeeze()
            else:
                cql_actions_term = logsumexp_h - next_observation_h.squeeze().detach()
            loss_cql_actions = self.w_CQL * torch.mean(cql_actions_term)
            return loss_cql_actions,logsumexp_h, avg_random_cbf 

        
    def compute_gradient_loss(self, observations, actions, safe_mask):
        # Forward pass CBF
        observations.requires_grad = True
        B = self.model.forward_cbf(observations).reshape(-1, 1)
        # print(B.shape)
        # print(B.sum().shape)
        grad_b = torch.autograd.grad(B, observations,grad_outputs=torch.ones_like(B),retain_graph=True)[0] 

        
        with torch.no_grad():   ##dont want gradient of CBF to propagate into the dynamics
            x_dot = self.model.forward_x_dot(observations, actions)

        b_dot = torch.einsum('bo,bo->b',grad_b,x_dot).reshape(-1,1)#compute dot product between grad B and x_dot in order to get b_dot

        gradient = b_dot + 1*B
        loss_grad_vector = self.w_grad * F.relu(self.eps_grad - gradient) * safe_mask
        num_grad_elements = safe_mask.sum()
        loss_grad = loss_grad_vector.sum() / (num_grad_elements + 1e-8)


        return loss_grad
    '''
B.shape torch.Size([128, 1])
next_observations.shape torch.Size([128, 72])
grad_b shape torch.Size([128, 72])
x_dot shape torch.Size([128, 72])
bdot shape torch.Size([128, 1])
b_dot.shape torch.Size([128, 1])
loss_grad_vector.shape torch.Size([128, 1])
'''
    def validate(self):
        self.model.eval()
            
        valloader_iter = iter(self.val_dataset)
        total_loss_safe = 0.0
        total_loss_unsafe = 0.0
        total_loss_grad = 0.0
        total_loss_cql=0.0
        total_dynamics_loss = 0.0
        total_avg_safe_B = 0.0
        total_avg_unsafe_B = 0.0
        total_safe_acc = 0.0
        total_unsafe_acc = 0.0
        total_avg_random_cbf = 0.0  #

        print("\nStarting validation...")
        # with torch.no_grad():##dont disable grad because need gradient through the barrier. 
        for step in range(self.eval_steps):
            batch = next(valloader_iter)
            observations, next_observations, actions, _, costs, done = [b.to(torch.float32).to(self.device) for b in batch]
            debug_results = self.debug_action_coherence([observations, next_observations, actions, _, costs, done])#added this now for debugging
            
            if self.train_dynamics:
                loss_safe, loss_unsafe, loss_grad,loss_cql, dynamics_loss, avg_safe_B, avg_unsafe_B, safe_acc, unsafe_acc, avg_random_cbf = self.compute_loss(  # - added avg_random_cbf
                    observations, next_observations, actions, costs, training_bool=False
                )
                total_dynamics_loss += dynamics_loss
            else:
                loss_safe, loss_unsafe, loss_grad, loss_cql, avg_safe_B, avg_unsafe_B, safe_acc, unsafe_acc, avg_random_cbf = self.compute_loss(  # - added avg_random_cbf
                    observations, next_observations, actions, costs, training_bool=False
                )
            
            total_loss_safe += loss_safe
            total_loss_unsafe += loss_unsafe
            total_loss_grad += loss_grad
            total_loss_cql+=loss_cql
            total_avg_safe_B += avg_safe_B
            total_avg_unsafe_B += avg_unsafe_B
            total_safe_acc += safe_acc
            total_unsafe_acc += unsafe_acc
            total_avg_random_cbf += avg_random_cbf  #

        # Calculate averages
        avg_loss_safe = total_loss_safe / self.eval_steps
        avg_loss_unsafe = total_loss_unsafe / self.eval_steps
        avg_loss_grad = total_loss_grad / self.eval_steps
        avg_loss_cql = total_loss_cql /self.eval_steps
        avg_dynamics_loss = total_dynamics_loss / self.eval_steps if self.train_dynamics else 0.0
        avg_safe_B = total_avg_safe_B / self.eval_steps
        avg_unsafe_B = total_avg_unsafe_B / self.eval_steps
        avg_random_cbf = total_avg_random_cbf / self.eval_steps  #
        
        total_cbf_loss = avg_loss_safe + avg_loss_unsafe + avg_loss_grad #+ avg_loss_cql##TODO ADD HERE LIPCHITZ LOSS 
        total_loss = total_cbf_loss + (avg_dynamics_loss if self.train_dynamics else 0.0)
        
        avg_safe_acc = total_safe_acc / self.eval_steps
        avg_unsafe_acc = total_unsafe_acc / self.eval_steps

        log_dict = {
            "test_loss_safe": avg_loss_safe,
            "test_loss_unsafe": avg_loss_unsafe,
            "test_loss_grad": avg_loss_grad,
            "test_loss_cql": avg_loss_cql,
            "test_cbf_loss": total_cbf_loss,
            "test_avg_safe_B": avg_safe_B,
            "test_avg_unsafe_B": avg_unsafe_B,
            "test_safe_acc": avg_safe_acc,
            "test_unsafe_acc": avg_unsafe_acc,
            "test_total_loss": total_loss,
            "test_avg_random_cbf": avg_random_cbf  #
        }
        
        if self.train_dynamics:
            log_dict["val_dynamics_loss"] = avg_dynamics_loss
            
        wandb.log(log_dict)

        # Print validation results
        print("\nValidation Results:")
        print(f"Average Safe Loss: {avg_loss_safe:.4f}")
        print(f"Average Unsafe Loss: {avg_loss_unsafe:.4f}")
        print(f"Average cql Loss: {avg_loss_cql:.4f}")
        print(f"Average Gradient Loss: {avg_loss_grad:.4f}")
        if self.train_dynamics:
            print(f"Average Dynamics Loss: {avg_dynamics_loss:.4f}")
        print(f"Average Safe B Value: {avg_safe_B:.4f}")
        print(f"Average Unsafe B Value: {avg_unsafe_B:.4f}")
        print(f"Safe Accuracy: {avg_safe_acc:.4f}")
        print(f"Unsafe Accuracy: {avg_unsafe_acc:.4f}")       
        print(f"Total Loss: {total_loss:.4f}")

        self.model.train()
            
        return total_loss, avg_safe_acc, avg_unsafe_acc
                    
    def train(self):
        trainloader_iter = iter(self.train_dataset)
        lowest_eval_loss = float("inf")
        
        # Setup model save path
        base_path = f"examples/research/models/{self.args.task}_{self.random_value}"
        os.makedirs(base_path, exist_ok=True)

        print("\nStarting training combined CBF and dynamics...")
        for step in trange(self.train_steps, desc="Training"):
            batch = next(trainloader_iter)
            observations, next_observations, actions, _, costs, done = [b.to(torch.float32).to(self.device) for b in batch]
            
            if self.train_dynamics:
                loss_safe, loss_unsafe, loss_grad,loss_cql, dynamics_loss, avg_safe_B, avg_unsafe_B, safe_acc, unsafe_acc, avg_random_cbf = self.compute_loss(  # - added avg_random_cbf
                    observations, next_observations, actions, costs, training_bool=True
                )
                
                # Log training metrics
                log_dict = {
                    "train_loss_safe": loss_safe,
                    "train_loss_unsafe": loss_unsafe,
                    "train_loss_grad": loss_grad,
                    "train_loss_cql":loss_cql,
                    "train_dynamics_loss": dynamics_loss,
                    "train_cbf_loss": loss_safe + loss_unsafe + loss_grad + loss_cql,
                    "train_total_loss": loss_safe + loss_unsafe + loss_grad + dynamics_loss + loss_cql,
                    "train_avg_safe_B": avg_safe_B,
                    "train_avg_unsafe_B": avg_unsafe_B,
                    "train_safe_acc": safe_acc,
                    "train_unsafe_acc": unsafe_acc,
                    "train_avg_random_cbf": avg_random_cbf,  #
                    "step": step
                }
            else:
                loss_safe, loss_unsafe, loss_grad,loss_cql, avg_safe_B, avg_unsafe_B, safe_acc, unsafe_acc, avg_random_cbf = self.compute_loss(  # - added avg_random_cbf
                    observations, next_observations, actions, costs, training_bool=True
                )
                
                # Log training metrics
                log_dict = {
                    "train_loss_safe": loss_safe,
                    "train_loss_unsafe": loss_unsafe,
                    "train_loss_grad": loss_grad,
                    "train_loss_cql":loss_cql,
                    "train_cbf_loss": loss_safe + loss_unsafe + loss_grad,
                    "train_total_loss": loss_safe + loss_unsafe + loss_grad,
                    "train_avg_safe_B": avg_safe_B,
                    "train_avg_unsafe_B": avg_unsafe_B,
                    "train_safe_acc": safe_acc,
                    "train_unsafe_acc": unsafe_acc,
                    "train_avg_random_cbf": avg_random_cbf,  #
                    "step": step
                }
                
            wandb.log(log_dict)

            if (step+1) % self.train_steps == 0:
                model_save_path = os.path.join(base_path, "combined_model_laststep.pth")
                torch.save(self.model.state_dict(), model_save_path)
                hyperparameters = {
                            "task": self.args.task,
                            "seed": self.args.seed,
                            "cbf_hidden_dim": self.args.cbf_hidden_dim,
                            "dynamics_hidden_dim": self.args.dynamics_hidden_dim,
                            "cbf_num_layers": self.args.cbf_num_layers,
                            "dynamics_num_layers": self.args.dynamics_num_layers,
                            "batch_size": self.args.batch_size,
                            "learning_rate": self.lr,
                            "dynamics_learning_rate": self.dynamics_lr,
                            "eps_safe": self.eps_safe,
                            "eps_unsafe": self.eps_unsafe,
                            "eps_grad": self.eps_grad,
                            "w_safe": self.w_safe,
                            "w_unsafe": self.w_unsafe,
                            "w_grad": self.w_grad,
                            "train_dynamics": self.train_dynamics,
                            "dt": self.dt,
                            "num_action":self.args.num_action,
                            "state_dim":self.args.state_dim,
                            "best_safe acc":val_avg_safe_acc,
                            "best_unsafe_acc":val_avg_unsafe_acc
                        }
                hyperparameters_path = os.path.join(base_path, "hyperparameters.json") #f"
                    # Save the hyperparameters to a JSON file
                with open(hyperparameters_path, 'w') as f:
                    json.dump(hyperparameters, f, indent=4)
                print(f"\last combined model saved at step {step} with eval loss {total_eval_loss:.4f}")
                print(f"Hyperparameters saved to {hyperparameters_path}")
                
            if (step+1) % self.eval_every_n_steps == 0:
                total_eval_loss,val_avg_safe_acc, val_avg_unsafe_acc = self.validate()
                if "Hopper" in self.args.task:
                    thresh=0.86
                else:
                    thresh=0.8
                print("thresh",thresh)
                if (total_eval_loss < lowest_eval_loss) and (val_avg_safe_acc>thresh) and (val_avg_unsafe_acc>thresh):
                    lowest_eval_loss = total_eval_loss
                    
                    # Save combined model
                    model_save_path = os.path.join(base_path, "combined_model.pth")
                    torch.save(self.model.state_dict(), model_save_path)
                    hyperparameters = {
                            "task": self.args.task,
                            "seed": self.args.seed,
                            "cbf_hidden_dim": self.args.cbf_hidden_dim,
                            "dynamics_hidden_dim": self.args.dynamics_hidden_dim,
                            "cbf_num_layers": self.args.cbf_num_layers,
                            "dynamics_num_layers": self.args.dynamics_num_layers,
                            "batch_size": self.args.batch_size,
                            "learning_rate": self.lr,
                            "dynamics_learning_rate": self.dynamics_lr,
                            "eps_safe": self.eps_safe,
                            "eps_unsafe": self.eps_unsafe,
                            "eps_grad": self.eps_grad,
                            "w_safe": self.w_safe,
                            "w_unsafe": self.w_unsafe,
                            "w_grad": self.w_grad,
                            "train_dynamics": self.train_dynamics,
                            "dt": self.dt,
                            "num_action":self.args.num_action,
                            "state_dim":self.args.state_dim,
                            "final_safe acc":val_avg_safe_acc,
                            "final_unsafe_acc":val_avg_unsafe_acc
                        }
                    hyperparameters_path = os.path.join(base_path, "hyperparameters.json") #f"
                    # Save the hyperparameters to a JSON file
                    with open(hyperparameters_path, 'w') as f:
                        json.dump(hyperparameters, f, indent=4)
                    print(f"\nBest combined model saved at step {step} with eval loss {total_eval_loss:.4f}")
                    print(f"Hyperparameters saved to {hyperparameters_path}")
                    
    def setup_optimizer(self):
        # Create parameter groups with different learning rates if needed
        if self.train_dynamics:
            # Identify which parameters belong to CBF vs dynamics
            cbf_params = list(self.model.cbf.parameters())
            dynamics_params = list(self.model.f.parameters()) + list(self.model.g.parameters())
            
            self.optim = torch.optim.Adam([
                {'params': cbf_params, 'lr': self.lr, 'weight_decay': 2e-5},
                {'params': dynamics_params, 'lr': self.dynamics_lr, 'weight_decay': 1e-5}
            ])
        else:
            # Only optimize CBF parameters
            self.optim = torch.optim.Adam(self.model.cbf.parameters(), lr=self.lr, weight_decay=2e-5)
        
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        return self.model
    
    def debug_action_coherence(self, test_batch):
        """
        Test if CBF distinguishes between coherent vs random action patterns
        """
        observations, next_observations, actions, _, costs, done = test_batch
        safe_mask = (costs <= 0).reshape(-1, 1)
        observations_safe = observations[safe_mask.reshape(-1,)]
        dataset_actions = actions[safe_mask.reshape(-1,)]
        
        # 1. Dataset actions (coherent, goal-directed)
        dataset_next_states = self.compute_next_states(observations_safe, dataset_actions)
        cbf_dataset_actions = self.model.forward_cbf(dataset_next_states)
        
        # 2. Smoothed random actions (partially coherent)
        alpha = 0.7  # Mix with dataset actions
        partial_random = alpha * dataset_actions + (1-alpha) * self.sample_random_actions(observations_safe.shape[0])
        partial_next_states = self.compute_next_states(observations_safe, partial_random)
        cbf_partial_random = self.model.forward_cbf(partial_next_states)
        
        # 3. Pure random actions (chaotic)
        pure_random = self.sample_random_actions(observations_safe.shape[0])
        pure_random_states = self.compute_next_states(observations_safe, pure_random)
        cbf_pure_random = self.model.forward_cbf(pure_random_states)
        
        # 4. Reversed actions (anti-coherent)
        reversed_actions = -dataset_actions  # Opposite direction
        reversed_next_states = self.compute_next_states(observations_safe, reversed_actions)
        cbf_reversed = self.model.forward_cbf(reversed_next_states)
        
        results = {
            "cbf_dataset": torch.mean(cbf_dataset_actions).item(),
            "cbf_partial_random": torch.mean(cbf_partial_random).item(),
            "cbf_pure_random": torch.mean(cbf_pure_random).item(),
            "cbf_reversed": torch.mean(cbf_reversed).item()
        }
        
        wandb.log({
            "coherence_dataset": results["cbf_dataset"],
            "coherence_partial": results["cbf_partial_random"],
            "coherence_random": results["cbf_pure_random"],
            "coherence_reversed": results["cbf_reversed"]
        })
        
        print(f"Action Coherence: Dataset={results['cbf_dataset']:.4f} > Partial={results['cbf_partial_random']:.4f} > Random={results['cbf_pure_random']:.4f}, Reversed={results['cbf_reversed']:.4f}")
        
        return results
    
@pyrallis.wrap()
def main(args: BCTrainConfig):
    cfg, old_cfg = asdict(args), asdict(BCTrainConfig())
    differing_values = {key: cfg[key] for key in cfg if cfg[key] != old_cfg[key]}
    cfg = asdict(BC_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)
    

    # args.seed=7 ##i changed the seed
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    import gymnasium as gym
    env = gym.make(args.task)
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)
     
    process_bc_dataset(data, args.cost_limit, args.gamma, "all")
    
    # Set model parameters
    
    args.num_action = env.action_space.shape[0]
    args.state_dim = env.observation_space.shape[0]
    
    #For transition-level split (original behavior):
    trainloader = DataLoader(TransitionDataset(data, split='train', trajectory_split=False), 
                            batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    valloader = DataLoader(TransitionDataset(data, split='val', trajectory_split=False), 
                        batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
        

    ###For trajectory-level split (completely separate trajectories):
    # trainloader = DataLoader(TransitionDataset(data, split='train', trajectory_split=True), 
    #                         batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    # valloader = DataLoader(TransitionDataset(data, split='val', trajectory_split=True), 
    #                     batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    # Create the combined model
    
    combined_model = CombinedCBFDynamics(
        num_action=args.num_action,
        state_dim=args.state_dim,
        cbf_hidden_dim=args.cbf_hidden_dim,
        dynamics_hidden_dim=args.dynamics_hidden_dim,
        cbf_num_layers=args.cbf_num_layers,
        dynamics_num_layers=args.dynamics_num_layers,
        dt=0.1
    )
    

    # Create trainer
    trainer = CombinedCBFTrainer(
        model=combined_model,
        lr=1e-5,
        device=args.device,
        
        train_steps=args.train_steps,##change 50k hopper and 15k swimmer
        
        eval_every_n_steps=300,
        train_dataset=trainloader,
        val_dataset=valloader,
        eval_steps=20,
        args=args,
        
        eps_safe=args.eps_safe,
        eps_unsafe=args.eps_unsafe, 
        eps_grad=args.eps_grad,
        w_safe=args.w_safe,
        w_unsafe=args.w_unsafe,
        w_grad=args.w_grad,
        lambda_lip=args.lambda_lip,
        
        train_dynamics=True,
        dynamics_lr=1e-4,
        w_CQL=args.cql,
        num_action_samples=args.num_action_samples_cql,
        temp=args.temp,
        detach=args.detach
    )
    
    # Train models
    trainer.train()
    
    # Optionally extract individual models
    #  can be useful if you need to use them separately later
    standalone_cbf = combined_model.get_cbf_model()
    standalone_dynamics = combined_model.get_dynamics_model()
    
    # Save individual models if needed
    base_path = "examples/research/models"
    torch.save(standalone_cbf.state_dict(), f"{base_path}extracted_cbf_model_task{args.task}_seed{args.seed}.pth")
    torch.save(standalone_dynamics.state_dict(), f"{base_path}extracted_dynamics_model_task{args.task}_seed{args.seed}.pth")
    
    wandb.finish()


if __name__ == "__main__":
    main()
    

# python examples/research/check/trainer.py --task OfflineHopperVelocityGymnasium-v1  --cql 0.1 --temp 1 --detach True --batch_size 256 --device="mps" --num_action_samples_cql 10 --seed 7 --train_steps 50000 --w_grad 2
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 1 --temp 1 --detach True --batch_size 256 --device="mps" --num_action_samples_cql 10 --seed 7 --w_grad 2 --train_steps 15000