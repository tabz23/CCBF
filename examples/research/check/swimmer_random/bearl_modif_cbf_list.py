# reference: https://github.com/aviralkumar2907/BEAR
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from fsrl.utils import DummyLogger, WandbLogger
from tqdm.auto import trange  # noqa
from network_ihab import CombinedCBFDynamics
from dataset_ihab import TransitionDataset
from osrl.common.net import (VAE, EnsembleDoubleQCritic, LagrangianPIDController,
                             SquashedGaussianMLPActor)

from scipy.optimize import minimize
from cvxopt import matrix, solvers
import cvxpy as cp
import json

class BEARL(nn.Module):
    """
    Bootstrapping Error Accumulation Reduction with PID Lagrangian (BEARL)
    
    Args:
        state_dim (int): dimension of the state space.
        action_dim (int): dimension of the action space.
        max_action (float): Maximum action value.
        a_hidden_sizes (list): List of integers specifying the sizes 
            of the layers in the actor network.
        c_hidden_sizes (list): List of integers specifying the sizes 
            of the layers in the critic network.
        vae_hidden_sizes (int): Number of hidden units in the VAE. 
            sample_action_num (int): Number of action samples to draw. 
        gamma (float): Discount factor for the reward.
        tau (float): Soft update coefficient for the target networks. 
        beta (float): Weight of the KL divergence term.            
        lmbda (float): Weight of the Lagrangian term.
        mmd_sigma (float): Width parameter for the Gaussian kernel used in the MMD loss.
        target_mmd_thresh (float): Target threshold for the MMD loss.
        num_samples_mmd_match (int): Number of samples to use in the MMD loss calculation.
        PID (list): List of three floats containing the coefficients of the PID controller.
        kernel (str): Kernel function to use in the MMD loss calculation.
        num_q (int): Number of Q networks in the ensemble.
        num_qc (int): Number of cost Q networks in the ensemble.
        cost_limit (int): Upper limit on the cost per episode.
        episode_len (int): Maximum length of an episode.
        start_update_policy_step (int): Number of steps to wait before updating the policy.
        device (str): Device to run the model on (e.g. 'cpu' or 'cuda:0'). 
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 a_hidden_sizes: list = [128, 128],
                 c_hidden_sizes: list = [128, 128],
                 vae_hidden_sizes: int = 64,
                 sample_action_num: int = 10,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 beta: float = 0.5,
                 lmbda: float = 0.75,
                 mmd_sigma: float = 50,
                 target_mmd_thresh: float = 0.05,
                 num_samples_mmd_match: int = 10,
                 PID: list = [0.1, 0.003, 0.001],
                 kernel: str = "gaussian",
                 num_q: int = 1,
                 num_qc: int = 1,
                 cost_limit: int = 10,
                 episode_len: int = 300,
                 start_update_policy_step: int = 20_000,
                 device: str = "cpu"):

        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = self.action_dim * 2
        self.max_action = max_action
        self.a_hidden_sizes = a_hidden_sizes
        self.c_hidden_sizes = c_hidden_sizes
        self.vae_hidden_sizes = vae_hidden_sizes
        self.sample_action_num = sample_action_num
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.lmbda = lmbda
        self.mmd_sigma = mmd_sigma
        self.target_mmd_thresh = target_mmd_thresh
        self.num_samples_mmd_match = num_samples_mmd_match
        self.start_update_policy_step = start_update_policy_step
        self.KP, self.KI, self.KD = PID
        self.kernel = kernel
        self.num_q = num_q
        self.num_qc = num_qc
        self.cost_limit = cost_limit
        self.episode_len = episode_len
        self.device = device
        self.n_train_steps = 0

        ################ create actor critic model ###############
        self.actor = SquashedGaussianMLPActor(self.state_dim, self.action_dim,
                                              self.a_hidden_sizes,
                                              nn.ReLU).to(self.device)
        self.critic = EnsembleDoubleQCritic(self.state_dim,
                                            self.action_dim,
                                            self.c_hidden_sizes,
                                            nn.ReLU,
                                            num_q=self.num_q).to(self.device)
        self.cost_critic = EnsembleDoubleQCritic(self.state_dim,
                                                 self.action_dim,
                                                 self.c_hidden_sizes,
                                                 nn.ReLU,
                                                 num_q=self.num_qc).to(self.device)
        self.vae = VAE(self.state_dim, self.action_dim, self.vae_hidden_sizes,
                       self.latent_dim, self.max_action, self.device).to(self.device)
        self.log_alpha = torch.tensor(0.0, device=self.device)

        self.actor_old = deepcopy(self.actor)
        self.actor_old.eval()
        self.critic_old = deepcopy(self.critic)
        self.critic_old.eval()
        self.cost_critic_old = deepcopy(self.cost_critic)
        self.cost_critic_old.eval()

        self.qc_thres = cost_limit * (1 - self.gamma**self.episode_len) / (
            1 - self.gamma) / self.episode_len
        self.controller = LagrangianPIDController(self.KP, self.KI, self.KD,
                                                  self.qc_thres)

    def _soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        """
        Softly update the parameters of target module towards the parameters
        of source module.
        """
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def _actor_forward(self,
                       obs: torch.tensor,
                       deterministic: bool = False,
                       with_logprob: bool = True):
        """
        Return action distribution and action log prob [optional].
        """
        a, logp = self.actor(obs, deterministic, with_logprob)
        return a * self.max_action, logp

    def vae_loss(self, observations, actions):
        recon, mean, std = self.vae(observations, actions)
        recon_loss = nn.functional.mse_loss(recon, actions)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        loss_vae = recon_loss + self.beta * KL_loss

        self.vae_optim.zero_grad()
        loss_vae.backward()
        self.vae_optim.step()
        stats_vae = {"loss/loss_vae": loss_vae.item()}
        return loss_vae, stats_vae

    def critic_loss(self, observations, next_observations, actions, rewards, done):
        _, _, q1_list, q2_list = self.critic.predict(observations, actions)
        with torch.no_grad():
            batch_size = next_observations.shape[0]
            obs_next = torch.repeat_interleave(next_observations, self.sample_action_num,
                                               0).to(self.device)

            act_targ_next, _ = self.actor_old(obs_next, False, True, False)
            q1_targ, q2_targ, _, _ = self.critic_old.predict(obs_next, act_targ_next)

            q_targ = self.lmbda * torch.min(
                q1_targ, q2_targ) + (1. - self.lmbda) * torch.max(q1_targ, q2_targ)
            q_targ = q_targ.reshape(batch_size, -1).max(1)[0]

            backup = rewards + self.gamma * (1 - done) * q_targ

        loss_critic = self.critic.loss(backup, q1_list) + self.critic.loss(
            backup, q2_list)
        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()

        stats_critic = {"loss/critic_loss": loss_critic.item()}
        return loss_critic, stats_critic

    def cost_critic_loss(self, observations, next_observations, actions, costs, done):
        _, _, qc1_list, qc2_list = self.cost_critic.predict(observations, actions)
        with torch.no_grad():
            batch_size = next_observations.shape[0]
            obs_next = torch.repeat_interleave(next_observations, self.sample_action_num,
                                               0).to(self.device)

            act_targ_next, _ = self.actor_old(obs_next, False, True, False)
            qc1_targ, qc2_targ, _, _ = self.cost_critic_old.predict(
                obs_next, act_targ_next)

            qc_targ = self.lmbda * torch.min(
                qc1_targ, qc2_targ) + (1. - self.lmbda) * torch.max(qc1_targ, qc2_targ)
            qc_targ = qc_targ.reshape(batch_size, -1).max(1)[0]

            backup = costs + self.gamma * qc_targ

        loss_cost_critic = self.cost_critic.loss(
            backup, qc1_list) + self.cost_critic.loss(backup, qc2_list)

        self.cost_critic_optim.zero_grad()
        loss_cost_critic.backward()
        self.cost_critic_optim.step()

        stats_cost_critic = {"loss/cost_critic_loss": loss_cost_critic.item()}
        return loss_cost_critic, stats_cost_critic

    def actor_loss(self, observations):
        for p in self.critic.parameters():
            p.requires_grad = False
        for p in self.cost_critic.parameters():
            p.requires_grad = False
        for p in self.vae.parameters():
            p.requires_grad = False

        _, raw_sampled_actions = self.vae.decode_multiple(
            observations, num_decode=self.num_samples_mmd_match)

        batch_size = observations.shape[0]
        stacked_obs = torch.repeat_interleave(
            observations, self.num_samples_mmd_match,
            0)  # [batch_size*num_samples_mmd_match, obs_dim]
        actor_samples, raw_actor_actions = self.actor(stacked_obs,
                                                      return_pretanh_value=True)
        actor_samples = actor_samples.reshape(batch_size, self.num_samples_mmd_match,
                                              self.action_dim)
        raw_actor_actions = raw_actor_actions.view(batch_size,
                                                   self.num_samples_mmd_match,
                                                   self.action_dim)

        if self.kernel == 'laplacian':
            mmd_loss = self.mmd_loss_laplacian(raw_sampled_actions,
                                               raw_actor_actions,
                                               sigma=self.mmd_sigma)
        elif self.kernel == 'gaussian':
            mmd_loss = self.mmd_loss_gaussian(raw_sampled_actions,
                                              raw_actor_actions,
                                              sigma=self.mmd_sigma)

        q_val1, q_val2, _, _ = self.critic.predict(observations, actor_samples[:, 0, :])
        qc_val1, qc_val2, _, _ = self.cost_critic.predict(observations,
                                                          actor_samples[:, 0, :])
        qc_val = torch.min(qc_val1, qc_val2)
        with torch.no_grad():
            multiplier = self.controller.control(qc_val).detach()
        qc_penalty = ((qc_val - self.qc_thres) * multiplier).mean()
        q_val = torch.min(q_val1, q_val2)

        if self.n_train_steps >= self.start_update_policy_step:
            loss_actor = (-q_val + self.log_alpha.exp() *
                          (mmd_loss - self.target_mmd_thresh)).mean()
        else:
            loss_actor = (self.log_alpha.exp() *
                          (mmd_loss - self.target_mmd_thresh)).mean()
        loss_actor += qc_penalty

        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

        self.log_alpha += self.alpha_lr * self.log_alpha.exp() * (
            mmd_loss - self.target_mmd_thresh).mean().detach()
        self.log_alpha.data.clamp_(min=-5.0, max=5.0)
        self.n_train_steps += 1

        stats_actor = {
            "loss/actor_loss": loss_actor.item(),
            "loss/mmd_loss": mmd_loss.mean().item(),
            "loss/qc_penalty": qc_penalty.item(),
            "loss/lagrangian": multiplier.item(),
            "loss/alpha_value": self.log_alpha.exp().item()
        }

        for p in self.critic.parameters():
            p.requires_grad = True
        for p in self.cost_critic.parameters():
            p.requires_grad = True
        for p in self.vae.parameters():
            p.requires_grad = True
        return loss_actor, stats_actor

    # from https://github.com/Farama-Foundation/D4RL-Evaluations
    def mmd_loss_laplacian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Laplacian kernel for support matching"""
        # sigma is set to 20.0 for hopper, cheetah and 50 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1) / (2.0 * sigma)).exp(),
                              dim=(1, 2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1) / (2.0 * sigma)).exp(),
                              dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1) / (2.0 * sigma)).exp(),
                              dim=(1, 2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    # from https://github.com/Farama-Foundation/D4RL-Evaluations
    def mmd_loss_gaussian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Gaussian Kernel support matching"""
        # sigma is set to 20.0 for hopper, cheetah and 50 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.pow(2)).sum(-1) / (2.0 * sigma)).exp(),
                              dim=(1, 2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(),
                              dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(),
                              dim=(1, 2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def setup_optimizers(self, actor_lr, critic_lr, vae_lr, alpha_lr):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.cost_critic_optim = torch.optim.Adam(self.cost_critic.parameters(),
                                                  lr=critic_lr)
        self.vae_optim = torch.optim.Adam(self.vae.parameters(), lr=vae_lr)
        # self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha_lr = alpha_lr

    def sync_weight(self):
        """
        Soft-update the weight for the target network.
        """
        self._soft_update(self.critic_old, self.critic, self.tau)
        self._soft_update(self.cost_critic_old, self.cost_critic, self.tau)
        self._soft_update(self.actor_old, self.actor, self.tau)

    def act(self,
            obs: np.ndarray,
            deterministic: bool = False,
            with_logprob: bool = False):
        """
        Given a single obs, return the action, logp.
        """
        obs = torch.tensor(obs[None, ...], dtype=torch.float32).to(self.device)
        a, logp_a = self._actor_forward(obs, deterministic, with_logprob)
        a = a.data.numpy() if self.device == "cpu" else a.data.cpu().numpy()
        logp_a = logp_a.data.numpy() if self.device == "cpu" else logp_a.data.cpu(
        ).numpy()
        return np.squeeze(a, axis=0), np.squeeze(logp_a)


class BEARLTrainer:
    """
    BEARL Trainer
    
    Args:
        model (BEARL): The BEARL model to be trained.
        env (gym.Env): The OpenAI Gym environment to train the model in.
        logger (WandbLogger or DummyLogger): The logger to use for tracking training progress.
        actor_lr (float): learning rate for actor
        critic_lr (float): learning rate for critic
        alpha_lr (float): learning rate for alpha
        vae_lr (float): learning rate for vae
        reward_scale (float): The scaling factor for the reward signal.
        cost_scale (float): The scaling factor for the constraint cost.
        device (str): The device to use for training (e.g. "cpu" or "cuda").
    """

    def __init__(self,
                 model: BEARL,
                 env: gym.Env,
                 logger: WandbLogger = DummyLogger(),
                 actor_lr: float = 1e-3,
                 critic_lr: float = 1e-3,
                 alpha_lr: float = 1e-3,
                 vae_lr: float = 1e-3,
                 reward_scale: float = 1.0,
                 cost_scale: float = 1.0,
                 device="cpu",
                 model_path=None,
                 hyperparams_path=None):

        self.model = model
        self.logger = logger
        self.env = env
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.device = device
        self.model.setup_optimizers(actor_lr, critic_lr, vae_lr, alpha_lr)
        
        
        with open(hyperparams_path, 'r') as f:
            hyperparams = json.load(f)


        # Initialize and load the model
        self.combinedcbfdynamics = CombinedCBFDynamics(
            num_action=hyperparams['num_action'],
            state_dim=hyperparams['state_dim'],
            cbf_hidden_dim=hyperparams['cbf_hidden_dim'],
            dynamics_hidden_dim=hyperparams['dynamics_hidden_dim'],
            cbf_num_layers=hyperparams['cbf_num_layers'],
            dynamics_num_layers=hyperparams['dynamics_num_layers'],
            dt=hyperparams['dt']
        )
        # self.seed=1
        # print("seed: ",self.seed)
        checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        self.combinedcbfdynamics.load_state_dict(checkpoint)
        self.combinedcbfdynamics.to(self.device)
        self.combinedcbfdynamics.eval()
        print("loaded cbf")
        
    def train_one_step(self, observations, next_observations, actions, rewards, costs,
                       done):
        """
        Trains the model by updating the VAE, critic, cost critic, and actor.
        """

        # update VAE
        loss_vae, stats_vae = self.model.vae_loss(observations, actions)
        # update critic
        loss_critic, stats_critic = self.model.critic_loss(observations,
                                                           next_observations, actions,
                                                           rewards, done)
        # update cost critic
        loss_cost_critic, stats_cost_critic = self.model.cost_critic_loss(
            observations, next_observations, actions, costs, done)
        # update actor
        loss_actor, stats_actor = self.model.actor_loss(observations)

        self.model.sync_weight()

        self.logger.store(**stats_vae)
        self.logger.store(**stats_critic)
        self.logger.store(**stats_cost_critic)
        self.logger.store(**stats_actor)

    def evaluate(self, eval_episodes):
        """
        Evaluates the performance of the model on a number of episodes.
        """
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for _ in trange(eval_episodes, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout()
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)
        self.model.train()
        return np.mean(episode_rets) / self.reward_scale, np.mean(
            episode_costs) / self.cost_scale, np.mean(episode_lens)


    def rollout(self, render=False, plot=False,cbf=False):
        """
        Evaluates the performance of the model on a single episode.
        """

        obs, info = self.env.reset()
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        
        states_history = []
        nominal_actions_history = []
        safe_actions_history = []
        costs_history = []
        cbf_values_history = []
        
        for step in range(self.model.episode_len):
            nominal_action, _ = self.model.act(obs, True, True)
            if cbf:
                # Convert to tensor for CBF calculations
                current_state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                # print(current_state_tensor.shape)
                current_state_tensor.requires_grad_(True)
                    
                # Get CBF value
                h_x_tensor = self.combinedcbfdynamics.forward_cbf(current_state_tensor)
                h_x = h_x_tensor.item()  # Scalar CBF value
                
                # Get gradient of CBF with respect to state
                gradient_B = torch.autograd.grad(h_x_tensor, current_state_tensor)[0]
                        
                # Get dynamics
                f, g = self.combinedcbfdynamics.forward_dynamics(current_state_tensor)
                
                # Calculate LHS of CBF constraint: ∇B(x)f(x)
                cbf_lie_derivative = torch.einsum("bs,bs->b", f, gradient_B)
                
                # Add gamma(B(x)) term
                right_side = cbf_lie_derivative + h_x_tensor.squeeze(-1)###CHANGE ALPHA HERE CHANGE ALPHA HERE.  LOW ALPHA->SAFER, WORSE REWARD
                right_numpy = right_side.detach().cpu().numpy()
                
                # Calculate ∇B(x)g(x) for the control input term
                grad_b_g = torch.einsum(
                    'bs,bsa->ba', 
                    gradient_B, 
                    g.view(g.shape[0], self.combinedcbfdynamics.state_dim, self.combinedcbfdynamics.num_action)
                )
                grad_b_g_numpy = grad_b_g.detach().cpu().numpy()
                
                # Solve QP to find safe action
                u = cp.Variable(self.combinedcbfdynamics.num_action)
                objective = cp.Minimize(cp.sum_squares(u - nominal_action))
                constraints = [grad_b_g_numpy @ u >= -right_numpy,
                            # u<=1, u>=-1
                            ]
                
                prob = cp.Problem(objective, constraints)
                
                try:
                    # Solve using OSQP solver
                    prob.solve(solver=cp.OSQP)
                    
                    if prob.status in ["optimal", "optimal_inaccurate"]:
                        safe_action = u.value
                    else:
                        # print(f"Warning: QP at step {step} could not find optimal solution (status: {prob.status}), using nominal action")##I COMMENTED THIS OUT FOR NOW
                        safe_action = nominal_action
                except Exception as e:
                    print(f"Error in QP solver at step {step}: {e}")
                    safe_action = nominal_action
                
            # Store data for plotting
            states_history.append(obs.copy())
            nominal_actions_history.append(nominal_action.copy())
            if (cbf):
                safe_actions_history.append(safe_action.copy())
                cbf_values_history.append(h_x)
            
            # Apply safe action to environment
            if cbf:
                obs_next, reward, terminated, truncated, info = self.env.step(safe_action)#CHANGE HERE TO NOMINAL OR SAFE#CHANGE HERE TO NOMINAL OR SAFE#CHANGE HERE TO NOMINAL OR SAFE
            else:
                obs_next, reward, terminated, truncated, info = self.env.step(nominal_action)
            costs_history.append(info["cost"])
            

            # Update state and tracking variables
            obs = obs_next
            episode_ret += reward
            episode_len += 1
            # print(info)
            episode_cost += info["cost"]
            
            if render:
                self.env.render()
                
            if terminated or truncated:
                break
        
        # Create plots if requested
        if plot:
            self._create_plots(
                states_history, 
                nominal_actions_history, 
                safe_actions_history, 
                costs_history, 
                cbf_values_history,
                episode_len
            )
            
        return episode_ret, episode_len, episode_cost

    def _create_plots(self, states, nominal_actions, safe_actions, costs, cbf_values, episode_len):
        """
        Creates and displays plots of rollout data.
        
        Args:
            states: List of states
            nominal_actions: List of nominal actions from BC policy
            safe_actions: List of safe actions after CBF modification
            costs: List of costs at each step
            cbf_values: List of CBF values at each step
            episode_len: Length of the episode
        """
        import matplotlib.pyplot as plt
        
        # Convert lists to numpy arrays for easier plotting
        states = np.array(states)
        nominal_actions = np.array(nominal_actions)
        safe_actions = np.array(safe_actions)
        costs = np.array(costs)
        cbf_values = np.array(cbf_values)
        
        # Create time steps
        time_steps = np.arange(episode_len)
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(10, 8))
        
        # Plot 1: Actions (nominal vs safe)
        action_dim = nominal_actions.shape[1]
        ax1 = fig.add_subplot(3, 1, 1)
        for i in range(1):
            ax1.plot(time_steps, nominal_actions[:, i], '--', label=f'Nominal Action {i+1}')
            ax1.plot(time_steps, safe_actions[:, i], '--', label=f'Safe Action {i+1}')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Action Value')
        ax1.set_title('Nominal vs Safe Actions')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Costs
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(time_steps, costs, '-o', label='Step Cost')
        ax2.plot(time_steps, np.cumsum(costs), '-', label='Cumulative Cost')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Cost')
        ax2.set_title('Cost per Time Step and Cumulative Cost')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: CBF Values
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(time_steps, cbf_values, '-b', label='CBF Value')
        ax3.axhline(y=0, color='r', linestyle='--', label='Safety Threshold')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('CBF Value')
        ax3.set_title('Control Barrier Function Values')
        ax3.legend()
        ax3.grid(True)
        
        # Add a tight layout to prevent overlapping
        plt.tight_layout()
        
        # Save the figure if needed
        plt.savefig('/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/figs/cbf_rollout_analysis.png', dpi=300, bbox_inches='tight')
        
        # Display the figure
        plt.show()
        
           
