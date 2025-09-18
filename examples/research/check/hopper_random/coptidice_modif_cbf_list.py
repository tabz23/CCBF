# reference: https://github.com/deepmind/constrained_optidice.git
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from fsrl.utils import DummyLogger, WandbLogger
from torch import distributions as pyd
from torch.distributions.beta import Beta
from torch.nn import functional as F  # noqa
from tqdm.auto import trange  # noqa

from osrl.common.net import EnsembleQCritic, SquashedGaussianMLPActor
from gymnasium.utils import RecordConstructorArgs, seeding
from scipy.optimize import minimize
from cvxopt import matrix, solvers
import cvxpy as cp
import json
from network_ihab import CombinedCBFDynamics
from dataset_ihab import TransitionDataset



def get_f_div_fn(f_type: str):
    """
    Returns a function that computes the provided f-divergence type.
    """
    f_fn = None
    f_prime_inv_fn = None

    if f_type == 'chi2':
        f_fn = lambda x: 0.5 * (x - 1)**2
        f_prime_inv_fn = lambda x: x + 1

    elif f_type == 'softchi':
        f_fn = lambda x: torch.where(x < 1,
                                     x * (torch.log(x + 1e-10) - 1) + 1, 0.5 *
                                     (x - 1)**2)
        f_prime_inv_fn = lambda x: torch.where(x < 0, torch.exp(x.clamp(max=0.0)), x + 1)

    elif f_type == 'kl':
        f_fn = lambda x: x * torch.log(x + 1e-10)
        f_prime_inv_fn = lambda x: torch.exp(x - 1)
    else:
        raise NotImplementedError('Not implemented f_fn:', f_type)

    return f_fn, f_prime_inv_fn


class COptiDICE(nn.Module):
    """
    Offline Constrained Policy Optimization 
    via stationary DIstribution Correction Estimation (COptiDICE)
    
    Args:
        state_dim (int): dimension of the state space.
        action_dim (int): dimension of the action space.
        max_action (float): Maximum action value.
        f_type (str): The type of f-divergence function to use.
        init_state_propotion (float): The proportion of initial states to include in the optimization.
        observations_std (np.ndarray): The standard deviation of the observation space.
        actions_std (np.ndarray): The standard deviation of the action space.
        a_hidden_sizes (list): List of integers specifying the sizes 
                               of the layers in the actor network.
        c_hidden_sizes (list): List of integers specifying the sizes 
                               of the layers in the critic network (nu and chi networks).
        gamma (float): Discount factor for the reward.
        alpha (float): The coefficient for the cost term in the loss function.
        cost_ub_epsilon (float): A small value added to the upper bound on the cost term.
        num_nu (int): The number of critics to use for the nu-network.
        num_chi (int): The number of critics to use for the chi-network.
        cost_limit (int): Upper limit on the cost per episode.
        episode_len (int): Maximum length of an episode.
        device (str): Device to run the model on (e.g. 'cpu' or 'cuda:0'). 
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 f_type: str,
                 init_state_propotion: float,
                 observations_std: np.ndarray,
                 actions_std: np.ndarray,
                 a_hidden_sizes: list = [128, 128],
                 c_hidden_sizes: list = [128, 128],
                 gamma: float = 0.99,
                 alpha: float = 0.5,
                 cost_ub_epsilon: float = 0.01,
                 num_nu: int = 1,
                 num_chi: int = 1,
                 cost_limit: int = 10,
                 episode_len: int = 300,
                 device: str = "cpu"):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.a_hidden_sizes = a_hidden_sizes
        self.c_hidden_sizes = c_hidden_sizes
        self.gamma = gamma
        self.alpha = alpha
        self.cost_ub_epsilon = cost_ub_epsilon
        self.num_nu = num_nu
        self.num_chi = num_chi
        self.cost_limit = cost_limit
        self.episode_len = episode_len
        self.device = device

        self.qc_thres = cost_limit * (1 - self.gamma**self.episode_len) / (
            1 - self.gamma) / self.episode_len
        self.tau = torch.ones(1, requires_grad=True, device=self.device)
        self.lmbda = torch.ones(1, requires_grad=True, device=self.device)
        self.actor = SquashedGaussianMLPActor(self.state_dim, self.action_dim,
                                              self.a_hidden_sizes,
                                              nn.ReLU).to(self.device)
        self.nu_network = EnsembleQCritic(self.state_dim,
                                          0,
                                          self.c_hidden_sizes,
                                          nn.ReLU,
                                          num_q=self.num_nu).to(self.device)
        self.chi_network = EnsembleQCritic(self.state_dim,
                                           0,
                                           self.c_hidden_sizes,
                                           nn.ReLU,
                                           num_q=self.num_chi).to(self.device)

        self.f_fn, self.f_prime_inv_fn = get_f_div_fn(f_type)

        self.init_state_propotion = init_state_propotion
        self.observations_std = torch.tensor(observations_std, device=self.device)
        self.actions_std = torch.tensor(actions_std, device=self.device)

    def _optimal_w(self, observations, next_observations, rewards, costs, done):
        nu_s, _ = self.nu_network.predict(observations, None)
        nu_s_next, _ = self.nu_network.predict(next_observations, None)
        # \hat{e}_{\lambda, \nu}(s,a,s')
        e_nu_lambda = rewards - self._lmbda.detach() * costs
        e_nu_lambda += self.gamma * (1.0 - done) * nu_s_next - nu_s
        # w_{lambda,\nu}^*(s,a)
        w_sa = F.relu(self.f_prime_inv_fn(e_nu_lambda / self.alpha))
        return nu_s, nu_s_next, e_nu_lambda, w_sa

    def update(self, batch):
        observations, next_observations, actions, rewards, costs, done, is_init = batch
        # 1. Learn the optimal distribution
        self._lmbda = F.softplus(self.lmbda)  # lmbda >= 0

        nu_s, nu_s_next, e_nu_lambda, w_sa = self._optimal_w(observations,
                                                             next_observations, rewards,
                                                             costs, done)
        nu_init = nu_s * is_init / self.init_state_propotion
        w_sa_no_grad = w_sa.detach()
        # divergence between distributions of policy & dataset
        Df = self.f_fn(w_sa_no_grad).mean()

        # 1.1 (chi, tau) loss
        if self.cost_ub_epsilon == 0:
            weighted_c = (w_sa_no_grad * costs).mean()
            chi_loss = torch.zeros(1, device=self.device)
            tau_loss = torch.zeros(1, device=self.device)
            D_kl = torch.zeros(1, device=self.device)
            self._tau = F.softplus(self.tau)

        else:
            self._tau = F.softplus(self.tau)
            batch_size = observations.shape[0]

            chi_s, _ = self.chi_network.predict(observations, None)
            chi_s_next, _ = self.chi_network.predict(next_observations, None)
            chi_init = chi_s * is_init / self.init_state_propotion

            ell = (1- self.gamma) * chi_init + \
                    w_sa_no_grad * (costs + self.gamma * (1 - done) * chi_s_next - chi_s)
            logits = ell / self._tau.detach()
            weights = torch.softmax(logits, dim=0) * batch_size
            log_weights = torch.log_softmax(logits, dim=0) + np.log(batch_size)
            D_kl = (weights * log_weights - weights + 1).mean()

            # an upper bound estimation
            weighted_c = (weights * w_sa_no_grad * costs).mean()

            chi_loss = (weights * ell).mean()
            self.chi_optim.zero_grad()
            chi_loss.backward(retain_graph=True)
            self.chi_optim.step()

            tau_loss = self._tau * (self.cost_ub_epsilon - D_kl.detach())
            self.tau_optim.zero_grad()
            tau_loss.backward()
            self.tau_optim.step()

        # 1.2 nu loss
        nu_loss = (1 - self.gamma) * nu_init.mean() + \
            (w_sa * e_nu_lambda - self.alpha * self.f_fn(w_sa)).mean()
        td_error = e_nu_lambda.pow(2).mean()

        self.nu_optim.zero_grad()
        nu_loss.backward(retain_graph=True)
        self.nu_optim.step()

        # 1.3 lambda loss
        lmbda_loss = self._lmbda * (self.qc_thres - weighted_c.detach())

        self.lmbda_optim.zero_grad()
        lmbda_loss.backward()
        self.lmbda_optim.step()

        # 2. Extract policy
        obs_eps = torch.randn_like(observations) * self.observations_std * 0.1
        act_eps = torch.randn_like(actions) * self.actions_std * 0.1

        _, _, dist = self.actor.forward(observations + obs_eps, False, True, True)

        with torch.no_grad():
            _, _, e_nu_lambda, w_sa = self._optimal_w(observations, next_observations,
                                                      rewards, costs, done)

        actor_loss = -(w_sa * dist.log_prob(actions + act_eps).sum(axis=-1)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        stats_loss = {
            "loss/chi_loss": chi_loss.item(),
            "loss/tau_loss": tau_loss.item(),
            "loss/D_kl": D_kl.item(),
            "loss/Df": Df.item(),
            "loss/td_error": td_error.item(),
            "loss/nu_loss": nu_loss.item(),
            "loss/lmbda_loss": lmbda_loss.item(),
            "loss/actor_loss": actor_loss.item(),
            "loss/tau": self._tau.item(),
            "loss/lmbda": self._lmbda.item()
        }
        return stats_loss

    def setup_optimizers(self, actor_lr, critic_lr, scalar_lr):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.nu_optim = torch.optim.Adam(self.nu_network.parameters(), lr=critic_lr)
        self.chi_optim = torch.optim.Adam(self.chi_network.parameters(), lr=critic_lr)
        self.lmbda_optim = torch.optim.Adam([self.lmbda], lr=scalar_lr)
        self.tau_optim = torch.optim.Adam([self.tau], lr=scalar_lr)

    def act(self,
            obs: np.ndarray,
            deterministic: bool = False,
            with_logprob: bool = False):
        """
        Given a single obs, return the action, logp.
        """
        obs = torch.tensor(obs[None, ...], dtype=torch.float32).to(self.device)
        a, logp_a = self.actor.forward(obs, deterministic, with_logprob)
        a = a.data.numpy() if self.device == "cpu" else a.data.cpu().numpy()
        logp_a = logp_a.data.numpy() if self.device == "cpu" else logp_a.data.cpu(
        ).numpy()
        return np.squeeze(a, axis=0), np.squeeze(logp_a)


class COptiDICETrainer:
    """
    COptiDICE trainer
    
    Args:
        model (COptiDICE): The COptiDICE model to train.
        env (gym.Env): The OpenAI Gym environment to train the model in.
        logger (WandbLogger or DummyLogger): The logger to use for tracking training progress.
        actor_lr (float): learning rate for actor
        critic_lr (float): learning rate for critic (nu and chi networks)
        scalar_lr (float, optional): The learning rate for the scalar (tau, lmbda).
        reward_scale (float): The scaling factor for the reward signal.
        cost_scale (float): The scaling factor for the constraint cost.
        device (str): The device to use for training (e.g. "cpu" or "cuda").
    """

    def __init__(self,
                 model: COptiDICE,
                 env: gym.Env,
                 logger: WandbLogger = DummyLogger(),
                 actor_lr: float = 1e-3,
                 critic_lr: float = 1e-3,
                 scalar_lr: float = 1e-3,
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
        self.model.setup_optimizers(actor_lr, critic_lr, scalar_lr)
        
        # self.seed=1
        # print("seed: ",self.seed)
        
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
        
        checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        self.combinedcbfdynamics.load_state_dict(checkpoint)
        self.combinedcbfdynamics.to(self.device)
        self.combinedcbfdynamics.eval()
        # print("loaded cbf")

    def train_one_step(self, batch):
        stats_loss = self.model.update(batch)
        self.logger.store(**stats_loss)

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


    def rollout(self, render=False, plot=False,cbf=True):
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

        obs, info = self.env.reset()
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
        
        
        
