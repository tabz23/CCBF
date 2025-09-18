import torch
import torch.nn as nn

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Creates a multi-layer perceptron with the specified sizes and activations.
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layer = nn.Linear(sizes[j], sizes[j + 1])
        layers += [layer, act()]
    return nn.Sequential(*layers)

class CombinedCBFDynamics(nn.Module):
    """
    A combined model that includes both a Control Barrier Function (CBF) and dynamics model.
    This allows for joint training and single checkpoint saving.
    """
    def __init__(
        self,
        num_action,
        state_dim,
        cbf_hidden_dim=128,
        dynamics_hidden_dim=64,
        cbf_num_layers=3,
        dynamics_num_layers=3,
        dt=0.1,
        **kwargs
    ):
        super().__init__()
        
        # Store model parameters
        
        self.num_action = num_action
        self.state_dim = state_dim
        self.cbf_hidden_dim = cbf_hidden_dim
        self.dynamics_hidden_dim = dynamics_hidden_dim
        self.cbf_num_layers = cbf_num_layers
        self.dynamics_num_layers = dynamics_num_layers
        self.dt = dt
        
        # CBF network - outputs a scalar value
        self.cbf = mlp(
            [state_dim] + cbf_num_layers * [cbf_hidden_dim] + [1], 
            activation=nn.ReLU, 
            output_activation=nn.Tanh
        )
        
        # Dynamics networks - f(x) and g(x) for affine dynamics
        self.f = mlp(
            [state_dim] + dynamics_num_layers * [dynamics_hidden_dim] + [state_dim], 
            activation=nn.ReLU
        )
        
        self.g = mlp(
            [state_dim] + dynamics_num_layers * [dynamics_hidden_dim] + [state_dim * num_action], 
            activation=nn.ReLU
        )
    
    def forward_cbf(self, state):
        """Forward pass for CBF."""
        return self.cbf(state)
    
    def forward_dynamics(self, state):
        """Forward pass for dynamics, returns f(x) and g(x)."""
        return self.f(state), self.g(state)
    
    def forward_x_dot(self, state, action):
        """Compute state derivative using the dynamics model."""
        f, g = self.forward_dynamics(state)
        gu = torch.einsum('bsa,ba->bs', g.view(g.shape[0], self.state_dim, self.num_action), action)
        x_dot = f + gu
        return x_dot
    
    def forward_next_state(self, state, action):
        """Predict next state using the dynamics model."""
        return self.forward_x_dot(state, action) * self.dt + state
    
    def get_cbf_model(self):
        """Returns a standalone CBF model."""
        standalone_cbf = CBF(
            num_action=self.num_action,
            state_dim=self.state_dim,
            hidden_dim=self.cbf_hidden_dim,
            num_layers=self.cbf_num_layers,
            dt=self.dt
        )
        standalone_cbf.cbf = self.cbf
        return standalone_cbf
    
    def get_dynamics_model(self):
        """Returns a standalone dynamics model."""
        standalone_dynamics = AffineDynamics(
            num_action=self.num_action,
            state_dim=self.state_dim,
            hidden_dim=self.dynamics_hidden_dim,
            num_layers=self.dynamics_num_layers,
            dt=self.dt
        )
        standalone_dynamics.f = self.f
        standalone_dynamics.g = self.g
        return standalone_dynamics


# Keep the original classes for backward compatibility
class CBF(nn.Module):
    def __init__(
        self,        
        num_action,
        state_dim,
        hidden_dim=128,
        num_layers=3,
        dt=0.1):
        
        super().__init__()
        
        self.num_action = num_action
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dt = dt
        self.cbf = mlp([self.state_dim] + num_layers*[self.hidden_dim] + [1], activation=nn.ReLU, output_activation=nn.Tanh)
        
    def forward(self, state):
        return self.cbf(state)


class AffineDynamics(nn.Module):
    def __init__(
        self,
        num_action,
        state_dim,
        hidden_dim=64,
        num_layers=3,
        dt=0.1):
        super().__init__()
        
        self.num_action = num_action
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dt = dt
        
        self.f = mlp([self.state_dim] + num_layers*[self.hidden_dim] + [self.state_dim], activation=nn.ReLU)
        self.g = mlp([self.state_dim] + num_layers*[self.hidden_dim] + [self.state_dim*self.num_action], activation=nn.ReLU)
        
    def forward(self, state):
        return self.f(state), self.g(state)
    
    def forward_x_dot(self, state, action):
        f, g = self.forward(state)
        gu = torch.einsum('bsa,ba->bs', g.view(g.shape[0], self.state_dim, self.num_action), action)
        x_dot = f + gu
        return x_dot
    
    def forward_next_state(self, state, action):
        return self.forward_x_dot(state, action) * self.dt + state