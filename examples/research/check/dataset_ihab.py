from torch.utils.data import DataLoader, Dataset
import torch
import gymnasium as gym
import dsrl
import numpy as np
from torch.utils.data import IterableDataset
import numpy as np
from torch.utils.data import IterableDataset
class TransitionDataset(IterableDataset):
    """
    A dataset of transitions (state, action, reward, next state) used for training RL agents.
    
    Args:
        dataset (dict): A dictionary of NumPy arrays containing the observations, actions, rewards, etc.
        reward_scale (float): The scale factor for the rewards.
        cost_scale (float): The scale factor for the costs.
        state_init (bool): If True, the dataset will include an "is_init" flag indicating if a transition
            corresponds to the initial state of an episode.
        split (str): One of 'train' or 'val', to specify which subset of the data to use.
        train_ratio (float): Ratio of data to be used for training (0.8 means 80% for training and 20% for validation).
        trajectory_split (bool): If True, split by trajectories instead of individual transitions.  
    """

    def __init__(self,
                 dataset: dict,
                 reward_scale: float = 1.0,
                 cost_scale: float = 1.0,
                 state_init: bool = False,
                 split: str = 'train',
                 train_ratio: float = 0.85,
                 trajectory_split: bool = False):  
        self.dataset = dataset
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.state_init = state_init
        self.split = split
        self.train_ratio = train_ratio
        self.trajectory_split = trajectory_split  

        self.dataset_size = self.dataset["observations"].shape[0]
        self.train_size = int(self.dataset_size * self.train_ratio)
        
        self.dataset["done"] = np.logical_or(self.dataset["terminals"], 
                                             self.dataset["timeouts"]).astype(np.float32)
        if self.state_init:
            self.dataset["is_init"] = self.dataset["done"].copy()
            self.dataset["is_init"][1:] = self.dataset["is_init"][:-1]
            self.dataset["is_init"][0] = 1.0

        # Choose splitting strategy based on trajectory_split flag  
        if self.trajectory_split:  
            self._split_by_trajectories()  
        else:  
            self._split_by_transitions()  

    def _split_by_transitions(self):  
        """Original splitting method - by individual transitions"""  
        indices = np.arange(self.dataset_size)
        np.random.shuffle(indices)  # Shuffle the indices

        self.train_indices = indices[:self.train_size]
        self.val_indices = indices[self.train_size:]

        if self.split == 'train':
            self.indices = self.train_indices
        elif self.split == 'val':
            self.indices = self.val_indices
        else:
            raise ValueError("split must be one of 'train' or 'val'")

    def _split_by_trajectories(self):  
        """New splitting method - by complete trajectories"""  
        # Find trajectory boundaries using done flags  
        done_flags = self.dataset["done"]  
        trajectory_ends = np.where(done_flags == 1)[0]  
        
        # Create trajectory start and end indices  
        trajectory_starts = np.concatenate([[0], trajectory_ends[:-1] + 1])  
        trajectory_ends = trajectory_ends  
        
        num_trajectories = len(trajectory_starts)  
        print(f"Found {num_trajectories} trajectories in dataset")  
        
        # Split trajectories into train/val  
        traj_indices = np.arange(num_trajectories)  
        np.random.shuffle(traj_indices)  
        
        num_train_trajs = int(num_trajectories * self.train_ratio)  
        train_traj_indices = traj_indices[:num_train_trajs]  
        val_traj_indices = traj_indices[num_train_trajs:]  
        
        print(f"Train trajectories: {len(train_traj_indices)}, Val trajectories: {len(val_traj_indices)}")  
        
        # Get transition indices for each split  
        self.train_indices = []  
        for traj_idx in train_traj_indices:  
            start = trajectory_starts[traj_idx]  
            end = trajectory_ends[traj_idx]  
            self.train_indices.extend(range(start, end + 1))  
        
        self.val_indices = []  
        for traj_idx in val_traj_indices:  
            start = trajectory_starts[traj_idx]  
            end = trajectory_ends[traj_idx]  
            self.val_indices.extend(range(start, end + 1))  
        
        self.train_indices = np.array(self.train_indices)  
        self.val_indices = np.array(self.val_indices)  
        
        print(f"Train transitions: {len(self.train_indices)}, Val transitions: {len(self.val_indices)}")  
        
        if self.split == 'train':  
            self.indices = self.train_indices  
        elif self.split == 'val':  
            self.indices = self.val_indices  
        else:  
            raise ValueError("split must be one of 'train' or 'val'")  

    def get_dataset_states(self):
        """
        Returns the proportion of initial states in the dataset, 
        as well as the standard deviations of the observation and action spaces.
        """
        init_state_propotion = self.dataset["is_init"].mean()
        obs_std = self.dataset["observations"].std(0, keepdims=True)
        act_std = self.dataset["actions"].std(0, keepdims=True)
        return init_state_propotion, obs_std, act_std

    def __prepare_sample(self, idx):
        observations = self.dataset["observations"][idx, :]
        next_observations = self.dataset["next_observations"][idx, :]
        actions = self.dataset["actions"][idx, :]
        rewards = self.dataset["rewards"][idx] * self.reward_scale
        costs = self.dataset["costs"][idx] * self.cost_scale
        done = self.dataset["done"][idx]
        if self.state_init:
            is_init = self.dataset["is_init"][idx]
            return observations, next_observations, actions, rewards, costs, done, is_init
        return observations, next_observations, actions, rewards, costs, done

    def __iter__(self):
        """
        Iterates over the dataset, yielding samples based on the chosen split.
        """
        while True:
            idx = np.random.choice(self.indices)
            yield self.__prepare_sample(idx)


def test_TransitionDataset():
    # Create dummy data for testing
    dummy_data = {
        "observations": np.random.randn(100, 4),  # 100 samples, 4 features (e.g., state size)
        "next_observations": np.random.randn(100, 4),
        "actions": np.random.randn(100, 2),  # 100 samples, 2 actions
        "rewards": np.random.randn(100),  # 100 rewards
        "costs": np.random.randn(100),  # 100 costs
        "terminals": np.random.choice([0, 1], size=100),  # 0 or 1 indicating if episode ended
        "timeouts": np.random.choice([0, 1], size=100),  # 0 or 1 indicating if time limit reached
    }

    # Initialize the TransitionDataset for train and validation
    train_dataset = TransitionDataset(dataset=dummy_data, reward_scale=1.0, cost_scale=1.0, state_init=True, split='train')
    val_dataset = TransitionDataset(dataset=dummy_data, reward_scale=1.0, cost_scale=1.0, state_init=True, split='val')

    # Check the size of both datasets
    print(f"Train dataset size: {len(train_dataset.indices)}")
    print(f"Validation dataset size: {len(val_dataset.indices)}")

    # Create DataLoaders for both train and val datasets
    train_dataloader = DataLoader(train_dataset, batch_size=2)
    val_dataloader = DataLoader(val_dataset, batch_size=2)

    # Iterate over the DataLoaders and print a sample batch for both train and val
    print("\nTraining Batch Example:")
    for batch in train_dataloader:
        print("Observations:", batch[0])
        print("Next Observations:", batch[1])
        print("Actions:", batch[2])
        print("Rewards:", batch[3])
        print("Costs:", batch[4])
        print("Done:", batch[5])

        break  # Stop after one batch for testing

    print("\nValidation Batch Example:")
    for batch in val_dataloader:
        print("Observations:", batch[0])
        print("Next Observations:", batch[1])
        print("Actions:", batch[2])
        print("Rewards:", batch[3])
        print("Costs:", batch[4])
        print("Done:", batch[5])

        break  # Stop after one batch for testing