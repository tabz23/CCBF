
# Learning Conservative Neural Control Barrier Functions from Offline Data

## Installation


### Step 1: Clone the Repository
```bash
git clone https://github.com/tabz23/CCBF.git
cd CCBF
```

### Step 2: Create and Activate Conda Environment
```bash
# Create environment from the provided environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate CCBF
pip install -e .
```

### Step 3: Verify Installation
```bash
# Test that key packages are installed correctly
python -c 'import torch; import gymnasium; import cvxpy; print("Installation successful!")'
```



## Reproduce paper results
```bash
# Hopper BC
python "examples/research/check/hopper_random/eval_bc_cbf_modif_list.py" --path "logs/OfflineHopperVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-0912"  --eval_episode 20 --device cpu
# Hopper BC-Safe
python "examples/research/check/hopper_random/eval_bc_cbf_modif_list.py" --path "logs/OfflineHopperVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-70bc" --eval_episode 20 --device cpu
# Hopper BCQL
python "examples/research/check/hopper_random/eval_bcql_modif_list.py" --path "logs/OfflineHopperVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-257f" --eval_episodes 20  --device cpu
# Hopper BEARL
python "examples/research/check/hopper_random/eval_bearl_modif_list.py" --path "logs/OfflineHopperVelocityGymnasium-v1-cost-20/BEARL_cost20-7857" --eval_episodes 20  --device cpu
# Hopper COptiDICE
python "examples/research/check/hopper_random/eval_coptidice_modif_list.py" --path "logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3"  --eval_episodes 20  --device cpu

# Swimmer BC
python "examples/research/check/swimmer_random/eval_bc_cbf_modif_list.py"  --path "logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567"  --eval_episode 20 --device cpu
# Swimmer BC-Safe
python "examples/research/check/swimmer_random/eval_bc_cbf_modif_list.py"  --path "logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180" --eval_episode 20 --device cpu
# Swimmer BCQL
python "examples/research/check/swimmer_random/eval_bcql_modif_list.py" --path "logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5" --eval_episodes 20
# Swimmer BEARL
python "examples/research/check/swimmer_random/eval_bearl_modif_list.py" --path "logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BEARL_cost20_seed10-f1fd/BEARL_cost20_seed10-f1fd" --eval_episodes 20
# Swimmer COptiDICE
python "examples/research/check/Swimmer_random/eval_coptidice_modif_list.py" --path "logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/COptiDICE_cost20-3187/COptiDICE_cost20-3187" --eval_episodes 20
```

## Train CCBF in Hopper and Swimmer
```bash
python examples/research/check/trainer.py --task OfflineHopperVelocityGymnasium-v1  --device="cuda" --cql 0.1 --temp 1 --detach True --batch_size 256  --num_action_samples_cql 10 --seed 7 --train_steps 50000 --w_grad 2
python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  -device="cuda" --cql 1 --temp 0.5 --detach True --batch_size 256 - --num_action_samples_cql 10 --seed 7 --w_grad 2 --train_steps 15000
```
## Debug
Incase DSRL throws the following error: PermissionError: [Errno 13] Permission denied: '/home/...' 
```bash
export DSRL_DATASET_DIR="path"
mkdir -p "path"
```
Incase you face other issues while running the training script due to dataset download issues, please refer to https://github.com/liuzuxin/DSRL.
We have also uploaded the datasets online (https://limewire.com/d/R3dC5#8sa8LzaziD). After donwloading, you can manually move the datasets into ```DSRL_DATASET_DIR``` directory

## Citation
If you use this code, please cite our paper:
```bibtex
@article{tabbara2025learning,
  title={Learning Neural Control Barrier Functions from Offline Data with Conservatism},
  author={Tabbara, Ihab and Sibai, Hussein},
  journal={arXiv preprint arXiv:2505.00908},
  year={2025}
}
```

