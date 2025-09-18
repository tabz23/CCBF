# Learning-Conservative-Neural-Control-Barrier-Functions-from-Offline-Data
Repository for "Learning Conservative Neural Control Barrier Functions from Offline Data"
# Learning-Conservative-Neural-Control-Barrier-Functions-from-Offline-Data
Repository for the paper "Learning Conservative Neural Control Barrier Functions from Offline Data"
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
```

### Step 3: Verify Installation
```bash
# Test that key packages are installed correctly
python -c "import torch; import gymnasium; import cvxpy; print('Installation successful!')"
```

### Step 4: Run the Code
```bash
# Replace with your main script name
python main.py

# Or if you have a specific training script
python train_ccbf.py

# Or if you have Jupyter notebooks
jupyter notebook
```

## Quick Start Example
```bash
# Example command to run training (adjust as needed)
python train_ccbf.py --env_name SafetyPointGoal1-v0 --epochs 100 --lr 0.001
```

## Citation
If you use this code, please cite our paper:
```bibtex
@article{your_paper_2024,
  title={Learning Conservative Neural Control Barrier Functions from Offline Data},
  author={Your Name and Coauthors},
  journal={Conference/Journal Name},
  year={2024}
}
```

