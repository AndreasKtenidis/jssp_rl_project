# Config file for hyperparameters

# Reptile params
meta_iterations = 10
meta_batch_size = 10
inner_steps = 3
inner_lr = 1e-4
meta_lr = 1e-3
inner_update_batch_size = 16
validate_every = 10

# PPO / training
epochs         = 3          # epochs per update
num_epochs     = 90         # total training iterations
batch_size     = 64         # mini-batch transitions per update step
lr             = 3e-4       # typical for PPO+Adam
clip_epsilon   = 0.2
gamma          = 0.99
gae_lambda     = 0.95
value_coef     = 0.5
entropy_coef   = 0.005      # lower to move away from uniform policy

# Reward shapping 
w1=0.2
w2=0.2
w3=0.4
w4=0.2
alpha_idle=-0.1
delta_bonus=1


# 
mini_batch     = 1024
total_updates  = 1000
VAL_LIMIT      = 1
BEST_OF_K      = 10          
PROGRESS_EVERY = 10
LOG_BOTH       = True
SAVE_GANTT_FOR_BEST = True
use_est_boost = True   
est_beta = 20

# Curriculum Learning Configuration
# Phase Name -> (N, M, file_path)
CURRICULUM_PHASES = {
    "Phase_1": {"size": (6, 6),   "file": "Synthetic_instances_6x6_5000.pkl"},
    "Phase_2": {"size": (10, 10), "file": "Synthetic_instances_10x10_5000.pkl"},
    "Phase_3": {"size": (15, 15), "file": "Synthetic_instances_15x15_5000.pkl"},
    "Phase_4": {"size": (20, 15), "file": "Synthetic_instances_20x15_5000.pkl"},
    "Phase_5": {"size": (20, 20), "file": "Synthetic_instances_20x20_5000.pkl"},
    "Phase_6": {"size": (30, 20), "file": "Synthetic_instances_30x20_5000.pkl"},
    "Phase_7": {"size": (50, 20), "file": "Synthetic_instances_50x20_5000.pkl"},
}

# Training Control
ACTIVE_PHASE = "Phase_1"  # Default
RUN_FULL_CURRICULUM = True # Set to True to loop through all phases
