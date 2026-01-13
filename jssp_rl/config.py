# Config file for hyperparameters

# PPO / training
epochs         = 1   
num_epochs     = 1
batch_size     = 32         # mini-batch transitions per update 
lr             = 3e-4      
clip_epsilon   = 0.2
gamma          = 0.99
gae_lambda     = 0.95
value_coef     = 0.5
entropy_coef   = 0.005      

mini_batch     = 1024
total_updates  = 1000
VAL_LIMIT      = 50
BEST_OF_K      = 10          
PROGRESS_EVERY = 10
LOG_BOTH       = True
SAVE_GANTT_FOR_BEST = True

# Reward / environment parameters
w1 = 1.0
w2 = 1.0
w3 = 1.0
w4 = 1.0

alpha_idle  = 0.0
delta_bonus = 0.0