# Config file for hyperparameters


epochs = 4
# PPO hyperparameters
clip_epsilon = 0.2          # Clipping parameter for PPO
lr = 1e-4                   # Learning rate
gamma = 0.99                # Discount factor
gae_lambda = 0.95           # Lambda for Generalized Advantage Estimation (GAE)
num_epochs = 25            # Number of epochs per PPO update
batch_size = 32            # Mini-batch size for each epoch
value_coef = 0.2            # Weight for value loss
entropy_coef = 0.03         # Weight for entropy bonus (encourages exploration)

mini_batch = 1024     # transitions per mini‑batch update
total_updates = 1000  # training iterations (outer loop)

VAL_LIMIT = 50         # cap validation instances so meta/eval doesn't hang
BEST_OF_K = 10         # try K stochastic rollouts and keep the best (set 1 to disable)
PROGRESS_EVERY = 10    # print progress every N instances
LOG_BOTH = True        # log both greedy and best-of-k

SAVE_GANTT_FOR_BEST = True  # save gantt chart for the best-of-K run
