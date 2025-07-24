# Config file for hyperparameters

batch_size = 16
epochs = 50
# PPO hyperparameters
clip_epsilon = 0.2          # Clipping parameter for PPO
lr = 3e-4                   # Learning rate
gamma = 0.99                # Discount factor
gae_lambda = 0.95           # Lambda for Generalized Advantage Estimation (GAE)
num_epochs = 10             # Number of epochs per PPO update
batch_size = 64             # Mini-batch size for each epoch
value_coef = 0.5            # Weight for value loss
entropy_coef = 0.01         # Weight for entropy bonus (encourages exploration)

mini_batch = 1024     # transitions per mini‑batch update
total_updates = 1000  # training iterations (outer loop)