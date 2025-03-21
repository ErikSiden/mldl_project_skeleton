import wandb

# 1. Start a W&B run
wandb.init(project = 'gpt3')

# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 0.01

# Model training code here

# 3. Log metrics over time to visualize performance
for i in range(10):
    wandb.log({"loss": loss})
    