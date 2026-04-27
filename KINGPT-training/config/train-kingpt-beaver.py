# Forked directly from https://github.com/karpathy/nanogpt
# Config for KINGPT-Beaver, trained on SF18 selfplay game positions only

init_from = 'scratch'
out_dir = 'out-kingpt'
eval_interval = 10000
eval_iters = 200
log_interval = 1000 # don't print too too often

always_save_checkpoint = True

dataset = 'chess-data'
# tokens per iteration = grad_steps * batch_size * block_size
# 524,288 tokens per iteration
gradient_accumulation_steps = 4
batch_size = 1024
block_size = 128 # context of up to 128 previous tokens

# KINGPT -> approx 25M parameters (similar scale to GPT-2)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.05

learning_rate = 6e-4 
max_iters = 50000 # equivalent to training on ~25B characters (tokens since I use char-level tokenization)
# step 50000: train loss 0.0974, val loss 1.7554

# I also trained a massively overfit version (because I am a fool) for 500k iters which achieved 
# step 500000: train loss 0.0957, val loss 2.2518

lr_decay_iters = 50000 # make equal to max_iters usually
min_lr = 6e-5 # learning_rate / 10 usually
beta2 = 0.95

