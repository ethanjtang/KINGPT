# Forked directly from https://github.com/karpathy/nanogpt
# Config for KINGPT-Chimera (same as Woodpecker variant), trained on combined dataset of puzzle + SF18 selfplay positions
# I will note that the puzzle positions massively outnumber the selfplay positions at 13M vs. 15K

init_from = 'resume'
out_dir = 'out-kingpt'
eval_interval = 50000
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
max_iters = 1000000 # equivalent to training on ~500B tokens (tokens = chars because char-level tokenization, blah blah blah)
lr_decay_iters = 1000000 # make equal to max_iters usually
min_lr = 6e-5 # learning_rate / 10 usually
beta2 = 0.95

