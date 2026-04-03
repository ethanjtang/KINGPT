init_from = 'scratch'
out_dir = 'out-kingpt'
eval_interval = 500000
eval_iters = 200
log_interval = 1000 # don't print too too often

always_save_checkpoint = True

dataset = 'chess-data'
# tokens per iteration = grad_steps * batch_size * block_size
# 524,288 tokens per iteration
gradient_accumulation_steps = 4
batch_size = 1024
block_size = 128 # context of up to 128 previous tokens

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.05

learning_rate = 6e-4 
max_iters = 1000000 # TODO: change max_iters and lr_decay_iters to an actual large number
lr_decay_iters = 1000000 # make equal to max_iters usually
min_lr = 6e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
