# Install JAX.
!pip install --upgrade jax
!pip install --upgrade jaxlib
!pip install --upgrade trax

# Make sure the Colab Runtime is set to Accelerator: TPU.
import requests
import os
if 'TPU_DRIVER_MODE' not in globals():
  url = 'http://' + os.environ['COLAB_TPU_ADDR'].split(':')[0] + ':8475/requestversion/tpu_driver0.1-dev20191206'
  resp = requests.post(url)
  TPU_DRIVER_MODE = 1

# The following is required to use TPU Driver as JAX's backend.
from jax.config import config
config.FLAGS.jax_xla_backend = "tpu_driver"
config.FLAGS.jax_backend_target = "grpc://" + os.environ['COLAB_TPU_ADDR']
print(config.FLAGS.jax_backend_target)


!pip install --upgrade -q sentencepiece
!pip install --upgrade -q gin

from tensorflow.compat.v1.io.gfile import GFile
import gin
import os
import jax
import trax
from trax.data import inputs

import numpy as np
import jax.numpy as jnp

from scipy.special import softmax

from sentencepiece import SentencePieceProcessor



file = open('/content/t.txt')
text = file.read()

text = text.strip()

import sentencepiece as spm
spm.SentencePieceTrainer.train('--input=t.txt --model_prefix=m --vocab_size=12000')
sp = spm.SentencePieceProcessor()
sp.load('m.model')



IDS = sp.EncodeAsIds(text)
IDS = np.asarray(IDS, dtype=np.int32)
print("Number of tokens:", IDS.shape[0])
PAD_AMOUNT = 256 * 256 - len(IDS)



# Set up the data pipeline.
def my_inputs(n_devices):
  while True:
    inputs = []
    mask = []
    pad_amounts = np.random.choice(PAD_AMOUNT, n_devices)
    for i in range(n_devices):
      inputs.append(np.pad(IDS, (pad_amounts[i], PAD_AMOUNT - pad_amounts[i]),
                            mode='constant'))
      mask.append(np.pad(np.ones_like(IDS, dtype=np.float32),
                          (pad_amounts[i], PAD_AMOUNT - pad_amounts[i]),
                          mode='constant'))
    inputs = np.stack(inputs)
    mask = np.stack(mask)
    yield (inputs, inputs, mask)

print("(device count, tokens per device) = ",
      next(my_inputs(trax.fastmath.device_count()))[0].shape)



# Configure hyperparameters.
gin.parse_config("""
import trax.layers
import trax.models
import trax.optimizers
import trax.data.inputs
import trax.supervised.trainer_lib

# Parameters that will vary between experiments:
# ==============================================================================
train.model = @trax.models.ReformerLM
# Our model will have 6 layers, alternating between the LSH attention proposed
# in the Reformer paper and local attention within a certain context window.
n_layers = 6
attn_type = [
  @trax.layers.SelfAttention,
  @LSHSelfAttention,
  @trax.layers.SelfAttention,
  @LSHSelfAttention,
  @trax.layers.SelfAttention,
  @LSHSelfAttention,
  ]
share_qk = False  # LSH attention ignores this flag and always shares q & k
n_heads = 2
attn_kv = 64
dropout = 0.05
n_tokens = 65536

# Parameters for multifactor:
# ==============================================================================
multifactor.constant = 0.01
multifactor.factors = 'constant * linear_warmup * cosine_decay'
multifactor.warmup_steps = 100
multifactor.steps_per_cycle = 900

# Parameters for Adam:
# ==============================================================================
Adam.weight_decay_rate=0.0
Adam.b1 = 0.86
Adam.b2 = 0.92
Adam.eps = 1e-9

# Parameters for SelfAttention:
# ==============================================================================
trax.layers.SelfAttention.attention_dropout = 0.05
trax.layers.SelfAttention.chunk_len = 64
trax.layers.SelfAttention.n_chunks_before = 1
trax.layers.SelfAttention.n_parallel_heads = 1

# Parameters for LSHSelfAttention:
# ==============================================================================
LSHSelfAttention.attention_dropout = 0.0
LSHSelfAttention.chunk_len = 64
LSHSelfAttention.n_buckets = [64, 128]
LSHSelfAttention.n_chunks_after = 0
LSHSelfAttention.n_chunks_before = 1
LSHSelfAttention.n_hashes = 1
LSHSelfAttention.n_parallel_heads = 1
LSHSelfAttention.predict_drop_len = 128
LSHSelfAttention.predict_mem_len = 1024

# Parameters for ReformerLM:
# ==============================================================================
ReformerLM.attention_type = %attn_type
ReformerLM.d_attention_key = %attn_kv
ReformerLM.d_attention_value = %attn_kv
ReformerLM.d_model = 256
ReformerLM.d_ff = 512
ReformerLM.dropout = %dropout
ReformerLM.ff_activation = @trax.layers.Relu
ReformerLM.max_len = %n_tokens
ReformerLM.mode = 'train'
ReformerLM.n_heads = %n_heads
ReformerLM.n_layers = %n_layers
ReformerLM.vocab_size = 12000
ReformerLM.axial_pos_shape = (256,256)
ReformerLM.d_axial_pos_embs= (64,192)
""")



# Set up a Trainer.
output_dir = os.path.expanduser('/content')

#!rm -f ~/train_dir/model.pkl.gz  # Remove old model

trainer = trax.supervised.Trainer(
    model=trax.models.ReformerLM,
    loss_fn=trax.layers.CrossEntropyLoss(),
    optimizer=trax.optimizers.Adam,
    lr_schedule=trax.lr.multifactor(),
    inputs=trax.data.inputs.Inputs(my_inputs),
    output_dir=output_dir)




# Run one training step, to make sure the model fits in memory.
# The first time trainer.train_epoch is called, it will JIT the entire network
# architecture, which takes around 2 minutes. The JIT-compiled model is saved
# so subsequent runs will be much faster than the first.
trainer.train_epoch(n_steps=1, n_eval_steps=1)




# Train for 600 steps total
# The first ~20 steps are slow to run, but after that it reaches steady-state
# speed. This will take at least 30 minutes to run to completion, but can safely
# be interrupted by selecting "Runtime > Interrupt Execution" from the menu.
# The language model won't be exceptionally good when trained for just a few
# steps and with minimal regularization. However, we can still sample from it to
# see what it learns.
trainer.train_epoch(n_steps=9, n_eval_steps=1)
for _ in range(21):
  trainer.train_epoch(n_steps=10, n_eval_steps=1)





 # As we report in the Reformer paper, increasing the number of hashing rounds
# helps with quality. We can even increase the number of hashing rounds at
# evaluation time only.

gin.parse_config("""LSHSelfAttention.n_hashes = 4""")





# Load the trained Reformer in 'predict' mode
model = trax.models.ReformerLM(mode='predict')
model.init_from_file(os.path.join(output_dir,'model.pkl.gz'),
                     weights_only=True)

# Sample from ReformerLM
output_token_ids = trax.supervised.decoding.autoregressive_sample(
    model, temperature=0.0)

# Decode token IDs
# Reformer outputed a batch with one item, we access it using [0]
# tolist() converts from int64 to int, the type SentencePiece expects
sp.DecodeIds(output_token_ids[0].tolist()) 
