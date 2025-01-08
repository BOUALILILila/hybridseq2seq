# Hybrid Semantic-Syntactic Decoding for Compositional Generalization
We propose a Euclidean-Hyperbolic hybrid seq2seq model based on the Transformer architecture for compositional generalization in COGS semantic parsing tasks. The goal of this approach is to inject structural information about the sentence into the attention mechanism at the decoder level, which has been proven ineffective at using structural information. We propose using hyperbolic instead of Euclidean embeddings to encode structural information as recent research demonstrates its odds of capturing complex hierarchical structures with exceptionally high capacity and continuous tree-like properties.

We use a traditional Euclidean encoder model combined with a hybrid decoder model. The attention mechanism in the decoder layers combines usual attention weights computed from Euclidean token representations that capture semantic information, with attention weights computed based on hyperbolic token representations that capture structural information. The hyperbolic token representations encode the syntactic and semantic structures of the source and target sequences in COGS, respectively.

# Setting the Environment
### Using conda
```shell
conda env create -f hybridseq2seq_env.yml
```
### Using poetry
Some dependencies might be missing in the ```pyproject.toml```, including torch (install the appropriate version for the cuda version)
```shell
poetry install
```

# Launching Experiments
### Config file
Below is an example of a ```config.yaml``` file including the main paramters for training/testing a model. Optional settings can be left out if not needed.

```yaml
# Task
task: COGS # the task class 
seed: 42 # for reproducibility

# Data parameters
data:
  cache_dir: /cache # path to the cache directory for COGS dataset
  syntax_parser: gold   # Syntactic parser for computing the source-source distance either: "gold" => use the syntax trees from syntax-cogs or "predicted" (default) => use stanza depparse
  max_seq_length: 128
  uncased_vocab: True
  shared_vocab: True
  add_sos_token: True # add sos token for the source sequence
  add_eos_token: True  # add eos token for the source sequence
  # the target sequence always starts with sos and ends with eos

# Training parameters
## Use torch-lightning (most parameters are for the Trainer)
training:
  devices: 
    - 0 # list of gpu devices (optional)
  output_folder: /out/COGS/v1_cat_pretrain_euc_200_ft_gold # path to the output dir for saving the model ckpts
  last_ckpt: /out/COGS/v1_cat_pretrain_euc_200_ft_gold/last.ckpt # (optional) ckpt to continue training from 
  task: classification
  learner: default
  max_epochs: 20 # number of training epochs
  precision: 32-true
  check_val_every_n_epoch: 6 # this affects the checkpointing as well => save ckpt is only done when validation is completed => you get a ckpt every n epochs after validation. The choice of 6 though is to limit the full training/validation time for the hybrid model which is very slow
  log_every_n_steps: 100
  save_every_n_epochs: 1
  save_top_k: 3 # save the last 3 ckpts only 
  loss:
    name: cross-entropy
  optimizer:
    name: RiemannianAdam
    learning_rate: 1.0e-4
    weight_decay: 0.0001
  train_batch_size: 32
  valid_batch_size: 64
  test_batch_size: 64
  train_num_workers: 4
  valid_num_workers: 4

# Model parameters
model:
  euclidean_weights_path: /out/COGS/euclidean_200/last.ckpt # (optional) when specified, this path indicates the ckpt from which the euclidean weights will be initialized (pre-trained experiments)
  freeze_euclidean_weights: false # when initializing the euclidean weights from a ckpt choose wether to freeze these weights or not
  name: hybrid # "hybrid" model or "euclidean" model
  encoder: euclidean-encoder 
  decoder: hybrid-decoder # "hybrid-decoder" or "euclidean-decoder" model
  tie_input_output_emb: true # tie the input embeddings with the output layer in the decoder
  hidden_size: 256 # euclidean hidden-size
  hyperbolic_hidden_size: 2
  num_attention_heads: 4
  attention_probs_dropout_prob: 0.1
  max_position_embeddings: 512
  layer_norm_eps: 1.0e-12
  hidden_dropout_prob: 0.1
  num_hidden_layers: 3
  intermediate_size: 256
  hidden_act: gelu
  position_embedding_type: sinusoidal # "sinusoidal" or "absolute"(learned absolute embeddings) or "relative_key" or "relative_key_query" (see euclidean_attention.py)
  scale_mode: down # "down" or "up" or "none" check (https://aclanthology.org/2021.emnlp-main.49.pdf)
  init_embeddings: kaiming # "kaiming" or "xavier" or "normal"
  combine_euclidean_hyperbolic_attention_scores: False # True => use the hyperbolic attention scores as bias, False => use the hyperbolic attention weights to compute an new representation which is concatenated with the euclidean one (see section 1.2.4 in docs/hybrid_approach.pdf)
  hyperbolic_attention_score_weight: 1. # if "combine_euclidean_hyperbolic_attention_scores" is True chose the weight alpha of the hyperbolic attention scores 
  semantic_parser_epsilon: 1.0e-3 
  semantic_parser_relation_edge_weight: 0.5
  default_max_distance: 30.
  hyperbolic_scale: 1.0e-2 # for hmds
  generation:
    stop_criteria:
      name: "max-length"
    searcher:
      name: "greedy-search" # "beam-search" not tested
```
### Run script
```python
from hybridseq2seq.tasks import COGSTask
import argparse

parser = argparse.ArgumentParser(description='Gen COGS.')
parser.add_argument('--config_file', type=str) # path to the config.yaml file

args = parser.parse_args()
print("="*20)
print(args)
print("="*20)

task = COGSTask(config_file=args.config_file, no_cuda=False)

print("\n\nTraining...\n")
task.run_train()

print("\n\nTesting...\n")
task.run_test()
```