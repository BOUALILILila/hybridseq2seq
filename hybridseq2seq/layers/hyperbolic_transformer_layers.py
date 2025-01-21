import torch
from torch import nn

from .hyperbolic_linear_layer import PoincareLinear


class HyperbolicTransformerIntermediate(nn.Module):
    # config: intermediate_size, hidden_size
    def __init__(self, config, manifold):
        super().__init__()
        self.manifold = manifold
        self.dense = PoincareLinear(
            manifold, config.hyperbolic_hidden_size, config.hyperbolic_intermediate_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return hidden_states


class HyperbolicTransformerOutput(nn.Module):
    # config: intermediate_size, hidden_size, layer_norm_eps, hidden_dropout_prob
    def __init__(self, config, manifold):
        super().__init__()
        self.manifold = manifold
        self.dense = PoincareLinear(
            manifold, config.hyperbolic_intermediate_size, config.hyperbolic_hidden_size
        )
        self.dropout = nn.Dropout(config.hyperbolic_hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.manifold.mobius_fn_apply(
            lambda x: self.dropout(x), hidden_states
        )
        hidden_states = self.manifold.mobius_add(input_tensor, hidden_states)
        return hidden_states
