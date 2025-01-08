import pytest

import torch

from ..conftest import (
    get_config,
    prepare_config_and_hidden_states,
    prepare_config_and_inputs,
)


@pytest.fixture
def num_heads_wrong(get_config):
    config = get_config
    config.num_attention_heads = 3
    return config
