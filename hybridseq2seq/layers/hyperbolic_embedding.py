from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt import ManifoldParameter

from ..data import SemanticParser
from ..hmds import hmds
from ..manifolds import PoincareBall


class HMDSEmbeddingFromCrossDistances:
    def __init__(
        self,
        epsilon: float,
        relation_edge_weight: float,
        default_max_distance: float,
        add_sos_token: bool,
        hmds_scale: float,
        hyperbolic_embedding_size: int,
        manifold: PoincareBall,
    ) -> None:
        self.manifold = manifold
        self.hmds_scale = hmds_scale
        self.hyperbolic_embedding_size = hyperbolic_embedding_size
        self.semantic_parser = SemanticParser(
            epsilon=epsilon,
            relation_edge_weight=relation_edge_weight,
            default_max_distance=default_max_distance,
            pad_token="<pad>",
            add_sos_token=add_sos_token,
        )

    def embed(
        self,
        step: int,
        previous_cross_distance_matrix: torch.Tensor,  # (bs, step-1, step-1)
        source_distance_matrix: torch.Tensor,  # (bs, src_seq_length, src_seq_length)
        source_tokens_list: List[List[str]],  # (bs, src_seq_length)
        source_attention_mask: torch.Tensor,  # (bs, src_seq_length)
        previous_target_tokens_list: List[List[str]],  # (bs, step)
        previous_target_hyperbolic_attention_mask: torch.Tensor,  # (bs, step)
        running: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # Compute hyperbolic embeddings using cross distances from syntax (input) and semantic (target)
        with torch.no_grad():
            hyperbolic_cross_embeddings_step = []
            hyperbolic_target_mask_step = []
            cross_distance_matrices_step = []

            bs = previous_target_hyperbolic_attention_mask.shape[0]
            for i in range(bs):
                # Get the cross-distance matrix + mask
                (
                    cross_distance_mat_step,
                    hyperbolic_target_attention_mask,
                ) = self.semantic_parser.get_one_distance_matrix(
                    step=step,
                    previous_cross_distance_matrix=previous_cross_distance_matrix[i],
                    source_distance_matrix=source_distance_matrix[i],
                    source_tokens=source_tokens_list[i],
                    previous_target_tokens=previous_target_tokens_list[i],
                    target_attention_mask_step=previous_target_hyperbolic_attention_mask[
                        i
                    ],
                    ended=None if (running is None) else (not running[i]),
                )
                # hmds to recover the hyperbolic embeddings from the cross-distance matrix and mask
                hyperbolic_cross_hidden_states_step = hmds(
                    distance_matrix=cross_distance_mat_step,
                    mask=torch.cat(
                        [source_attention_mask[i], hyperbolic_target_attention_mask]
                    ),
                    k=self.hyperbolic_embedding_size,
                    scale=self.hmds_scale,
                )
                # Project on the PoincareBall
                hyperbolic_cross_hidden_states_step = (
                    self.manifold.projx_from_hyperboloid(
                        hyperbolic_cross_hidden_states_step
                    )
                )

                hyperbolic_cross_embeddings_step.append(
                    hyperbolic_cross_hidden_states_step
                )
                hyperbolic_target_mask_step.append(hyperbolic_target_attention_mask)
                cross_distance_matrices_step.append(cross_distance_mat_step)

            hyperbolic_cross_embeddings_step = torch.stack(
                hyperbolic_cross_embeddings_step
            )
            hyperbolic_target_mask_step = torch.stack(hyperbolic_target_mask_step)
            cross_distance_matrices_step = torch.stack(cross_distance_matrices_step)
        return (
            hyperbolic_cross_embeddings_step,
            hyperbolic_target_mask_step,
            cross_distance_matrices_step,
        )


# Orginal code https://github.com/mil-tokyo/hyperbolic_nn_plusplus/blob/main/geoopt_plusplus/modules/embedding.py
class PoincareEmbedding(nn.Module):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int, optional): If given, pads the output with the embedding vector at :attr:`padding_idx`
                                         (initialized to zeros) whenever it encounters the index.
        max_norm: Optional[float]
            If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (boolean, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor.
                                 See Notes for more details regarding sparse gradients.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
                         initialized from :math:`\mathcal{N}(0, 1)`

    Shape:
        - Input: :math:`(*)`, LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`

    .. note::
        Keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's :class:`optim.SGD` (`CUDA` and `CPU`),
        :class:`optim.SparseAdam` (`CUDA` and `CPU`) and :class:`optim.Adagrad` (`CPU`)

    .. note::
        With :attr:`padding_idx` set, the embedding vector at
        :attr:`padding_idx` is initialized to all zeros. However, note that this
        vector can be modified afterwards, e.g., using a customized
        initialization method, and thus changing the vector used to pad the
        output. The gradient for this vector from :class:`~torch.nn.Embedding`
        is always zero.

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding = nn.Embedding(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
        >>> embedding(input)
        tensor([[[-0.0251, -1.6902,  0.7172],
                 [-0.6431,  0.0748,  0.6969],
                 [ 1.4970,  1.3448, -0.9685],
                 [-0.3677, -2.7265, -0.1685]],

                [[ 1.4970,  1.3448, -0.9685],
                 [ 0.4362, -0.4004,  0.9400],
                 [-0.6431,  0.0748,  0.6969],
                 [ 0.9124, -2.3616,  1.1151]]])


        >>> # example with padding_idx
        >>> embedding = nn.Embedding(10, 3, padding_idx=0)
        >>> input = torch.LongTensor([[0,2,0,5]])
        >>> embedding(input)
        tensor([[[ 0.0000,  0.0000,  0.0000],
                 [ 0.1535, -2.0309,  0.9315],
                 [ 0.0000,  0.0000,  0.0000],
                 [-0.1655,  0.9897,  0.0635]]])
    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    def __init__(
        self,
        manifold,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        init_std=1e-2,
    ):
        super(PoincareEmbedding, self).__init__()
        self.manifold = manifold
        self.init_std = init_std
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq

        if _weight is None:
            self.weight = ManifoldParameter(
                torch.empty(num_embeddings, embedding_dim), manifold=manifold
            )
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], "Shape of weight does not match num_embeddings and embedding_dim"
            self.weight = ManifoldParameter(_weight, manifold=manifold)
        self.sparse = sparse

    def reset_parameters(self):
        with torch.no_grad():
            direction = torch.randn_like(self.weight)
            direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-7)
            distance = torch.empty(self.num_embeddings, 1).normal_(
                std=self.init_std / self.manifold.c.data.sqrt()
            )
            self.weight.data.copy_(self.manifold.expmap0(direction * distance))
            if self.padding_idx is not None:
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input):
        return F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def extra_repr(self):
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(
        cls,
        embeddings,
        freeze=True,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
    ):
        r"""Creates Embedding instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as ``num_embeddings``, second as ``embedding_dim``.
            freeze (boolean, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``
            padding_idx (int, optional): See module initialization documentation.
            max_norm (float, optional): See module initialization documentation.
            norm_type (float, optional): See module initialization documentation. Default ``2``.
            scale_grad_by_freq (boolean, optional): See module initialization documentation. Default ``False``.
            sparse (bool, optional): See module initialization documentation.

        Examples::

            >>> # FloatTensor containing pretrained weights
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embedding = nn.Embedding.from_pretrained(weight)
            >>> # Get embeddings for index 1
            >>> input = torch.LongTensor([1])
            >>> embedding(input)
            tensor([[ 4.0000,  5.1000,  6.3000]])
        """
        assert (
            embeddings.dim() == 2
        ), "Embeddings parameter is expected to be 2-dimensional"
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        embedding.weight.requires_grad = not freeze
        return embedding
