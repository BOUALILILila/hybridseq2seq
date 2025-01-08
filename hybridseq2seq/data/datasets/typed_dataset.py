# original code: https://github.com/RobertCsordas/transformer_generalization/blob/master/dataset/text/typed_text_dataset.py
from typing import Dict, List, Any

import numpy as np

from .dataset import TextDataset, IndexTable, TextDatasetCache

from ...utils import get_logger

logger = get_logger(__name__)


class TypedTextDatasetCache(TextDatasetCache):
    """Dataset cache with types."""

    def build(
        self,
        index_table: IndexTable,
        in_sentences: List[str],
        out_sentences: List[str],
        types: List[int],
        type_names: List[str],
        syntax_parser,
        syntax_parser_type: str,
        syn_trees_in_sentences: List[str] = None,
        split_punctuation: bool = True,
        allow_any_word: bool = False,
        is_uncased: bool = True,
    ):
        super().build(
            index_table,
            in_sentences,
            out_sentences,
            syntax_parser=syntax_parser,
            syntax_parser_type=syntax_parser_type,
            syn_trees_in_sentences=syn_trees_in_sentences,
            split_punctuation=split_punctuation,
            allow_any_word=allow_any_word,
            is_uncased=is_uncased,
        )
        self.types = types
        self.type_names = type_names
        return self

    def state_dict(self) -> Dict[str, Any]:
        res = super().state_dict()
        res["types"] = self.types
        res["type_names"] = self.type_names
        return res

    def load_state_dict(self, state: Dict[str, Any]):
        super().load_state_dict(state)

        self.types = state["types"]
        self.type_names = state["type_names"]


class TypedTextDataset(TextDataset):
    """Dataset with types."""

    _cache: TypedTextDatasetCache
    static_data: Dict[str, TypedTextDatasetCache] = {}

    def __init__(
        self,
        sets: List[str] = ["train"],
        split_type: List[str] = ["default"],
        cache_dir: str = "./cache/",
        syntax_parser: str = None,
        shared_vocabulary: bool = False,
        add_sos_token: bool = True,
        add_eos_token: bool = True,
        split_punctuation: bool = True,
        allow_any_word: bool = False,
        is_uncased: bool = True,
    ):
        super().__init__(
            sets,
            split_type,
            cache_dir,
            syntax_parser,
            shared_vocabulary,
            add_sos_token,
            add_eos_token,
            split_punctuation,
            allow_any_word,
            is_uncased,
        )

    def load_cache_file(self, file) -> TypedTextDatasetCache:
        return TypedTextDatasetCache.load(file)

    def build_cache(self) -> TypedTextDatasetCache:
        raise NotImplementedError()

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.my_indices[item]
        features = self.get_seqs(index)

        return {
            "in_seq_tokens": features["in_seq_tokens"],
            "in_seq_ids": features["in_seq_ids"],
            "out_seq_ids": features["out_seq_ids"],
            "in_dist_matrix": features["in_dist_matrix"],
            "in_len": len(features["in_seq_ids"]),
            "out_len": len(features["out_seq_ids"]),
            "type": self._cache.types[index],
        }
