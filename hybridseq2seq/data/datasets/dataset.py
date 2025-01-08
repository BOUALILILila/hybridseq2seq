# original code: https://github.com/RobertCsordas/transformer_generalization/blob/master/dataset/text/text_dataset.py
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ...utils import get_logger
from ..vocabulary import WordVocabulary
from ..parsers import get_syntax_parser

logger = get_logger(__name__)

IndexTable = Dict[str, Dict[str, List[int]]]
VERSION = 0.0


class TextDatasetCache:
    """Cache for a given dataset."""

    version: int
    in_sentences: List[List[int]]
    out_sentences: List[List[int]]
    in_dist_matrices: List[np.ndarray]
    # out_dist_matrices: List[np.ndarray]

    index_table: IndexTable  # the different splits of the dataset
    in_vocabulary: WordVocabulary
    out_vocabulary: WordVocabulary

    len_histogram: Dict[float, int]  # stats

    max_in_len: int
    max_out_len: int

    def build(
        self,
        index_table: IndexTable,
        in_sentences: List[str],
        out_sentences: List[str],
        syntax_parser,
        syntax_parser_type: str,
        syn_trees_in_sentences: List[str] = None,
        split_punctuation: bool = True,
        allow_any_word: bool = False,
        is_uncased: bool = True,
    ):
        """Create the cache.

        Parameters
        ----------
            index_table : IndexTable
                Dict of indices of all dataset splits sequentially.
                e.g., train: 0..2999, test: 3000..3546, valid: 3547..5540
            in_sentences : List[str]
                List of input sentences in the dataset.
            out_sentences : List[str]
                List of output sentences in the dataset.
            split_punctuation: bool, default = True
                Split sentences into tokens based on punctuation if True else on whitespaces.
            syntax_parser :
                The syntax parser for computing structural distance in the source sequence.
            syn_trees_in_sentences :
                The list of gold syntax trees from Syntax-COGS if syntax_parser type is gold.

        Returns
        -------
            TextDatasetCache
        """
        self.version = VERSION
        self.index_table = index_table
        self.syntax_parser_type = syntax_parser_type

        logger.info("Constructing vocabularies")
        self.in_vocabulary = WordVocabulary(
            in_sentences,
            split_punctuation=split_punctuation,
            allow_any_word=allow_any_word,
            is_uncased=is_uncased,
        )
        self.out_vocabulary = WordVocabulary(
            out_sentences,
            split_punctuation=split_punctuation,
            allow_any_word=allow_any_word,
            is_uncased=is_uncased,
        )

        # self.in_sentences = [self.in_vocabulary(s) for s in in_sentences] # encode input sentences (indices)
        # self.out_sentences = [self.out_vocabulary(s) for s in out_sentences] # encode output sentences (indices)
        self.in_sentences = [
            self.in_vocabulary.split_sentence(s) for s in in_sentences
        ]  # split sentence to tokens
        self.out_sentences = [
            self.out_vocabulary.split_sentence(s) for s in out_sentences
        ]  # split sentence to tokens

        logger.info("Calculating length statistics")
        counts, bins = np.histogram(
            [len(i) + len(o) for i, o in zip(self.in_sentences, self.out_sentences)]
        )
        self.sum_len_histogram = {k: v for k, v in zip(bins.tolist(), counts.tolist())}

        counts, bins = np.histogram([len(i) for i in self.in_sentences])
        self.in_len_histogram = {k: v for k, v in zip(bins.tolist(), counts.tolist())}

        counts, bins = np.histogram([len(o) for o in self.out_sentences])
        self.out_len_histogram = {k: v for k, v in zip(bins.tolist(), counts.tolist())}

        self.max_in_len = max(len(l) for l in self.in_sentences)
        self.max_out_len = max(len(l) for l in self.out_sentences)

        logger.info("Constructing structural distance matrices")
        self.in_dist_matrices = []
        self.lem_in_sentences = []
        if self.syntax_parser_type == "gold":
            for st in tqdm(syn_trees_in_sentences, desc="Gold Syntactic parsing"):
                dist, lem_s = syntax_parser.get_distance_matrix(st)
                self.in_dist_matrices.append(dist)
                self.lem_in_sentences.append(lem_s)
        else:
            for s in tqdm(in_sentences, desc="Syntactic parsing"):
                dist, lem_s = syntax_parser.get_distance_matrix(s)
                self.in_dist_matrices.append(dist)
                self.lem_in_sentences.append(lem_s)

        logger.info("Dataset cache built")
        return self

    def state_dict(self) -> Dict[str, Any]:
        """Construct state dict for saving."""
        return {
            "version": self.version,
            "syntax_parser_type": self.syntax_parser_type,
            "index": self.index_table,
            "in_sentences": self.in_sentences,
            "lemmatized_in_sentences": self.lem_in_sentences,
            "out_sentences": self.out_sentences,
            "in_voc": self.in_vocabulary.state_dict(),
            "out_voc": self.out_vocabulary.state_dict(),
            "in_dist_matrices": self.in_dist_matrices,
            # "out_dist_matrices": self.out_dist_matrices,
            "max_in_len": self.max_in_len,
            "max_out_len": self.max_out_len,
            "in_len_histogram": self.in_len_histogram,
            "sum_len_histogram": self.sum_len_histogram,
            "out_len_histogram": self.out_len_histogram,
        }

    def load_state_dict(self, data: Dict[str, Any]):
        """Load a recovered state dict."""
        self.version = data.get("version", -1)
        self.syntax_parser_type = data.get(
            "syntax_parser_type", "predicted"
        )  # compat with old cache
        if self.version != VERSION:
            return
        self.index_table = data["index"]
        self.in_vocabulary = WordVocabulary(None)
        self.out_vocabulary = WordVocabulary(None)
        self.in_vocabulary.load_state_dict(data["in_voc"])
        self.out_vocabulary.load_state_dict(data["out_voc"])
        self.in_sentences = data["in_sentences"]
        self.lem_in_sentences = data["lemmatized_in_sentences"]
        self.out_sentences = data["out_sentences"]
        self.in_dist_matrices = data["in_dist_matrices"]
        # self.out_dist_matrices = data["out_dist_matrices"]
        self.max_in_len = data["max_in_len"]
        self.max_out_len = data["max_out_len"]
        self.in_len_histogram = data["in_len_histogram"]
        self.out_len_histogram = data["out_len_histogram"]
        self.sum_len_histogram = data["sum_len_histogram"]

        logger.info("State restored from cache successfully.")

    def save(self, fn: str):
        """Save the current state of the cache."""
        torch.save(self.state_dict(), fn)

    @classmethod
    def load(cls, fn: str):
        """Load a saved state dict."""
        res = cls()
        try:
            logger.info("Loading cached state from: %s", fn)
            data = torch.load(fn)
        except:
            logger.info("Failed to load cache file.")
            res.version = -1
            return res

        res.load_state_dict(data)

        return res


class TextDataset(torch.utils.data.Dataset):
    """Dataset base class."""

    static_data: Dict[str, TextDatasetCache] = {}

    def __init__(
        self,
        sets: List[str] = ["train"],
        split_type: List[str] = ["simple"],
        cache_dir: str = "./cache/",
        syntax_parser: str = None,
        shared_vocabulary: bool = False,
        add_sos_token: bool = True,
        add_eos_token: bool = True,
        split_punctuation: bool = True,
        allow_any_word: bool = False,
        is_uncased: bool = True,
    ):
        super().__init__()

        self.syntax_parser, self.syntax_parser_type = get_syntax_parser(syntax_parser)

        self.add_sos_token = add_sos_token
        self.add_eos_token = add_eos_token

        self.split_punctuation = split_punctuation
        self.allow_any_word = allow_any_word
        self.is_uncased = is_uncased

        self.cache_dir = os.path.join(cache_dir, self.__class__.__name__)
        os.makedirs(self.cache_dir, exist_ok=True)

        assert isinstance(sets, List)
        assert isinstance(split_type, List)

        # self._cache = TextDataset.static_data.get(self.__class__.__name__)
        # just_loaded = self._cache is None
        # if just_loaded:
        self._cache = self._load_dataset()
        TextDataset.static_data[self.__class__.__name__] = self._cache

        if shared_vocabulary:
            self.in_vocabulary = self._cache.in_vocabulary + self._cache.out_vocabulary
            self.out_vocabulary = self.in_vocabulary
            # self.in_remap = self.in_vocabulary.mapfrom(self._cache.in_vocabulary)
            # self.out_remap = self.out_vocabulary.mapfrom(self._cache.out_vocabulary)
        else:
            self.in_vocabulary = self._cache.in_vocabulary
            self.out_vocabulary = self._cache.out_vocabulary

        # if just_loaded:
        for k, t in self._cache.index_table.items():
            logger.info(
                "%s: split %s data: %s",
                self.__class__.__name__,
                k,
                ", ".join([f"{k}: {len(v)}" for k, v in t.items()]),
            )
        logger.info(
            "%s: vocabulary sizes: in: %s, out: %s",
            self.__class__.__name__,
            len(self._cache.in_vocabulary),
            len(self._cache.out_vocabulary),
        )
        logger.info(
            "%s: max input length: %s, max output length: %s",
            self.__class__.__name__,
            self._cache.max_in_len,
            self._cache.max_out_len,
        )
        logger.info(
            "%s sum length histogram: %s",
            self.__class__.__name__,
            self.hist_to_text(self._cache.sum_len_histogram),
        )
        logger.info(
            "%s in length histogram: %s",
            self.__class__.__name__,
            self.hist_to_text(self._cache.in_len_histogram),
        )
        logger.info(
            "%s out length histogram: %s",
            self.__class__.__name__,
            self.hist_to_text(self._cache.out_len_histogram),
        )
        # end if

        self.my_indices = []
        for t in split_type:
            for s in sets:
                self.my_indices += self._cache.index_table[t][s]

        self.shared_vocabulary = shared_vocabulary

    def build_cache(self) -> TextDatasetCache:
        raise NotImplementedError()

    def build_and_save_cache(self, cache_file: str) -> TextDatasetCache:
        res = self.build_cache()
        logger.info(f"Saving dataset into cache: {cache_file}")
        res.save(cache_file)
        return res

    def load_cache_file(self, file) -> TextDatasetCache:
        return TextDatasetCache.load(file)

    def _load_dataset(self):
        """Load dataset from saved cache file if exsits or build the cache and save it."""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, "cache.pth")

        if os.path.isfile(cache_file):
            logger.info(f"Cache file found in path: {cache_file}")
            res = self.load_cache_file(cache_file)
            if (
                res.version == VERSION
                and res.syntax_parser_type == self.syntax_parser_type
            ):
                return res
            else:
                if res.syntax_parser_type == self.syntax_parser_type:
                    logger.warn(
                        f"{self.__class__.__name__}: Invalid cache version: {res.version}, current: {VERSION}."
                    )
                    logger.warn(f"Re-Building cache with current version {VERSION}.")
                    return self.build_and_save_cache(cache_file)
                else:  # type error
                    raise ValueError(
                        f"{self.__class__.__name__}: Invalid cache syntax parser type: {res.syntax_parser_type}, current: {self.syntax_parser_type}."
                    )
        else:
            return self.build_and_save_cache(cache_file)

    def hist_to_text(self, histogram: Dict[float, int]) -> str:
        keys = list(sorted(histogram.keys()))
        values = [histogram[k] for k in keys]
        percent = (np.cumsum(values) * (100.0 / sum(histogram.values()))).tolist()
        return ", ".join(
            f"{k:.1f}: {v} (>= {p:.1f}%)" for k, v, p in zip(keys, values, percent)
        )

    def get_seqs(self, abs_index: int) -> Tuple[List[int], List[int]]:
        in_seq_tokens = self._cache.in_sentences[abs_index][:]
        lem_in_seq_tokens = self._cache.lem_in_sentences[abs_index][:]
        out_seq_tokens = self._cache.out_sentences[abs_index][:]
        in_dist_matrix = self._cache.in_dist_matrices[abs_index][:]

        in_seq_ids = self.in_vocabulary(in_seq_tokens)
        out_seq_ids = self.out_vocabulary(out_seq_tokens)

        out_seq_ids += [self.out_vocabulary.eos_idx]
        
        if self.add_sos_token:
            in_seq_ids = [self.in_vocabulary.sos_idx] + in_seq_ids

        in_dist_matrix, lem_in_seq_tokens = self.syntax_parser.manage_sos_token(
            in_dist_matrix,
            lem_in_seq_tokens,
            self.add_sos_token,
            self.in_vocabulary[self.in_vocabulary.sos_idx],
        )

        if self.add_eos_token:
            in_seq_ids += [self.in_vocabulary.eos_idx]

        in_dist_matrix, lem_in_seq_tokens = self.syntax_parser.manage_eos_token(
            in_dist_matrix,
            lem_in_seq_tokens,
            self.add_eos_token,
            self.in_vocabulary[self.in_vocabulary.eos_idx],
        )

        assert len(in_seq_ids) == len(
            lem_in_seq_tokens
        ), f"Source sequence length {in_seq_ids} != Lematized source sequence length {lem_in_seq_tokens}"
        assert (
            len(in_seq_ids) == in_dist_matrix.shape[0]
        ), f"Source sequence length {len(in_seq_ids)} != Source distance shape [{in_dist_matrix.shape[0]}, {in_dist_matrix.shape[1]}]"

        return {
            "in_seq_tokens": lem_in_seq_tokens,  # after lemmatizer for mapping with out_seq
            "in_seq_ids": in_seq_ids,
            "out_seq_ids": out_seq_ids,
            "in_dist_matrix": in_dist_matrix,
            "in_len": len(in_seq_ids),
            "out_len": len(out_seq_ids),
        }

    def __len__(self) -> int:
        return len(self.my_indices)

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
        }

    def get_output_size(self):
        return len(self._cache.out_vocabulary)

    def get_input_size(self):
        return len(self._cache.in_vocabulary)

    # def start_test(self) -> TextSequenceTestState:
    #     return TextSequenceTestState(lambda x: " ".join(self.in_vocabulary(x)),
    #                                  lambda x: " ".join(self.out_vocabulary(x)))

    @property
    def max_in_len(self) -> int:
        return self._cache.max_in_len

    @property
    def max_out_len(self) -> int:
        return self._cache.max_out_len
