# original code: https://github.com/RobertCsordas/transformer_generalization/blob/38f2734f5dcad331d4a9658b73b33889ce607c87/framework/data_structures/vocabulary.py
import re
from typing import List, Union, Optional, Dict, Any, Set

from ..utils import get_logger

logger = get_logger(__name__)


class WordVocabulary:
    """Extract a vocabulary based on words."""

    def __init__(
        self,
        list_of_sentences: Optional[List[Union[str, List[str]]]] = None,
        allow_any_word: bool = False,
        split_punctuation: bool = True,
        special_tokens: Optional[List[str]] = None,
        is_uncased: Optional[bool] = True,
    ):
        """Create a vocabulary from a given list of textual sentences.

        Parameters
        ----------
        list_of_sentences : Optional[List[Union[str, List[str]]]]
            The list of sentences, full text or tokenized into a list of tokens.
        allow_any_word : bool, default = False
            Allow out of vocabulary OOV tokens represented with a special UNK token.
        split_punctuation: bool, default = True
            Split sentences into tokens based on punctuation.
        """
        self.is_uncased = is_uncased

        self.words: Dict[str, int] = {}
        self.inv_words: Dict[int, str] = {}

        # Add special tokens
        # Default
        self._special_tokens = dict()
        for special_token in ["<pad>", "<s>", "</s>"]:
            self._special_tokens[special_token] = self._add_word(special_token)
        # Custom
        if special_tokens is not None:
            for st in special_tokens:
                self._special_tokens[st] = self._add_word(st)

        self.to_save = [
            "words",
            "inv_words",
            "_special_tokens",
            "allow_any_word",
            "split_punctuation",
        ]
        self.allow_any_word = allow_any_word
        self.initialized = False
        self.split_punctuation = split_punctuation

        if list_of_sentences is not None:
            words = set()
            for s in list_of_sentences:
                words |= set(self.split_sentence(s))

            self._add_set(words)
            self.finalize()

    def finalize(self):
        """unkonwn token index for OOV tokens."""
        if self.allow_any_word:
            self._special_tokens["<unk>"] = self._add_word("<unk>")
        self.initialized = True

    def _add_word(self, w: str):
        """Add a word to the vocabulary."""
        if self.is_uncased:
            w = w.lower()

        if w in self.words:
            return self.words[w]

        next_id = len(self.words)
        self.words[w] = next_id
        self.inv_words[next_id] = w
        return next_id

    def _add_set(self, words: Set[str]):
        """Add a set of words to the vocabulary."""
        for w in sorted(words):
            self._add_word(w)

    def _process_word(self, w: str) -> int:
        """Get the index of a word in the vocabulary."""
        if self.is_uncased:
            w = w.lower()
        res = self.words.get(w, self.unk_idx)
        assert (
            res != self.unk_idx
        ) or self.allow_any_word, f"WARNING: unknown word: '{w}'"
        return res

    def _process_index(self, i: int) -> str:
        """Get the word corresponding to an index in the vocabulary."""
        res = self.inv_words.get(i, None)
        if res is None:
            return f"<!INV: {i}!>"
        return res

    def __getitem__(self, item: Union[int, str]) -> Union[str, int]:
        if isinstance(item, int):
            return self._process_index(item)
        else:
            return self._process_word(item)

    def split_sentence(self, sentence: Union[str, List[str]]) -> List[str]:
        """Tokenize a given sentence, if it is not already tokenized, based on punctuation or whitespaces."""
        if isinstance(sentence, list):
            # Already tokenized.
            return sentence

        if self.split_punctuation:
            return re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
        else:
            return [x for x in sentence.split(" ") if x]

    def sentence_to_indices(self, sentence: Union[str, List[str]]) -> List[int]:
        """Find the correponding indices to each word (token) in a sentence."""
        assert self.initialized
        words = self.split_sentence(sentence)
        return [self._process_word(w) for w in words]

    def indices_to_sentence(self, indices: List[int]) -> List[str]:
        """Find the correponding words to each index in the given sequence."""
        assert self.initialized
        return [self._process_index(i) for i in indices]

    def __call__(self, seq: Union[List[Union[str, int]], str]) -> List[Union[int, str]]:
        if seq is None or (isinstance(seq, list) and not seq):
            return seq

        if isinstance(seq, str) or isinstance(seq[0], str):
            return self.sentence_to_indices(seq)
        else:
            return self.indices_to_sentence(seq)

    def __len__(self) -> int:
        return len(self.words)

    def get_string(
        self, indices: List[int], ignore_eos: bool = True, ignore_sos: bool = True
    ) -> str:
        ignore_tokens = set()
        if ignore_eos:
            ignore_tokens.add(self.eos_idx)
        if ignore_sos:
            ignore_tokens.add(self.sos_idx)
        seq = self.indices_to_sentence(
            [idx for idx in indices if idx not in ignore_tokens]
        )
        return " ".join(seq)

    def state_dict(self) -> Dict[str, Any]:
        """Construct a state dict of the vocabulary for saving."""
        return {k: self.__dict__[k] for k in self.to_save}

    def load_state_dict(self, state: Dict[str, Any]):
        """Load a vocabulary from a state dict."""
        self.initialized = True
        self.__dict__.update(state)

    def __add__(self, other):
        """Merge this vocabulary to an other one."""
        res = WordVocabulary(
            allow_any_word=self.allow_any_word and other.allow_any_word,
            split_punctuation=self.split_punctuation,
            is_uncased=self.is_uncased or other.is_uncased,
        )
        res._add_set(set(self.words.keys()) | set(other.words.keys()))
        res.finalize()

        for word in res.words:
            i = res.words[word]
            w = res.inv_words[i]
            assert word == w, f"Merged vocabulary is inconsistent for wrod = {word}"

        assert (
            res.sos_idx == res.words["<s>"]
        ), f"Merged vocabulary is inconsistent for sos token (sos_idx = {res.sos_idx}, words['<s>'] = {res.words['<s>']})"
        assert (
            res.eos_idx == res.words["</s>"]
        ), f"Merged vocabulary is inconsistent for eos token (sos_idx = {res.eos_idx}, words['</s>'] = {res.words['</s>']})"
        assert (
            res.pad_idx == res.words["<pad>"]
        ), f"Merged vocabulary is inconsistent for pad token (sos_idx = {res.pad_idx}, words['<pad>'] = {res.words['<pad>']})"

        return res

    def mapfrom(self, other) -> Dict[int, int]:
        """Create a mapping from the indices of words in other to this vocabulary."""
        return {other.words[w]: i for w, i in self.words.items() if w in other.words}

    @property
    def sos_idx(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.special_tokens["<s>"]

    @property
    def pad_idx(self):
        """Helper to get index of pad symbol"""
        return self.special_tokens["<pad>"]

    @property
    def eos_idx(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.special_tokens["</s>"]

    @property
    def unk_idx(self):
        """Helper to get index of unk symbol"""
        return self.special_tokens.get("<unk>")

    @property
    def special_tokens(self):
        return self._special_tokens


# class CharVocabulary:
#     def __init__(self, chars: Optional[Set[str]]):
#         self.initialized = False
#         if chars is not None:
#             self.from_set(chars)

#     def from_set(self, chars: Set[str]):
#         chars = list(sorted(chars))
#         self.to_index = {c: i for i, c in enumerate(chars)}
#         self.from_index = {i: c for i, c in enumerate(chars)}
#         self.initialized = True

#     def __len__(self):
#         return len(self.to_index)

#     def state_dict(self) -> Dict[str, Any]:
#         return {
#             "chars": set(self.to_index.keys())
#         }

#     def load_state_dict(self, state: Dict[str, Any]):
#         self.from_set(state["chars"])

#     def str_to_ind(self, data: str) -> List[int]:
#         return [self.to_index[c] for c in data]

#     def ind_to_str(self, data: List[int]) -> str:
#         return "".join([self.from_index[i] for i in data])

#     def _is_string(self, i):
#         return isinstance(i, str)

#     def __call__(self, seq: Union[List[int], str]) -> Union[List[int], str]:
#         assert self.initialized
#         if seq is None or (isinstance(seq, list) and not seq):
#             return seq

#         if self._is_string(seq):
#             return self.str_to_ind(seq)
#         else:
#             return self.ind_to_str(seq)

#     def __add__(self, other):
#         return self.__class__(set(self.to_index.values()) | set(other.to_index.values()))


# class ByteVocabulary(CharVocabulary):
#     def ind_to_str(self, data: List[int]) -> bytearray:
#         return bytearray([self.from_index[i] for i in data])

#     def _is_string(self, i):
#         return isinstance(i, bytearray)
