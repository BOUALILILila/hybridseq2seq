# original code: https://github.com/RobertCsordas/transformer_generalization/blob/master/dataset/text/cogs.py
import os

from .typed_dataset import TypedTextDataset, TypedTextDatasetCache

from ...utils import get_logger, download

logger = get_logger(__name__)


class COGSDataset(TypedTextDataset):
    URL_BASE = "https://raw.githubusercontent.com/najoungkim/COGS/main/data/"
    SYN_BASE_URL = (
        "https://raw.githubusercontent.com/coli-saar/Syntax-COGS/main/data/syntax/"
    )
    SPLT_TYPES = ["train", "test", "valid", "gen"]
    NAME_MAP = {"valid": "dev"}

    def build_cache(self) -> TypedTextDatasetCache:
        types = []
        type_list = []
        type_map = {}

        index_table = {}
        in_sentences = []
        out_sentences = []

        if self.syntax_parser_type == "gold":
            syn_trees_in_sentences = []
            need_syn = True
        else:
            syn_trees_in_sentences = None
            need_syn = False

        for st in self.SPLT_TYPES:
            fname = self.NAME_MAP.get(st, st) + ".tsv"
            split_fn = os.path.join(self.cache_dir, fname)
            if not os.path.exists(split_fn):
                os.makedirs(os.path.dirname(split_fn), exist_ok=True)

                full_url = self.URL_BASE + fname
                logger.info(f"Downloading COGS {st} split from {full_url}")
                download(full_url, split_fn, ignore_if_exists=False)
            else:
                logger.info(f"Using cached COGS {st} split from {split_fn}")

            # syntax gold trees if parser is gold
            if need_syn:
                split_syn_fn = os.path.join(self.cache_dir, "syntax", fname)
                if not os.path.exists(split_syn_fn):
                    os.makedirs(os.path.dirname(split_syn_fn), exist_ok=True)

                    full_url = self.SYN_BASE_URL + fname
                    logger.info(f"Downloading SYNTAX-COGS {st} split from {full_url}")
                    download(full_url, split_syn_fn, ignore_if_exists=False)
                else:
                    logger.info(
                        f"Using cached SYNTAX-COGS {st} split from {split_syn_fn}"
                    )

            index_table[st] = []

            with open(split_fn, "r") as f:
                for line in f:
                    in_seq, out_seq, type = line.rstrip().split("\t")

                    index_table[st].append(len(in_sentences))
                    in_sentences.append(in_seq)
                    out_sentences.append(out_seq)

                    type_id = type_map.get(type)
                    if type_id is None:
                        type_map[type] = type_id = len(type_list)
                        type_list.append(type)

                    types.append(type_id)

                assert len(in_sentences) == len(out_sentences)

            if need_syn:
                with open(split_syn_fn, "r") as f:
                    for line in f:
                        _, syn_tree, _ = line.rstrip().split("\t")
                        syn_trees_in_sentences.append(syn_tree)

                    assert len(syn_trees_in_sentences) == len(in_sentences)

        # cache and dataset need to include optional distance matrices
        return TypedTextDatasetCache().build(
            index_table={"default": index_table},
            in_sentences=in_sentences,
            out_sentences=out_sentences,
            types=types,
            type_names=type_list,
            syntax_parser=self.syntax_parser,
            syntax_parser_type=self.syntax_parser_type,
            syn_trees_in_sentences=syn_trees_in_sentences,
            split_punctuation=self.split_punctuation,
            allow_any_word=self.allow_any_word,
            is_uncased=self.is_uncased,
        )
