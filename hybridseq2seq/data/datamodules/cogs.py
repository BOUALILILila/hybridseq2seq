from typing import Optional

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from ..datasets import COGSDataset
from .collate import VarLengthCollator


class COGSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_batch_size: int,
        valid_batch_size: int,
        test_batch_size: Optional[int] = None,
        train_num_workers: Optional[int] = 4,
        valid_num_workers: Optional[int] = 1,
        max_seq_length: Optional[int] = 128,
        data_dir: Optional[str] = "./cache/",
        syntax_parser: Optional[str] = None,
        uncased_vocab: Optional[bool] = True,
        shared_vocab: Optional[bool] = True,
        add_sos_token: bool = True,
        add_eos_token: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir

        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = (
            self.valid_batch_size if test_batch_size is None else test_batch_size
        )

        self.train_num_workers = train_num_workers
        self.valid_num_workers = valid_num_workers

        self.max_seq_length = max_seq_length

        self.uncased_vocab = uncased_vocab
        self.shared_vocab = shared_vocab

        self.syntax_parser = syntax_parser

        self.add_sos_token = add_sos_token
        self.add_eos_token = add_eos_token

    def prepare_data(self):
        # download and cache
        _data = COGSDataset(
            shared_vocabulary=self.shared_vocab,
            cache_dir=self.data_dir,
            syntax_parser=self.syntax_parser,
            is_uncased=self.uncased_vocab,
            add_sos_token=self.add_sos_token,
            add_eos_token=self.add_eos_token,
        )
        return (_data.in_vocabulary, _data.out_vocabulary, _data._cache.type_names)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders (uses cahced dataset)

        # Warning ! lemmatized token sequences returned by the SyntacticParser are in lower case

        assert stage in ("fit", "test", "validate"), f"unk stage {stage}"
        if stage == "fit":
            self.train_set = COGSDataset(
                ["train"],
                shared_vocabulary=self.shared_vocab,
                cache_dir=self.data_dir,
                syntax_parser=self.syntax_parser,
                is_uncased=self.uncased_vocab,
                add_sos_token=self.add_sos_token,
                add_eos_token=self.add_eos_token,
            )
            self.valid_set = COGSDataset(
                ["valid"],
                shared_vocabulary=self.shared_vocab,
                cache_dir=self.data_dir,
                syntax_parser=self.syntax_parser,
                is_uncased=self.uncased_vocab,
                add_sos_token=self.add_sos_token,
                add_eos_token=self.add_eos_token,
            )

        if stage == "validate":
            self.valid_set = COGSDataset(
                ["valid"],
                shared_vocabulary=self.shared_vocab,
                cache_dir=self.data_dir,
                syntax_parser=self.syntax_parser,
                is_uncased=self.uncased_vocab,
                add_sos_token=self.add_sos_token,
                add_eos_token=self.add_eos_token,
            )

        if stage == "test":
            self.test_set = COGSDataset(
                ["gen"],
                shared_vocabulary=self.shared_vocab,
                cache_dir=self.data_dir,
                syntax_parser=self.syntax_parser,
                is_uncased=self.uncased_vocab,
                add_sos_token=self.add_sos_token,
                add_eos_token=self.add_eos_token,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            collate_fn=VarLengthCollator(
                max_seq_length=self.max_seq_length,
                pad_id=self.train_set.in_vocabulary.pad_idx,
            ),
            num_workers=self.train_num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.valid_batch_size,
            collate_fn=VarLengthCollator(
                max_seq_length=self.max_seq_length,
                pad_id=self.valid_set.in_vocabulary.pad_idx,
            ),
            num_workers=self.valid_num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.test_batch_size,
            collate_fn=VarLengthCollator(
                max_seq_length=self.max_seq_length,
                pad_id=self.test_set.in_vocabulary.pad_idx,
            ),
            num_workers=self.valid_num_workers,
            pin_memory=True,
        )
