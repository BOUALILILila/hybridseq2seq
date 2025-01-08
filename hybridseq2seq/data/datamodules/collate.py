from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class VarLengthCollator:
    """
    Data collator that will dynamically pad the inputs received into the max length in the batch.
    """

    max_seq_length: int = 128
    pad_id: int = 0

    def __call__(self, features) -> Dict[str, Union[List[str], torch.Tensor]]:
        in_seq_ids = [torch.as_tensor(f["in_seq_ids"]) for f in features]

        # pad
        source_token_ids = pad_sequence(
            in_seq_ids, batch_first=True, padding_value=self.pad_id
        )
        source_token_ids = source_token_ids[:, : self.max_seq_length]
        source_len = source_token_ids.shape[1]

        source_dist_matrix = [
            torch.nn.functional.pad(
                input=torch.from_numpy(f["in_dist_matrix"]).float(),
                pad=(0, source_len - f["in_len"], 0, source_len - f["in_len"]),
                mode="constant",
                value=self.pad_id,
            )
            for f in features
        ]
        source_dist_matrix = torch.stack(source_dist_matrix)

        # mask
        source_attention_mask = (source_token_ids != self.pad_id).to(
            dtype=source_dist_matrix.dtype
        )

        target_token_ids = [torch.as_tensor(f["out_seq_ids"]) for f in features]
        # pad
        target_token_ids = pad_sequence(
            target_token_ids, batch_first=True, padding_value=self.pad_id
        )
        # mask
        target_attention_mask = (target_token_ids != self.pad_id).to(
            dtype=source_dist_matrix.dtype
        )

        target_seq_lengths = torch.as_tensor([f["out_len"] for f in features])

        return {
            "source_token_ids": source_token_ids,
            "source_attention_mask": source_attention_mask,
            "source_token_seqs": [f["in_seq_tokens"] for f in features],
            "source_distance_matrix": source_dist_matrix,
            "labels": target_token_ids,
            "target_attention_mask": target_attention_mask,
            "target_seq_len": target_seq_lengths,
            "type": torch.as_tensor([f["type"] for f in features]),
        }
