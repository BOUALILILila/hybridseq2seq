import sys
from typing import Callable, Dict, List
import re
import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from ..data import COGSDataModule
from ..models import get_model, get_decoder, get_encoder
from ..utils import get_logger
from .base import Task

logger = get_logger(__name__)


class COGSTask(Task):
    def create_model(self):
        logger.info(f"Instantiating model using config = {self.config.model}")
        model = get_model(
            self.config.model,
            encoder_model=get_encoder(self.config.model),
            decoder_model=get_decoder(self.config.model),
        )

        def map_parameter_name(p_name):
            p_name = p_name.replace("model.", "")
            p_name = p_name.replace("decoder.decoder", "decoder.hybrid_decoder")
            if re.match(r"decoder\.hybrid_decoder\.layers\.\d\.*", p_name):
                p_name = p_name.replace("self_attention", "euclidean_self_attention")
                p_name = p_name.replace("cross_attention", "euclidean_cross_attention")
                p_name = p_name.replace("intermediate", "euclidean_intermediate")
            if re.match(r"decoder\.hybrid_decoder\.layers\.\d\.output\.*", p_name):
                p_name = p_name.replace("output", "euclidean_output")
            return p_name

        def load_weights_from_path(path):
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            for p_name in list(state_dict.keys()):
                state_dict[map_parameter_name(p_name)] = state_dict.pop(p_name)
            return state_dict

        if self.config.model.euclidean_weights_path is not None:
            logger.info(
                f"Loading Euclidean model weights from: {self.config.model.euclidean_weights_path}"
            )
            state_dict = load_weights_from_path(
                self.config.model.euclidean_weights_path
            )
            load_result = model.load_state_dict(state_dict, strict=False)
            if len(load_result.missing_keys) == 0:
                logger.error(
                    f"The hyperbolic decoder model keys should be missing when loading checkpoint from Euclidean model."
                )
            elif not all(
                (re.match(r"decoder(\.hybrid_decoder)?\.manifold\.isp_c", key))
                or (
                    re.match(
                        r"decoder\.hybrid_decoder\.layers\.\d\..*\.proj_out\.*", key
                    )
                )
                for key in set(load_result.missing_keys)
            ):
                logger.error(
                    f"There were missing keys (not hyperbolic) when loading weights from the Euclidean checkpoint: {load_result.missing_keys}"
                )
            if len(load_result.unexpected_keys) != 0:
                logger.error(
                    f"There were unexpected keys when loading weights from the Euclidean checkpoint: {load_result.unexpected_keys}."
                )

            if self.config.model.freeze_euclidean_weights:
                logger.info(f"Freezing all Euclidean model weights.")
                trainable_params = set()
                for name, param in model.named_parameters():
                    if name in state_dict:
                        param.requires_grad = False
                    else:
                        trainable_params.add(name)

                logger.info(f"Trainable parameters: {trainable_params}")
        return model

    def create_data_module(self):
        return COGSDataModule(
            train_batch_size=self.config.training.train_batch_size,
            valid_batch_size=self.config.training.valid_batch_size,
            test_batch_size=self.config.training.test_batch_size,
            max_seq_length=self.config.data.max_seq_length,
            data_dir=self.config.data.cache_dir,
            syntax_parser=self.config.data.syntax_parser,
            uncased_vocab=self.config.data.uncased_vocab,
            shared_vocab=self.config.data.shared_vocab,
            add_sos_token=self.config.data.add_sos_token,
            add_eos_token=self.config.data.add_eos_token,
        )

    def create_metric(self, type_names: List["str"], max_bad_samples: int = 0):
        return COGSAccuracy(type_names=type_names, max_bad_samples=max_bad_samples)

    def training_step(self, batch, batch_idx):
        logits = self.model(
            source_token_ids=batch["source_token_ids"],
            source_attention_mask=batch["source_attention_mask"],
            source_token_seqs=batch["source_token_seqs"],
            source_distance_matrix=batch["source_distance_matrix"],
            decode_fn=self.decode,
            target_attention_mask=batch["target_attention_mask"],
            labels=batch["labels"],
        )
        loss = self.criterion(logits, batch["labels"])

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch["labels"].shape[0],
        )
        return {"loss": loss, "logits": logits, "target": batch["labels"]}

    def validation_step(self, batch, batch_idx):
        self.config.model.generation.stop_criteria.max_len = batch["labels"].shape[1]
        outputs = self.model.generate(
            source_token_ids=batch["source_token_ids"],
            source_attention_mask=batch["source_attention_mask"],
            source_token_seqs=batch["source_token_seqs"],
            source_distance_matrix=batch["source_distance_matrix"],
            decode_fn=self.decode,
        )
        loss = self.criterion(outputs["logits"], batch["labels"])
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch["labels"].shape[0],
        )

        self.val_metric.update(
            preds=outputs["pred_seqs"],
            len_preds=outputs["pred_lens"],
            targets=batch["labels"],
            len_targets=batch["target_seq_len"],
            data=batch,
            decode=self.decode,
        )
        return {"loss": loss}

    def on_validation_epoch_end(self):
        # update and log
        metrics = self.val_metric.compute(split="val")
        self.log_dict(metrics, logger=True, prog_bar=True)
        self.val_metric.reset()

    def test_step(self, batch, batch_idx):
        self.config.model.generation.stop_criteria.max_len = batch["labels"].shape[1]

        outputs = self.model.generate(
            source_token_ids=batch["source_token_ids"],
            source_attention_mask=batch["source_attention_mask"],
            source_token_seqs=batch["source_token_seqs"],
            source_distance_matrix=batch["source_distance_matrix"],
            decode_fn=self.decode,
        )
        loss = self.criterion(outputs["logits"], batch["labels"])
        self.log(
            "test/loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=batch["labels"].shape[0],
        )

        self.test_metric.update(
            preds=outputs["pred_seqs"],
            len_preds=outputs["pred_lens"],
            targets=batch["labels"],
            len_targets=batch["target_seq_len"],
            data=batch,
            decode=self.decode,
        )
        return {"loss": loss}

    def on_test_epoch_end(self):
        # update and log
        metrics = self.test_metric.compute(split="test")
        self.log_dict(metrics, logger=True)

        self.test_metric.reset()


class COGSAccuracy:
    """Compute Accuracy on cogs data. A prediction is only considered correct if the entire predicted sequence is the same as the gold target."""

    def __init__(self, type_names: List[str], max_bad_samples: int = 0) -> None:
        self.type_names = type_names
        self.sample_types = list(range(len(self.type_names)))
        self.max_bad_samples = max_bad_samples
        self.reset()

    def reset(self):
        """Reset the metric."""
        self.correct = 0
        self.total = 0
        self.bad_sequences = []

        self.correct_per_type = {s_type: 0 for s_type in self.sample_types}
        self.total_per_type = {s_type: 0 for s_type in self.sample_types}

    def compare_out_ref(
        self,
        preds: torch.Tensor,
        len_preds: torch.Tensor,
        targets: torch.Tensor,
        len_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compare the prediction with the gold target in terms of length.

        Parameters
        ----------
        preds: torch.Tensor : (bs, T)
            The model's output predictions `(bs, T)`.
        len_preds: torch.Tensor
            The actual lengths of each predicted sequence (end of sequence token).
        targets: torch.Tensor : (bs, tgt_seq_length)
            The gold target sequences.
        len_targets: torch.Tensor
            The actual lengths of each target sequence (end of sequence token).

        Returns
        -------
        torch.Tensor
        """
        if preds.shape[1] > targets.shape[1]:
            preds = preds[:, : targets.shape[1]]
        elif preds.shape[1] < targets.shape[1]:
            targets = targets[:, : preds.shape[1]]

        unused = torch.arange(
            0, preds.shape[1], dtype=torch.long, device=targets.device
        ).unsqueeze(0) >= len_targets.unsqueeze(1)

        ok_mask = ((preds == targets) | unused).all(1) & (len_preds == len_targets)
        return ok_mask

    def update(
        self,
        preds: torch.Tensor,
        len_preds: torch.Tensor,
        targets: torch.Tensor,
        len_targets: torch.Tensor,
        data: Dict[str, torch.Tensor],
        decode: Callable,
    ) -> None:
        """
        Update the metric from the new batch of predictions.

        Parameters
        ----------
        preds: torch.Tensor : (bs, T)
            The model's output predictions `(bs, T)`.
        len_preds: torch.Tensor
            The actual lengths of each predicted sequence (end of sequence token).
        targets: torch.Tensor : (bs, tgt_seq_length)
            The gold target sequences.
        len_targets: torch.Tensor
            The actual lengths of each target sequence (end of sequence token).
        data: Dict[str, torch.Tensor]
            All features in the batch.
        decode: callable
            The decoding function token_ids -> tokens.

        """
        assert preds.shape[0] == targets.shape[0]

        ok_mask = self.compare_out_ref(preds, len_preds, targets, len_targets)

        if len(self.bad_sequences) < self.max_bad_samples:
            sample = torch.nonzero(~ok_mask).squeeze(-1)[
                : self.max_bad_samples - len(self.bad_sequences)
            ]
            for i in sample:
                self.bad_sequences.append(
                    {
                        "src_seq": " ".join(data["source_token_seqs"][i]),
                        "tgt_seq": " ".join(decode(targets[i].unsqueeze(0))[0]),
                        "type": self.type_names[int(data["type"][i].item())],
                        "pred": " ".join(decode(preds[i].unsqueeze(0))[0]),
                    }
                )

        for sample_type in torch.unique(data["type"]).int().cpu().numpy().tolist():
            mask = data["type"] == sample_type
            self.total_per_type[sample_type] += mask.float().sum().item()
            self.correct_per_type[sample_type] += ok_mask[mask].float().sum().item()

        self.total += ok_mask.nelement()
        self.correct += ok_mask.long().sum().item()

    def compute(self, split: str) -> Dict[str, float]:
        """
        Compute the final metric on all data points.

        Parameters
        ----------

        Returns
        -------
        Dict[str, float]
            A dictionary containing the accuracy per sample type.
        """
        if self.max_bad_samples > 0:
            print(">> Bad samples:")
            for num, seq in enumerate(self.bad_sequences):
                print(f" + sample [{num}]: type = {seq['type']}")
                print(f"\t source     = {seq['src_seq']}")
                print(f"\t target     = {seq['tgt_seq']}")
                print(f"\t prediction = {seq['pred']}")
                print()

        metrics = {f"{split}/accuracy": self.correct / self.total}
        for sample_type in self.sample_types:
            if self.total_per_type[sample_type] > 0:
                metrics[f"{split}/accuracy/{self.type_names[sample_type]}"] = (
                    self.correct_per_type[sample_type]
                    / self.total_per_type[sample_type]
                )
        return metrics
