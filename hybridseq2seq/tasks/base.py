from typing import Optional, List, Tuple
import os, shutil

import torch
from torch import nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from geoopt.optim import RiemannianAdam, RiemannianSGD

from .config import load_config_from_yaml
from .loss import get_criterion

from ..utils import get_logger

logger = get_logger(__name__)


class Task(pl.LightningModule):
    def __init__(self, config_file: str, no_cuda: Optional[bool] = False) -> None:
        super().__init__()
        self.config = load_config_from_yaml(config_file)
        os.makedirs(self.config.training.output_folder, exist_ok=True)
        shutil.copy(config_file, self.config.training.output_folder)

        logger.info(f"Creating the data module...")
        self.data_module = self.create_data_module()
        logger.info(f"Preparing data...")
        in_vocab, self.out_vocab, type_names = self.data_module.prepare_data()
        # Vocab related config for model
        self.config.model.vocab_size = len(in_vocab)
        self.config.model.add_sos_token = self.config.data.add_sos_token
        self.config.model.pad_idx = in_vocab.pad_idx
        self.config.model.sos_idx = self.out_vocab.sos_idx
        self.config.model.eos_idx = self.out_vocab.eos_idx

        logger.info(f"Creating the Model...")
        self.model = self.create_model()

        self.criterion = self.create_criterion()
        self.val_metric = self.create_metric(type_names=type_names)
        self.test_metric = self.create_metric(type_names=type_names, max_bad_samples=5)

        # self.train_metric = self.create_metric(type_names=type_names, max_bad_samples=20)

        self.trainer = self.create_trainer(no_cuda)

        pl.seed_everything(self.config.seed, workers=True)

    def create_model(self) -> pl.LightningModule:
        raise NotImplementedError()

    def create_data_module(self) -> pl.LightningDataModule:
        raise NotImplementedError()

    def create_trainer(self, no_cuda: Optional[bool] = False) -> pl.Trainer:
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.training.output_folder,
            filename="{epoch}-{step}",
            # save_on_train_epoch_end=True,
            # every_n_train_steps=100,
            every_n_epochs=self.config.training.save_every_n_epochs,
            save_top_k=self.config.training.save_top_k,
            mode="max",
            monitor="step",
            save_last=True,
        )
        devices = (
            1 if self.config.training.devices is None else self.config.training.devices
        )
        logger.info(
            f"Using devices: {devices}"
        )  # "auto" / int indicating number of devices / list indicating the devices to use
        if self.config.training.log_wandb:
            self.wandb_logger = WandbLogger(project=self.config.task)
            log_type = [self.wandb_logger]
        else:
            log_type = True
        logger.info(f"Using logger: {log_type}")
        return pl.Trainer(
            logger=log_type,
            accelerator="auto",
            devices=devices,
            max_epochs=self.config.training.max_epochs,
            precision=self.config.training.precision,
            check_val_every_n_epoch=self.config.training.check_val_every_n_epoch,
            log_every_n_steps=self.config.training.log_every_n_steps,
            default_root_dir=self.config.training.output_folder,
            callbacks=[checkpoint_callback],
            deterministic=True,  # For reproducibility
        )

    def create_criterion(self):
        return get_criterion(
            self.config.training.loss, ignore_idx=self.config.model.pad_idx
        )

    def configure_optimizers(
        self,
    ) -> Tuple[
        List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]
    ]:
        decay_parameters = get_parameter_names(self, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        named_parameters = list(
            filter(lambda p: p[1].requires_grad, self.named_parameters())
        )
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in named_parameters if n in decay_parameters],
                "weight_decay": self.config.training.optimizer.weight_decay,
            },
            {
                "params": [p for n, p in named_parameters if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        if self.config.training.optimizer.name == "RiemannianAdam":
            optimizer = RiemannianAdam(
                optimizer_grouped_parameters,
                lr=self.config.training.optimizer.learning_rate,
            )
        elif self.config.training.optimizer.name == "RiemannianSGD":
            optimizer = RiemannianSGD(
                optimizer_grouped_parameters,
                lr=self.config.training.optimizer.learning_rate,
            )
        else:
            raise ValueError(f"Invalid optimizer {self.config.training.optimizer} !")

        if self.config.training.lr_scheduler == None:
            return optimizer

        # Setup the scheduler
        scheduler = None
        if self.config.training.lr_scheduler.name == "StepLR":
            step_size = self.config.training.lr_scheduler.step_size
            gamma = self.config.training.lr_scheduler.gamma
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
        else:
            raise ValueError(f"Invalid scheduler {self.config.training.lr_scheduler} !")
        return [optimizer], [scheduler]

    def decode(self, batched_token_ids: torch.Tensor) -> List[List[str]]:
        return [self.out_vocab(seq.tolist()) for seq in batched_token_ids]

    def run_train(self) -> None:
        # self.wandb_logger.watch(self, log="all")
        last_model_path = (
            self.config.training.last_ckpt
            if self.config.training.last_ckpt is not None
            else None
        )
        if last_model_path is not None:
            logger.info(f"Resuming from checkpoint at : {last_model_path}")
        self.trainer.fit(
            self,
            datamodule=self.data_module,
            ckpt_path=last_model_path,
        )
        # self.wandb_logger.experiment.unwatch(self)

    def run_validation(self) -> None:
        self.trainer.validate(self, datamodule=self.data_module)

    def run_test(self, ckpt_path: Optional[str] = None) -> None:
        if ckpt_path is None:
            self.trainer.test(self, datamodule=self.data_module)
        else:
            self.trainer.test(self, datamodule=self.data_module, ckpt_path=ckpt_path)


def get_parameter_names(
    model: nn.Module, forbidden_layer_types: List[nn.Module]
) -> List[str]:
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result
