from typing import Dict, Any
import os

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
                                                                   
from lightning_utilities.core.rank_zero import rank_zero_only

from .mixins import ImageLoggerMixin
from .enhanced_model_summary import EnhancedModelSummary


__all__ = [
    "ModelCheckpoint",
    "ImageLogger",
    "EnhancedModelSummary"
]

class ImageLogger(Callback):




    def __init__(
        self,
        log_every_n_steps: int=2000,
        max_images_each_step: int=4,
        log_images_kwargs: Dict[str, Any]=None
    ) -> "ImageLogger":
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.max_images_each_step = max_images_each_step
        self.log_images_kwargs = log_images_kwargs or dict()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert isinstance(pl_module, ImageLoggerMixin)

                     
                            

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if pl_module.global_step % self.log_every_n_steps != 0:
            return

                                           
        was_training = pl_module.training
        if was_training:
            pl_module.eval()

        with torch.no_grad():
            images: Dict[str, torch.Tensor] = pl_module.log_images(
                batch, **self.log_images_kwargs
            )

                                 
        logger = (
            trainer.logger
            if not isinstance(trainer.loggers, list)
            else trainer.loggers[0]
        )
        base = getattr(logger, "log_dir", logger.save_dir)
        save_dir = os.path.join(base, "image_log", "train")
        os.makedirs(save_dir, exist_ok=True)

                       
        for key, img_tensor in images.items():
            N = min(self.max_images_each_step, img_tensor.size(0))
            grid = torchvision.utils.make_grid(img_tensor[:N], nrow=4)
            arr = (
                grid.permute(1, 2, 0)                 
                .cpu()
                .numpy()
            )
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
            fname = f"{key}_step-{pl_module.global_step:06}_e-{pl_module.current_epoch:03}_b-{batch_idx:03}.png"
            Image.fromarray(arr).save(os.path.join(save_dir, fname))

                                  
        if was_training:
            pl_module.train()
