import os.path as osp
from typing import Any, List, Optional
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from src.models.components.gat_model import GAT
from src.utils import save_mat


class LocalizationModule(LightningModule):
    def __init__(
            self,
            datamodule,
            hidden_channels: Optional[List[int]] = None,
            activation: str = "relu",
            heads: int = 1,
            conv_dropout: float = 0.0,
            node_dropout: float = 0.0,
            edge_dropout: float = 0.0,
            add_self_loops: bool = False,
            fill_value: str = "max",
            share_weights: bool = False,
            output_activation: str = "none",
            normalize: bool = True,
            post_mlp_hiddens: Optional[List[int]] = None,
            post_activation: str = "relu",
            post_input_dropout: float = 0.0,
            post_dropout: float = 0.0,
            use_post_mlp: bool = False,
            log_dir: str = None,
            criterion=torch.nn.MSELoss(),
            optimizer: Optional[torch.optim.Optimizer] = None
    ):
        super().__init__()

        data = datamodule.data
        x_dim = data.x_dim
        edge_dim = data.edge_dim if hasattr(data, 'edge_dim') else None

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["datamodule", "criterion"])

        self.gat = GAT(x_dim, hidden_channels, 2, edge_dim, activation=activation, heads=heads,
                       conv_dropout=conv_dropout, node_dropout=node_dropout, edge_dropout=edge_dropout,
                       add_self_loops=add_self_loops, fill_value=fill_value, share_weights=share_weights,
                       output_activation=output_activation, normalize=normalize, post_mlp_hiddens=post_mlp_hiddens,
                       post_activation=post_activation, post_dropout=post_dropout, use_post_mlp=use_post_mlp,
                       post_input_dropout=post_input_dropout)

        self.log_dir = log_dir

        # loss function
        self.criterion = criterion

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_error = MeanMetric()
        self.val_error = MeanMetric()
        self.test_error = MeanMetric()

        self.val_error_best = MinMetric()

    def forward(self, batch, stage="train"):
        if hasattr(batch, 'batch_size'):
            # mini-batch
            batch_size = batch.batch_size
            mask = None
        else:
            # full-batch
            if stage == "train":
                mask = batch.train_mask
            elif stage == "valid":
                mask = batch.val_mask
            else:
                mask = batch.test_mask
            batch_size = None

        out = self.gat(batch.x, batch.edge_index, batch.edge_attr, stage=stage)

        if mask is not None:
            pred = out[mask]
            actual = batch.y[mask]
        else:
            pred = out[:batch_size]
            actual = batch.y[:batch_size]

        return pred, actual

    def on_train_start(self):
        self.val_error_best.reset()

    def training_step(self, batch, batch_idx):
        pred, actual = self.forward(batch, stage="train")

        loss = self.criterion(pred, actual)
        interval = batch.y_max - batch.y_min
        errors = torch.sqrt(torch.sum(((pred - actual) * interval) ** 2, dim=1))

        self.train_loss(loss)
        self.train_error(errors)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("train/mean_error", self.train_error, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        pred, actual = self.forward(batch, stage="valid")

        loss = self.criterion(pred, actual)
        interval = batch.y_max - batch.y_min
        errors = torch.sqrt(torch.sum(((pred - actual) * interval) ** 2, dim=1))

        self.val_loss(loss)
        self.val_error(errors)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("val/mean_error", self.val_error, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))

    def validation_epoch_end(self, outputs):
        mean_error = self.val_error.compute()
        self.val_error_best(mean_error)

        self.log("val/mean_error_best", self.val_error_best.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        pred, actual = self.forward(batch, stage="test")

        loss = self.criterion(pred, actual)
        interval = batch.y_max - batch.y_min
        errors = torch.sqrt(torch.sum(((pred - actual) * interval) ** 2, dim=1))

        self.test_loss(loss)
        self.test_error(errors)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("test/mean_error", self.test_error, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))

        ret = {"loss": loss, "pred": pred, "actual": actual, "errors": errors}

        return ret

    def test_epoch_end(self, outputs: List[Any]):
        pred_list = []
        actual_list = []
        errors_list = []
        for ret in outputs:
            pred_ = ret["pred"].cpu().detach().numpy()
            actual_ = ret["actual"].cpu().detach().numpy()
            errors_ = ret["errors"].cpu().detach().numpy()

            pred_list.append(pred_)
            actual_list.append(actual_)
            errors_list.append(errors_)

        pred = np.vstack(pred_list)
        actual = np.vstack(actual_list)
        errors = np.concatenate(errors_list)

        if self.log_dir is None:
            return

        # save matlab
        mat_path = osp.join(self.log_dir, "result.mat")
        result_dict = {
            "pred": pred,
            "actual": actual,
            "errors": errors
        }
        save_mat(mat_path, result_dict)

    def configure_optimizers(self):
        scheduler, optimizer = self.hparams.optimizer(params=self.parameters())

        if scheduler is None:
            return optimizer
        else:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
