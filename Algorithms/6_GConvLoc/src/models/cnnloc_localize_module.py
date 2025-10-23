import os.path as osp
from typing import Any, List, Optional, Tuple
import torch
from pytorch_lightning import LightningModule
from src.models.components.prediction_model import MLPNet
from src.models.sae_module import StackedAutoencoderModule
from src.utils import save_mat, get_activation
from torchmetrics import MinMetric, MeanMetric
import numpy as np


class LocalizationModule(LightningModule):
    def __init__(
            self,
            datamodule,
            sae_encoder_ckpt_path: str,
            conv_filters_kernel_sizes: List[Tuple[int, int]],
            mlp_hiddens: Optional[List[int]] = None,
            hidden_activation: str = "relu",
            output_activation: str = "none",
            log_dir: str = None,
            criterion=torch.nn.MSELoss(),
            optimizer: Optional[torch.optim.Optimizer] = None
    ):
        super().__init__()

        datamodule.setup()

        self.y_min = datamodule.y_min.clone()
        self.y_max = datamodule.y_max.clone()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["datamodule", "criterion"])

        sae_module = StackedAutoencoderModule.load_from_checkpoint(sae_encoder_ckpt_path, datamodule=datamodule)
        self.sae_encoder = sae_module.encoder_model

        for param in self.sae_encoder.parameters():
            print(param.requires_grad)

        # convolution layers
        self.convs = torch.nn.ModuleList()

        in_channel = 1
        for filter_size, kernel_size in conv_filters_kernel_sizes:
            layers = [torch.nn.Conv1d(in_channel, filter_size, kernel_size), get_activation(hidden_activation)]
            in_channel = filter_size
            self.convs.append(torch.nn.Sequential(*layers))

        # flatten
        self.flatten = torch.nn.Flatten()

        # 1d hiddens
        self.fc_layer = MLPNet(33, 2, mlp_hiddens, output_activation=output_activation)

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

    def forward(self, x):
        x = self.sae_encoder(x)

        # 2d -> 3d
        x = x.unsqueeze(1)

        for conv in self.convs:
            x = conv(x)

        x = self.flatten(x)

        out = self.fc_layer(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        actual = y

        loss = self.criterion(pred, actual)
        interval = (self.y_max - self.y_min).to(pred.device)
        errors = torch.sqrt(torch.sum(((pred - actual) * interval) ** 2, dim=1))

        self.train_loss(loss)
        self.train_error(errors)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("train/mean_error", self.train_error, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=pred.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        actual = y

        loss = self.criterion(pred, actual)
        interval = (self.y_max - self.y_min).to(pred.device)
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
        x, y = batch
        pred = self.forward(x)
        actual = y

        loss = self.criterion(pred, actual)
        interval = (self.y_max - self.y_min).to(pred.device)
        errors = torch.sqrt(torch.sum(((pred - actual) * interval) ** 2, dim=1))

        self.test_loss(loss)
        self.test_error(errors)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("test/mean_error", self.test_error, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=pred.size(0))

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

        # save to matlab file
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
