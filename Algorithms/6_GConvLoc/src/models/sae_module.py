from typing import List, Optional
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric
from src.models.components.prediction_model import MLPNet


class StackedAutoencoderModule(LightningModule):
    def __init__(
            self,
            datamodule,
            mlp_hiddens: Optional[List[int]] = None,
            hidden_activation: str = "relu",
            output_activation: str = "none",
            log_dir: str = None,
            criterion=torch.nn.MSELoss(),
            optimizer: Optional[torch.optim.Optimizer] = None
    ):
        super().__init__()

        datamodule.setup()
        x_dim = datamodule.x_dim

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["datamodule", "criterion"])

        self.encoder_model = MLPNet(x_dim, mlp_hiddens[-1], hidden_layer_sizes=mlp_hiddens[:-1],
                                    hidden_activation=hidden_activation, output_activation=hidden_activation)
        self.decoder_model = MLPNet(mlp_hiddens[-1], x_dim, hidden_layer_sizes=list(reversed(mlp_hiddens[:-1])),
                                    hidden_activation=hidden_activation, output_activation=output_activation)

        self.log_dir = log_dir

        # loss function
        self.criterion = criterion

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x):
        return self.decoder_model.forward(self.encoder_model.forward(x))

    def training_step(self, batch, batch_idx):
        x, _ = batch
        pred = self.forward(x)
        actual = x

        loss = self.criterion(pred, actual)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        pred = self.forward(x)
        actual = x

        loss = self.criterion(pred, actual)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))

    def test_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        scheduler, optimizer = self.hparams.optimizer(params=self.parameters())

        if scheduler is None:
            return optimizer
        else:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
