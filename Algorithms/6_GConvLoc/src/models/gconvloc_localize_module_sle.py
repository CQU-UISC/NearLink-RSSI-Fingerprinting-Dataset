import os.path as osp
from typing import Any, List, Optional
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from src.models.components.gat_model import GAT
from src.utils import save_mat
import matplotlib.pyplot as plt
import pandas as pd

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

        self.save_hyperparameters(logger=False, ignore=["datamodule", "criterion"])

        self.gat = GAT(x_dim, hidden_channels, 2, edge_dim, activation=activation, heads=heads,
                       conv_dropout=conv_dropout, node_dropout=node_dropout, edge_dropout=edge_dropout,
                       add_self_loops=add_self_loops, fill_value=fill_value, share_weights=share_weights,
                       output_activation=output_activation, normalize=normalize, post_mlp_hiddens=post_mlp_hiddens,
                       post_activation=post_activation, post_dropout=post_dropout, use_post_mlp=use_post_mlp,
                       post_input_dropout=post_input_dropout)

        self.log_dir = log_dir
        self.criterion = criterion

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        self.train_error = MeanMetric()
        self.val_error = MeanMetric()
        self.test_error = MeanMetric()

        self.train_mse = MeanMetric()
        self.val_mse = MeanMetric()
        self.test_mse = MeanMetric()
        
        self.train_rmse = MeanMetric()
        self.val_rmse = MeanMetric()
        self.test_rmse = MeanMetric()
        
        self.train_variance = MeanMetric()
        self.val_variance = MeanMetric()
        self.test_variance = MeanMetric()

        self.train_std = MeanMetric()
        self.val_std = MeanMetric()
        self.test_std = MeanMetric()

        self.val_error_best = MinMetric()

    def forward(self, batch, stage="train"):
        if hasattr(batch, 'batch_size'):
            batch_size = batch.batch_size
            mask = None
        else:
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
        
        # Loss remains unchanged (MSE over all samples and dimensions)
        loss = self.criterion(pred, actual)
        
        # Compute Euclidean errors for each sample
        errors = torch.sqrt(torch.sum((pred - actual) ** 2, dim=1))
        
        # Compute mean_error as mean of Euclidean errors
        mean_error = torch.mean(errors)
        
        # Compute mse as mean of squared Euclidean errors
        mse = torch.mean(errors ** 2)
        
        # Compute rmse as sqrt(mse)
        rmse = torch.sqrt(mse)
        
        # Compute variance of Euclidean errors
        variance = torch.var(errors, unbiased=False)
        
        # Compute standard deviation
        std = torch.sqrt(variance)

        # Update metrics
        self.train_loss(loss)
        self.train_error(mean_error)
        self.train_mse(mse)
        self.train_rmse(rmse)
        self.train_variance(variance)
        self.train_std(std)

        # Log metrics
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("train/mean_error", self.train_error, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("train/mean_mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("train/mean_rmse", self.train_rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("train/mean_variance", self.train_variance, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("train/mean_std", self.train_std, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        pred, actual = self.forward(batch, stage="valid")

        # Loss remains unchanged
        loss = self.criterion(pred, actual)
        
        # Compute Euclidean errors for each sample
        errors = torch.sqrt(torch.sum((pred - actual) ** 2, dim=1))
        
        # Compute mean_error as mean of Euclidean errors
        mean_error = torch.mean(errors)
        
        # Compute mse as mean of squared Euclidean errors
        mse = torch.mean(errors ** 2)
        
        # Compute rmse as sqrt(mse)
        rmse = torch.sqrt(mse)
        
        # Compute variance of Euclidean errors
        variance = torch.var(errors, unbiased=False)
        
        # Compute standard deviation
        std = torch.sqrt(variance)

        # Update metrics
        self.val_loss(loss)
        self.val_error(mean_error)
        self.val_mse(mse)
        self.val_rmse(rmse)
        self.val_variance(variance)
        self.val_std(std)

        # Log metrics
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("val/mean_error", self.val_error, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("val/mean_mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("val/mean_rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("val/mean_variance", self.val_variance, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("val/mean_std", self.val_std, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))

    def validation_epoch_end(self, outputs):
        mean_error = self.val_error.compute()
        self.val_error_best(mean_error)

        self.log("val/mean_error_best", self.val_error_best.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        pred, actual = self.forward(batch, stage="test")

        # Loss remains unchanged
        loss = self.criterion(pred, actual)
        
        # Compute Euclidean errors for each sample
        errors = torch.sqrt(torch.sum((pred - actual) ** 2, dim=1))
        
        # Compute mean_error as mean of Euclidean errors
        mean_error = torch.mean(errors)
        
        # Compute mse as mean of squared Euclidean errors
        mse = torch.mean(errors ** 2)
        
        # Compute rmse as sqrt(mse)
        rmse = torch.sqrt(mse)
        
        # Compute variance of Euclidean errors
        variance = torch.var(errors, unbiased=False)
        
        # Compute standard deviation
        std = torch.sqrt(variance)

        # Update metrics
        self.test_loss(loss)
        self.test_error(mean_error)
        self.test_mse(mse)
        self.test_rmse(rmse)
        self.test_variance(variance)
        self.test_std(std)

        # Log metrics
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("test/mean_error", self.test_error, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("test/mean_mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("test/mean_rmse", self.test_rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("test/mean_variance", self.test_variance, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))
        self.log("test/mean_std", self.test_std, on_step=False, on_epoch=True, prog_bar=True, batch_size=pred.size(0))

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

        # Save results to .mat file
        mat_path = osp.join(self.log_dir, "result.mat")
        result_dict = {
            "pred": pred,
            "actual": actual,
            "errors": errors
        }
        save_mat(mat_path, result_dict)

                
        # Save individual errors to CSV
        errors_data = pd.DataFrame({
            'error': errors
        })
        errors_csv_path = osp.join(self.log_dir, "errors.csv")
        errors_data.to_csv(errors_csv_path, index=False)


        # Compute ECDF
        sorted_errors = np.sort(errors)
        ecdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

        # Save ECDF data to CSV
        ecdf_data = pd.DataFrame({
            'error': sorted_errors,
            'ecdf': ecdf
        })
        csv_path = osp.join(self.log_dir, "ecdf_data.csv")
        ecdf_data.to_csv(csv_path, index=False)

        # Plot ECDF
        plt.figure(figsize=(8, 6))
        plt.step(sorted_errors, ecdf, label='ECDF of Errors', where='post')
        plt.xlabel('Error')
        plt.ylabel('Cumulative Probability')
        plt.title('Empirical Cumulative Distribution Function of Errors')
        plt.grid(True)
        plt.legend()
        plot_path = osp.join(self.log_dir, "ecdf_plot.png")
        plt.savefig(plot_path)
        plt.close()

        # Save test metrics to CSV with values rounded to four decimal places
        test_metrics = {
            'test/loss': round(self.test_loss.compute().item(), 4),
            'test/mean_error': round(self.test_error.compute().item(), 4),
            'test/mean_mse': round(self.test_mse.compute().item(), 4),
            'test/mean_rmse': round(self.test_rmse.compute().item(), 4),
            'test/mean_variance': round(self.test_variance.compute().item(), 4),
            'test/mean_std': round(self.test_std.compute().item(), 4)
        }
        test_metrics_data = pd.DataFrame([test_metrics])
        metrics_csv_path = osp.join(self.log_dir, "test_metrics.csv")
        test_metrics_data.to_csv(metrics_csv_path, index=False)

    def configure_optimizers(self):
        scheduler, optimizer = self.hparams.optimizer(params=self.parameters())

        if scheduler is None:
            return optimizer
        else:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}