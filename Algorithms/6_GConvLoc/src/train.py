import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import sys 
sys.path.append('..')

from src import utils
from typing import List, Tuple
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from shutil import copyfile
import time  # 添加 time 模块用于计时

log = utils.get_pylogger(__name__)

@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    # 初始化训练时间和测试时间
    train_time = 0.0
    test_time = 0.0

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, datamodule=datamodule)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        start_time = time.time()  # 记录训练开始时间
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        train_time = time.time() - start_time  # 计算训练时间
        log.info(f"Total training time: {train_time:.2f} seconds")

    train_metrics = trainer.callback_metrics

    # save checkpoint file into model directory
    if cfg.get("save_path"):
        copyfile(trainer.checkpoint_callback.best_model_path, cfg.save_path)

    if cfg.get("test"):
        log.info("Starting testing!")
        start_time = time.time()  # 记录测试开始时间
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = cfg.get("ckpt_path")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        test_time = time.time() - start_time  # 计算测试时间
        log.info(f"Total testing time: {test_time:.2f} seconds")
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    # 记录训练和测试时间到日志（可选：保存到 metric_dict）
    metric_dict["train_time_seconds"] = train_time
    metric_dict["test_time_seconds"] = test_time

    return metric_dict, object_dict

@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="train.yaml")
def main(cfg: DictConfig) -> float:
    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # 打印训练和测试时间
    log.info(f"Summary: Training time = {metric_dict.get('train_time_seconds', 0.0):.2f} seconds, "
             f"Testing time = {metric_dict.get('test_time_seconds', 0.0):.2f} seconds")

    # return optimized metric
    return metric_value

if __name__ == "__main__":
    print(type(root))
    main()