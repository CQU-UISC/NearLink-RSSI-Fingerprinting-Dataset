import os.path as osp
from typing import Union, List, Tuple, Optional
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src import utils
import torch

log = utils.get_pylogger(__name__)


class LocalizationDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


class UJIIndoorLocMatrixDataModule(LightningDataModule):
    def __init__(self,
                 root_dir: str,
                 powered: bool = False,
                 batch_size: Optional[int] = 128,
                 num_workers: int = 0,
                 pin_memory: bool = True):
        super().__init__()
        self.raw_dir = osp.join(root_dir, 'raw')
        self.processed_dir = osp.join(root_dir, 'processed')

        self.powered = powered

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.x_dim = None
        self.y_min = None
        self.y_max = None

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ['trainingData.csv', 'validationData.csv', 'testData.csv']

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_train is None or self.data_val is None or self.data_test is None:
            train_data_path = osp.join(self.raw_dir, 'trainingData.csv')
            valid_data_path = osp.join(self.raw_dir, 'validationData.csv')
            test_data_path = osp.join(self.raw_dir, 'testData.csv')

            train_df = pd.read_csv(train_data_path)
            valid_df = pd.read_csv(valid_data_path)
            test_df = pd.read_csv(test_data_path)

            train_df['mode'] = 0
            valid_df['mode'] = 1
            test_df['mode'] = 2

            df = pd.concat([train_df, valid_df, test_df])

            x = self.get_normalized_x(df, self.powered)
            y, y_min, y_max = self.get_normalized_y(df)

            train_mask, val_mask, test_mask = self.get_masks(df)

            train_x = torch.tensor(x[train_mask], dtype=torch.float)
            train_y = torch.tensor(y[train_mask], dtype=torch.float)

            val_x = torch.tensor(x[val_mask], dtype=torch.float)
            val_y = torch.tensor(y[val_mask], dtype=torch.float)

            test_x = torch.tensor(x[test_mask], dtype=torch.float)
            test_y = torch.tensor(y[test_mask], dtype=torch.float)

            y_min = torch.tensor(y_min, dtype=torch.float)
            y_max = torch.tensor(y_max, dtype=torch.float)

            self.data_train = LocalizationDataset(train_x, train_y)
            self.data_val = LocalizationDataset(val_x, val_y)
            self.data_test = LocalizationDataset(test_x, test_y)

            self.x_dim = train_x.shape[1]
            self.y_min = y_min
            self.y_max = y_max

    def train_dataloader(self):
        batch_size = self.batch_size if self.batch_size is not None else len(self.data_train)
        return DataLoader(
            dataset=self.data_train,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True
        )

    def val_dataloader(self):
        batch_size = self.batch_size if self.batch_size is not None else len(self.data_val)
        return DataLoader(
            dataset=self.data_val,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )

    def test_dataloader(self):
        batch_size = self.batch_size if self.batch_size is not None else len(self.data_test)
        return DataLoader(
            dataset=self.data_test,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )

    @staticmethod
    def get_normalized_x(df, powered):
        np_rssi = df.iloc[:, 0:520].replace(100, -104).to_numpy()
        rssi_min = np.min(np_rssi)
        rssi_max = np.max(np_rssi)
        # normalization
        np_rssi_normalized = (np_rssi - rssi_min) / (rssi_max - rssi_min)

        if powered:
            np_rssi_normalized = np_rssi_normalized ** np.exp(1)

        return np_rssi_normalized

    @staticmethod
    def get_normalized_y(df):
        np_y = df[['LATITUDE', 'LONGITUDE']].to_numpy()
        # MIN MAX Scaling
        y_min = np.min(np_y, axis=0)
        y_max = np.max(np_y, axis=0)

        np_y_normalized = (np_y - y_min) / (y_max - y_min)

        return np_y_normalized, y_min, y_max

    @staticmethod
    def get_masks(df):
        train_mask = (df['mode'] == 0).to_numpy()
        valid_mask = (df['mode'] == 1).to_numpy()
        test_mask = (df['mode'] == 2).to_numpy()

        return train_mask, valid_mask, test_mask
