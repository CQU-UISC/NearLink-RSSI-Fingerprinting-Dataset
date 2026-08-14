import os.path as osp
from typing import Union, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, LightningNodeData
from torch_geometric.data import Data
#from torch_geometric.data.lightning import LightningNodeData
from src import utils

log = utils.get_pylogger(__name__)


class SLEGraphDataModule(LightningNodeData):
    def __init__(self,
                 root_dir: str,
                 top_k: int,
                 loader: str = 'full',
                 batch_size: int = 128,
                 num_workers: int = 0,
                 num_neighbors: List[int] = None):
        self.raw_dir = osp.join(root_dir, 'raw')
        self.processed_dir = osp.join(root_dir, 'processed')

        self.top_k = top_k

        data = self.get_data()
        log.info('Data loading is done.')
        log.info(data)

        # default setting
        if loader == "neighbor" and num_neighbors is None:
            num_neighbors = [-1, -1]
        elif loader == "neighbor":
            num_neighbors = list(num_neighbors)

        if loader == 'neighbor':
            super().__init__(data=data, loader=loader, batch_size=batch_size, num_workers=num_workers,
                             num_neighbors=num_neighbors)
        else:
            super().__init__(data=data, loader=loader, batch_size=batch_size, num_workers=num_workers)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ['helipad_sle_train.csv', 'helipad_sle_val.csv', 'helipad_sle_test.csv']

    # 根据top_k的值生成处理后数据文件的名称
    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return f'cos_knn_graph_data_{self.top_k}.pt'

    # 组合 processed_dir 和 processed_file_names 来形成完整的处理后数据文件路径
    @property
    def data_file_path(self):
        return osp.join(self.processed_dir, self.processed_file_names)

    def get_data(self):
        data = self.try_to_load_data()

        if data is None:
            data = self.make_data()
            self.save_data(data)

        # isolated nodes
        isolated_node_mask = torch.full(data.train_mask.shape, True)
        unique_dst = torch.unique(data.edge_index[1])
        isolated_node_mask[unique_dst] = False
        isolated_nodes = torch.where(isolated_node_mask)[0]

        for isolated_node in isolated_nodes:
            if data.train_mask[isolated_node]:
                data.train_mask[isolated_node] = False
            elif data.val_mask[isolated_node]:
                data.val_mask[isolated_node] = False
        print(f"验证掩码大小: {data.val_mask.sum()}")  # 调试日志
        return data

    def try_to_load_data(self):
        data_path = self.data_file_path
        if osp.exists(data_path):
            data, _ = torch.load(data_path)
            log.info(f'Load data file from {data_path}')
            return data
        else:
            log.info(f'Data file {data_path} does not exists, configure data from raw files')
            return None

    def save_data(self, data):
        data_path = self.data_file_path
        torch.save((data, None), data_path)
        log.info(f'Data file saved at {data_path}.')

    def make_data(self):
        for raw_file in self.raw_file_names:
            if not osp.exists(osp.join(self.raw_dir, raw_file)):
                log.error(f'File {raw_file} should be exist in the {self.raw_dir}.')
                raise Exception(f'File {raw_file} should be exist in the {self.raw_dir}.')

        train_data_path = osp.join(self.raw_dir, 'helipad_sle_train.csv')
        valid_data_path = osp.join(self.raw_dir, 'helipad_sle_val.csv')
        test_data_path = osp.join(self.raw_dir, 'helipad_sle_test.csv')

        train_df = pd.read_csv(train_data_path)
        valid_df = pd.read_csv(valid_data_path)
        test_df = pd.read_csv(test_data_path)

        train_df['mode'] = 0
        valid_df['mode'] = 1
        test_df['mode'] = 2

        df = pd.concat([train_df, valid_df, test_df])

        x = self.get_normalized_x(df)
        x_dim = x.shape[1]
        reference_point, y= self.get_normalized_y(df)
        train_mask, valid_mask, test_mask = self.get_masks(df)

        edge_index = self.get_knn_edge_index(x, train_mask, valid_mask, test_mask, self.top_k)
        edge_dim = None

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        val_mask = torch.tensor(valid_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        data = Data(x=x, y=y, reference_point=reference_point, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                    edge_index=edge_index, x_dim=x_dim, edge_dim=edge_dim)

        log.info('Data object is created.')

        return data

    @staticmethod
    def get_normalized_x(df):
        np_rssi = df.iloc[:, 2:10].replace(100, -104).to_numpy()
        rssi_min = np.min(np_rssi)
        rssi_max = np.max(np_rssi)

        # normalization
        np_rssi_normalized = (np_rssi - rssi_min) / (rssi_max - rssi_min)

        return np_rssi_normalized

    @staticmethod
    def get_normalized_y(df):
        np_y = df[['x', 'y']].to_numpy()
        
        reference_point = np_y[0]
        print(reference_point)

        coordinates = np_y - reference_point
        # print()

        return reference_point, coordinates

    @staticmethod
    def get_masks(df):
        train_mask = (df['mode'] == 0).to_numpy()
        valid_mask = (df['mode'] == 1).to_numpy()
        test_mask = (df['mode'] == 2).to_numpy()

        return train_mask, valid_mask, test_mask

    @staticmethod
    def get_knn_edge_index(x, train_mask, val_mask, test_mask, top_k):
        # 生成一个外积矩阵
        gen_mask = lambda m1, m2: m1.reshape(-1, 1) @ m2.reshape(1, -1)
        # proper_mask为邻接矩阵
        proper_mask = gen_mask(train_mask | val_mask | test_mask, train_mask)
        # 设置对角线为0
        np.fill_diagonal(proper_mask, False)

        # 余弦相似度计算
        n_user = x.shape[0]
        denominator1 = np.sqrt(np.tile(np.sum(x ** 2, axis=1).reshape(-1, 1), (1, n_user)))
        denominator2 = denominator1.T
        x_sim_mat = (x @ x.T) / denominator1 / denominator2

        # invalidate edges
        x_sim_mat[~proper_mask] = -100
        x_sim_mat[np.isnan(x_sim_mat)] = -100
        x_dist_mat = 1 - x_sim_mat

        # 4. find top-k edges
        sorted_idx = np.argsort(x_dist_mat, axis=1)

        train_src_idx = sorted_idx[train_mask, :top_k]
        train_dst_idx = np.tile(np.where(train_mask)[0], (top_k, 1)).T

        train_edge_index = np.vstack((train_src_idx.flatten(), train_dst_idx.flatten()))

        val_test_src_idx = sorted_idx[val_mask | test_mask, :top_k]
        val_test_dst_idx = np.tile(np.where(val_mask | test_mask)[0], (top_k, 1)).T
        val_test_edge_index = np.vstack((val_test_src_idx.flatten(), val_test_dst_idx.flatten()))

        edge_index = np.hstack((train_edge_index, val_test_edge_index))

        invalid_mask = x_dist_mat[edge_index[1], edge_index[0]] > 1
        edge_index = edge_index[:, ~invalid_mask]

        edge_index = np.unique(edge_index, axis=1)

        return edge_index
