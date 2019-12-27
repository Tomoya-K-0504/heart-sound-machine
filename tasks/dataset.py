import numpy as np
import torch
from ml.src.dataset import ManifestDataSet


class ManualDataSet(ManifestDataSet):
    def __init__(self, manifest_path, data_conf, load_func=None, process_func=None, label_func=None, phase='train'):
        super(ManualDataSet, self).__init__(manifest_path, data_conf, load_func, process_func, label_func, phase)
        self.cache = data_conf['cache']
        self.cached_idx = set()

    def __getitem__(self, idx):
        label = self.labels[idx]

        if self.cache and idx in self.cached_idx:
            x = torch.from_numpy(np.load(self.path_df.iloc[idx, 0].replace('.wav', '.npy')))

        else:
            x = self.load_func(self.path_df.iloc[idx, :])

            if self.process_func:
                x, label = self.process_func(x, label)

            if self.cache:
                np.save(self.path_df.iloc[idx, 0].replace('.wav', '.npy'), x.numpy())
                self.cached_idx.add(idx)

        return x, label

    def get_image_size(self):
        return self.get_feature_size()[1:]

    def get_n_channels(self):
        return self.get_feature_size()[0]