from ml.src.dataset import ManifestDataSet


class ManualDataSet(ManifestDataSet):
    def __init__(self, manifest_path, data_conf, load_func=None, process_func=None, label_func=None, phase='train'):
        super(ManualDataSet, self).__init__(manifest_path, data_conf, load_func, process_func, label_func, phase)

    def get_image_size(self):
        return self.get_feature_size()[1:]

    def get_n_channels(self):
        return self.get_feature_size()[0]