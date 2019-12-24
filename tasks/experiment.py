import argparse
import io
import json
import sys
from pathlib import Path

import pandas as pd
from librosa.core import load
import shutil
import numpy as np
from tqdm import tqdm
import torch
from ml.src.dataloader import set_dataloader, set_ml_dataloader
from ml.src.signal_processor import to_spect, standardize
from ml.models.model_manager import model_manager_args, BaseModelManager
from ml.src.preprocessor import Preprocessor, preprocess_args
from ml.models.pretrained_models import supported_pretrained_models
from dataset import ManualDataSet
from ml.src.metrics import metrics2df, Metric


DATALOADERS = {'normal': set_dataloader, 'ml': set_ml_dataloader}


def train_args(parser):
    model_manager_args(parser)
    expt_parser = parser.add_argument_group("Experiment arguments")
    expt_parser.add_argument('--expt-id', help='data file for training', default='')
    expt_parser.add_argument('--dataloader-type', help='Dataloader type.', choices=['normal', 'ml'], default='normal')

    return parser


def label_func(row):
    return row[1]


def load_func(path):
    const_length = 4000 * 30
    wave = load(path[0], sr=4000)[0]
    if wave.shape[0] > const_length:
        wave = wave[:const_length]
    elif wave.shape[0] < const_length:
        n_pad = (const_length - wave.shape[0]) // 2 + 1
        wave = np.pad(wave[:const_length], n_pad)[:const_length]
    return wave.reshape((1, -1))


def create_manifest():
    DATA_DIR = Path(__file__).resolve().parents[1] / 'input'

    dic = {}
    for phase in ['train', 'devel', 'test']:
        dic[phase] = [str(p.resolve()) for p in (DATA_DIR / 'wav').iterdir() if phase in p.name]
        dic[phase].sort()

    train_dev_label = pd.read_csv(DATA_DIR / 'lab' / 'labels_train_dev.tsv', sep='\t')
    test = pd.read_csv(DATA_DIR / 'lab' / 'labels_test.txt', header=None)

    train = train_dev_label.iloc[:len(dic['train']), :]
    train['file_name'] = dic['train']
    # train = train[train['label'] != 2]
    train.to_csv(DATA_DIR / 'train_manifest.csv', index=False, header=None)

    val = train_dev_label.iloc[len(dic['train']):, :]
    assert val.shape[0] == len(dic['devel'])
    val['file_name'] = dic['devel']
    # val = val[val['label'] != 2]
    val.to_csv(DATA_DIR / 'val_manifest.csv', index=False, header=None)

    test[0] = dic['test']
    test.columns = ['file_name', 'label']
    # test = test[test['label'] != 2]
    test.to_csv(DATA_DIR / 'test_manifest.csv', index=False, header=None)


def set_process_func(cfg, sr):
    window_size = cfg['window_size']
    window_stride = cfg['window_stride']

    def process_func(wave, label):
        y = to_spect(wave, sr, window_size, window_stride, window='hamming')  # channel x freq x time
        return standardize(y), label

    return process_func


def experiment(train_conf) -> float:
    phases = ['train', 'val', 'test']

    if train_conf['task_type'] == 'regress':
        train_conf['class_names'] = [0]
    else:
        train_conf['class_names'] = [0, 1, 2]

    train_conf['prev_classes'] = [0, 1]

    sr = 4000

    dataloaders = {}
    for phase in phases:
        process_func = Preprocessor(train_conf, phase, sr).preprocess
        dataset = ManualDataSet(train_conf[f'{phase}_path'], train_conf, load_func, process_func, label_func, phase)
        dataloaders[phase] = set_dataloader(dataset, phase, train_conf)

    metrics = [
        Metric('loss', direction='minimize', save_model=True),
        Metric('uar', direction='maximize'),
    ]

    model_manager = BaseModelManager(train_conf['class_names'], train_conf, dataloaders, metrics)

    model_manager.train()
    _, _, metrics = model_manager.test(return_metrics=True)
    uar = [metric for metric in metrics if metric.name == 'uar'][0]

    (Path(__file__).resolve().parent.parent / 'output' / 'params').mkdir(exist_ok=True)
    with open(Path(__file__).resolve().parent.parent / 'output' / 'params' / f"{train_conf['log_id']}.txt", 'w') as f:
        f.write('\nParameters:\n')
        f.write(json.dumps(train_conf, indent=4))

    (Path(__file__).resolve().parent.parent / 'output' / 'metrics').mkdir(exist_ok=True)
    metrics2df(metrics, phase='test').to_csv(
        Path(__file__).resolve().parent.parent / 'output' / 'metrics' / f"{train_conf['log_id']}_test.csv", index=False)

    return uar.average_meter['test'].value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    train_conf = vars(train_args(preprocess_args(parser)).parse_args())
    assert train_conf['train_path'] != '' or train_conf['val_path'] != '', \
        'You need to select training, validation data file to training, validation in --train-path, --val-path argments'

    # create_manifest()

    results = {}
    for model in supported_pretrained_models.keys():
        uar_res = []
        for seed in range(3):
            train_conf['seed'] = seed
            uar_res.append(experiment(train_conf))

        print(np.array(uar_res).mean())
        print(np.array(uar_res).std())
        results[model] = np.array(uar_res).mean()

    expt_path = Path(__file__).resolve().parent.parent / 'output' / f"{train_conf['expt_id']}.txt"
    pd.DataFrame(results).to_csv(expt_path, index=False)
