import argparse
import itertools
import logging
import pprint
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from librosa.core import load
from ml.src.dataset import ManifestWaveDataSet
from ml.tasks.base_experiment import typical_train, base_expt_args, typical_experiment
from ml.utils.utils import dump_dict


def expt_args(parser):
    parser = base_expt_args(parser)
    expt_parser = parser.add_argument_group("Elderly Experiment arguments")
    expt_parser.add_argument('--n-parallel', default=1, type=int)
    expt_parser.add_argument('--mlflow', action='store_true')
    expt_parser.add_argument('--n-classes', default=2, type=int)

    return parser


def label_func(row):
    return row[1]


def set_load_func(sr, one_audio_sec):
    def load_func(path):
        const_length = sr * one_audio_sec
        wave = load(path[0], sr=sr)[0]
        if wave.shape[0] > const_length:
            wave = wave[:const_length]
        elif wave.shape[0] < const_length:
            n_pad = (const_length - wave.shape[0]) // 2 + 1
            wave = np.pad(wave[:const_length], n_pad)[:const_length]
        return wave.reshape((1, -1))

    return load_func


def set_process_func(model_type, seq_len=50):
    def rnn_process_func(x):
        return torch.from_numpy(x.reshape(-1, seq_len))

    if model_type == 'rnn':
        return rnn_process_func
    else:
        return None


def set_data_paths(expt_dir, expt_conf):
    data_dir = Path(__file__).resolve().parents[1] / 'input'

    # db = '1'
    db = '1.5'
    if db == '1':
        dic = {}
        for phase in ['train', 'devel', 'test']:
            dic[phase] = [str(p.resolve()) for p in (data_dir / 'wav').iterdir() if phase in p.name]
            dic[phase].sort()

        train_dev_label = pd.read_csv(data_dir / 'lab' / 'labels_train_dev.tsv', sep='\t')
        test = pd.read_csv(data_dir / 'lab' / 'labels_test.txt', header=None)

        train = train_dev_label.iloc[:len(dic['train']), :]
        train['file_name'] = dic['train']
        # train = train[train['label'] != 2]
        train.to_csv(expt_dir / 'train_manifest.csv', index=False, header=None)

        val = train_dev_label.iloc[len(dic['train']):, :]
        assert val.shape[0] == len(dic['devel'])
        val['file_name'] = dic['devel']
        # val = val[val['label'] != 2]
        val.to_csv(expt_dir / 'val_manifest.csv', index=False, header=None)

        test[0] = dic['test']
        test.columns = ['file_name', 'label']
        # test = test[test['label'] != 2]
        test.to_csv(expt_dir / 'test_manifest.csv', index=False, header=None)

        # TODO train_path等の設定
        raise NotImplementedError

    elif db == '1.5':
        wav_dir = data_dir / 'db1-5' / 'wav'
        for phase, part in zip(['train', 'val', 'test'], ['train', 'devel', 'test']):
            wav_path_list = sorted(
                [str(p.resolve()) for p in wav_dir.iterdir() if part in p.name and p.name.endswith('wav')])

            label_folder = 'binary_lab' if expt_conf['n_classes'] == 2 else 'lab'
            df = pd.read_csv(data_dir / 'db1-5' / label_folder / f'labels_{part}.tsv', sep='\t')
            df['file_name'] = wav_path_list
            df.to_csv(expt_dir / f'{phase}_manifest.csv', index=False, header=None)
            expt_conf[f'{phase}_path'] = expt_dir / f'{phase}_manifest.csv'

    return expt_conf


def main(expt_conf, hyperparameters) -> float:
    if expt_conf['expt_id'] == 'timestamp':
        expt_conf['expt_id'] = dt.today().strftime('%Y-%m-%d_%H:%M')

    expt_dir = Path(__file__).resolve().parents[1] / 'output' / expt_conf['expt_id']
    Path(expt_dir).mkdir(exist_ok=True, parents=True)
    expt_conf['log_dir'] = str(expt_dir / 'tensorboard')

    logging.basicConfig(level=logging.DEBUG, format="[%(name)s] [%(levelname)s] %(message)s",
                        filename=expt_dir / 'expt.log')

    if expt_conf['n_classes'] == 2:
        expt_conf['class_names'] = [0, 1]
    else:
        expt_conf['class_names'] = [0, 1, 2]

    metrics_names = {'train': ['loss', 'uar'],
                     'val': ['loss', 'uar'],
                     'test': ['loss', 'uar']}

    dataset_cls = ManifestWaveDataSet

    patterns = list(itertools.product(*hyperparameters.values()))
    val_results = pd.DataFrame(np.zeros((len(patterns), len(hyperparameters) + len(metrics_names['val']))),
                               columns=list(hyperparameters.keys()) + metrics_names['val'])

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hyperparameters)

    one_audio_sec = 10
    expt_conf['sample_rate'] = 4000
    seq_len = 50
    load_func = set_load_func(expt_conf['sample_rate'], one_audio_sec)
    process_func = set_process_func(expt_conf['model_type'], seq_len)
    expt_conf = set_data_paths(expt_dir, expt_conf)

    groups = None

    def experiment(pattern, expt_conf):
        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = pattern[i]

        expt_conf['model_path'] = str(expt_dir / f"{'_'.join([str(p).replace('/', '-') for p in pattern])}.pth")
        expt_conf['log_id'] = f"{'_'.join([str(p).replace('/', '-') for p in pattern])}"

        with mlflow.start_run():
            result_series, val_pred, _ = typical_train(expt_conf, load_func, label_func, process_func, dataset_cls,
                                                       groups)

            mlflow.log_params({hyperparameter: value for hyperparameter, value in zip(hyperparameters.keys(), pattern)})
            mlflow.log_artifacts(expt_dir)

        return result_series, val_pred

    # For debugging
    if expt_conf['n_parallel'] == 1:
        result_pred_list = [experiment(pattern, deepcopy(expt_conf)) for pattern in patterns]
    else:
        expt_conf['n_jobs'] = 0
        result_pred_list = Parallel(n_jobs=expt_conf['n_parallel'], verbose=0)(
            [delayed(experiment)(pattern, deepcopy(expt_conf)) for pattern in patterns])

    val_results.iloc[:, :len(hyperparameters)] = [[str(param) for param in p] for p in patterns]
    result_list = np.array([result for result, pred in result_pred_list])
    val_results.iloc[:, len(hyperparameters):] = result_list
    pp.pprint(val_results)
    pp.pprint(val_results.iloc[:, len(hyperparameters):].describe())

    val_results.to_csv(expt_dir / 'val_results.csv', index=False)
    print(f"Devel results saved into {expt_dir / 'val_results.csv'}")
    for (_, _), pattern in zip(result_pred_list, patterns):
        pattern_name = f"{'_'.join([str(p).replace('/', '-') for p in pattern])}"
        dump_dict(expt_dir / f'{pattern_name}.txt', expt_conf)

    # Train with train + devel dataset
    if expt_conf['test']:
        best_trial_idx = val_results['uar'].argmax()

        best_pattern = patterns[best_trial_idx]
        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = best_pattern[i]
        dump_dict(expt_dir / 'best_parameters.txt', {p: v for p, v in zip(hyperparameters.keys(), best_pattern)})

        metrics, pred_dict_list, _ = typical_experiment(expt_conf, load_func, label_func, process_func,
                                                        dataset_cls, groups)

        sub_name = f"sub_{'_'.join([str(p).replace('/', '-') for p in best_pattern])}.csv"
        pd.DataFrame(pred_dict_list['test']).to_csv(expt_dir / f'{sub_name}_prob.csv', index=False, header=None)
        pd.DataFrame(pred_dict_list['test'].argmax(axis=1) + 1).to_csv(expt_dir / sub_name, index=False, header=None)
        print(f"Submission file is saved in {expt_dir / sub_name}")

    mlflow.end_run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    expt_conf = vars(expt_args(parser).parse_args())

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.DEBUG)
    logging.getLogger("ml").addHandler(console)

    if expt_conf['model_type'] == 'cnn':
        hyperparameters = {
            'cnn_channel_list': [[4, 8]],
            # 'cnn_kernel_sizes': [[[8], [4]], [[4], [4]]],
            # 'cnn_stride_sizes': [[[4], [2]], [[4], [4]]],
            # 'cnn_padding_sizes': [[[0], [0]]],
            'lr': [1e-4],
            'transform': ['logmel'],
        }
    elif expt_conf['model_type'] == 'cnn_rnn':
        hyperparameters = {
            'cnn_channel_list': [[3, 6]],
            # 'cnn_kernel_sizes': [[[4], [4]]],
            # 'cnn_stride_sizes': [[[4], [4]]],
            # 'cnn_padding_sizes': [[[0], [0]]],
            'lr': [1e-4],
            'transform': ['logmel'],
            'rnn_type': [expt_conf['rnn_type']],
            'bidirectional': [True, False],
            'rnn_n_layers': [1, 2],
            'rnn_hidden_size': [10, 50],
        }
    elif expt_conf['model_type'] == 'rnn':
        hyperparameters = {
            'bidirectional': [True, False],
            'rnn_type': [expt_conf['rnn_type']],
            'rnn_n_layers': [1, 2],
            'rnn_hidden_size': [10, 50],
            'transform': [None],
            'lr': [1e-4],
        }
    else:
        hyperparameters = {
            'lr': [0.01],
        }

    hyperparameters['model_type'] = [expt_conf['model_type']]

    for seed in range(1):
        expt_conf['expt_id'] = f"{expt_conf['model_type']}_{expt_conf['rnn_type']}_{expt_conf['n_classes']}-class"
        expt_conf['seed'] = seed
        main(expt_conf, hyperparameters)

    if not expt_conf['mlflow']:
        import shutil

        shutil.rmtree('mlruns')
