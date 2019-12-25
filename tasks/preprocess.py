from pathlib import Path

import numpy as np
import pandas as pd
from librosa.core import load
from librosa.output import write_wav
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def split_audio(audio, sr, audio_length=30, length=10, overlap=5):
    segments = []
    one_section = sr * length
    n_loop = (audio_length - length) // overlap + 2
    for i in range(n_loop):
        start_index = i * overlap * sr
        segments.append(audio[start_index:start_index + one_section].copy())

    # For audios over 30 sec
    segments[-2] = np.hstack((segments[-2], segments[-1][overlap * sr:]))
    segments.pop(-1)

    return segments


def main():
    DATA_DIR = Path(__file__).resolve().parent / 'input'
    phases = ['train', 'devel', 'test']
    binary_label_converter = {0: 0, 1: 1, 2: 1}

    dic = {}
    for phase in phases:
        dic[phase] = [str(p.resolve()) for p in (DATA_DIR / 'wav').iterdir() if phase in p.name]
        dic[phase].sort()

    train_dev_label = pd.read_csv(DATA_DIR / 'lab' / 'labels_train_dev.tsv', sep='\t')
    test = pd.read_csv(DATA_DIR / 'lab' / 'labels_test.txt', header=None)

    train = train_dev_label.iloc[:len(dic['train']), :]
    train['file_name'] = dic['train']

    dev = train_dev_label.iloc[len(dic['train']):, :]
    assert dev.shape[0] == len(dic['devel'])
    dev['file_name'] = dic['devel']

    test[0] = dic['test']
    test.columns = ['file_name', 'label']

    wav_out_dir = DATA_DIR / 'db1-5' / 'wav'
    wav_out_dir.mkdir(exist_ok=True, parents=True)

    for phase, df in tqdm(zip(phases, [train, dev, test]), total=3):
        label_dic = {'file_name': [], 'label': []}
        binary_label_dic = {'file_name': [], 'label': []}

        for _, row in df.iterrows():
            wave, sr = load(row['file_name'], sr=4000)
            wave_sections = split_audio(wave, sr=sr)
            for i, section in enumerate(wave_sections):
                file_name = Path(row['file_name']).name.replace('.wav', f'_{i + 1}.wav')
                write_wav(wav_out_dir / file_name, section, sr)
                label_dic['file_name'].append(file_name)
                label_dic['label'].append(row['label'])
                binary_label_dic['file_name'].append(file_name)
                binary_label_dic['label'].append(binary_label_converter[row['label']])

        (wav_out_dir.parent / 'lab').mkdir(exist_ok=True)
        (wav_out_dir.parent / 'binary_lab').mkdir(exist_ok=True)
        label_file_path = wav_out_dir.parent / 'lab' / f'labels_{phase}.tsv'
        pd.DataFrame(label_dic).to_csv(label_file_path, index=False, sep='\t')
        label_file_path = wav_out_dir.parent / 'binary_lab' / f'labels_{phase}.tsv'
        pd.DataFrame(binary_label_dic).to_csv(label_file_path, index=False, sep='\t')


if __name__ == '__main__':
    main()
