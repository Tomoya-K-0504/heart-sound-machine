import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


def extract_args(parser):
    extract_parser = parser.add_argument_group("Experiment arguments")
    extract_parser.add_argument('--output-wav-dir', help='Wave files folder', default='../input/wav')
    extract_parser.add_argument('--input-wav-dir', help='Wave files folder', default='../input/wav')

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    extract_conf = vars(extract_args(parser).parse_args())

    wav_dir = Path(extract_conf['input_wav_dir'])
    Path(extract_conf['output_wav_dir']).mkdir(exist_ok=True)
    for path in tqdm(wav_dir.iterdir(), total=len(wav_dir.iterdir())):
        wav = librosa.load(path, sr=4000)[0].astype(np.float)
        sf.write(Path(extract_conf['output_wav_dir']) / path.name, wav, 4000, 'PCM_16')
