from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

if __name__ == '__main__':
    wav_dir = Path('/home/tomoya/workspace/research/heart-sound-machine/input/db15_small/wav')
    (wav_dir.parent / 'wav4smile').mkdir(exist_ok=True)
    for path in wav_dir.iterdir():
        wav = librosa.load(path, sr=4000)[0].astype(np.float)
        sf.write(wav_dir.parent / 'wav4smile' / path.name, wav, 4000, 'PCM_16')