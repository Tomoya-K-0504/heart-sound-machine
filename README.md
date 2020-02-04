# heart-sound-machine

## Virtual environment
```
conda create -n hss python=3.7
conda activate hss
git submodule update -i
cd ml_pkg
python setup.py develop
pip install -r requirements.txt
```

Next step: Move HSS1.5 database into "input" folder.

## matlab install
```
cd "matlabroot/extern/engines/python"
python setup.py install
```

Ref: https://jp.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

## Baseline setup
### OpenXBOW and OpenSMILE
"ComParE" is the standard feature set for the ComParE challenge, including functionals of 65 low-level descriptors (LLDs) and their deltas,
totalling up to 6373 acoustic features. The feature set is extracted with the openSMILE toolkit.
The bag-of-audio-words (BoAW) approach is based on the acoustic LLDs, generating a term-frequency histogram representation of
quantised audio words. The toolkit openXBOW is used to compute the BoAW features.

Refs are here
- [OpenXBOW](https://github.com/openXBOW/openXBOW)
- [OpenSMILE](https://www.audeering.com/opensmile/)
OpenXBOW jar file is already in baseline folder.

Now you setup OpenSMILE.
```
cd baseline
wget -O opensmile-2-3-0.zip https://www.audeering.com/download/opensmile-2-3-0-zip/?wpdmdl=4781
unzip opensmile-2-3-0.zip
```

Then, execute this after adjusting "label-dir" and "wav-dir"
```
python extractSMILE-XBOW.py --label-dir ../input/db1-5/binary_lab --wav-dir ../input/db1-5/wav --task-name HSS1-5_binary
```
This extracts both the ComParE feature set (6373 descriptors for each instance) and the 130 ComParE LLDs from the wav files of all partitions
using openSMILE. Then, openXBOW computes different BoAW representations using codebooks of different sizes.
All features are stored in the folder "features/".

