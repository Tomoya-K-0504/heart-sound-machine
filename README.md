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
```
cd input/
cp -r db1-5/ db15_binary/
rm -r db15_binary/lab
mv db15_binary/binary_lab db15_binary/lab

cp -r db1-5/ db15_3-class/
rm -r db15_3-class/binary_lab 
```

## matlab install
```
cd "matlabroot/extern/engines/python"
python setup.py install
```

Ref: https://jp.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

## Baseline features setup
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
python wav4smile.py --input-wav-dir ../input/db15_binary/wav --output-wav-dir ../input/db15_binary/wav_smile
python extractSMILE-XBOW.py --label-dir ../input/db15_binary/lab --wav-dir ../input/db15_binary/wav_smile --task-name HSS1-5_binary
```
This extracts both the ComParE feature set (6373 descriptors for each instance) and the 130 ComParE LLDs from the wav files of all partitions
using openSMILE. Then, openXBOW computes different BoAW representations using codebooks of different sizes.
All features are stored in the folder "features/".


### auDeep
1. Construct [auDeep](https://github.com/auDeep/auDeep) virtual environment. 
```
cd baseline
git clone https://github.com/auDeep/auDeep.git
conda create -n audeep python=3.5
conda activate audeep
cp ../src/hss15* auDeep/audeep/backend/parsers/
pip install ./auDeep
bash audeep_generate_binary.sh ../input/db_binary/
```

### deepSpectrum
This is the installation example using conda. Please see other install option [here](https://github.com/DeepSpectrum/DeepSpectrum)
You need to execute those commands after auDeep setup ended.
```
git clone https://github.com/auDeep/auDeep.git
git submodule update --init --recursive
conda config --add channels pytorch
conda config --add channels conda-forge
conda create -n DeepSpectrum -c deepspectrum deepspectrum
conda activate DeepSpectrum
pip install ./auDeep
bash deepspectrum_generate_binary.sh ../input/db_binary/
```

## Baseline evaluation (with Linear SVM)
