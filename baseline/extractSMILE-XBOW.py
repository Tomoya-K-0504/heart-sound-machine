#!/usr/bin/python
import argparse
import os
from pathlib import Path

import pandas as pd

# Modify openSMILE paths HERE:
SMILE_PATH = 'opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract'
SMILE_CONF = 'opensmile-2.3.0/config/emobase_live4.conf'


def extract_args(parser):
    extract_parser = parser.add_argument_group("Experiment arguments")
    extract_parser.add_argument('--task-name', help='Name of the task', default='HSS1-5_binary')
    extract_parser.add_argument('--label-dir', help='Label files folder', default='../input/lab')
    extract_parser.add_argument('--wav-dir', help='Wave files folder', default='../input/wav')

    return parser


def main(extract_conf):

    # Task name
    task_name = extract_conf['task_name']

    # Paths
    audio_folder = str(Path(extract_conf['wav_dir']).resolve()) + '/'
    labels_file = str(Path(extract_conf['label_dir']).resolve())
    features_folder = '../input/processed/'
    if not os.path.isdir(features_folder):
        os.mkdir(features_folder)

    # Define partition names (according to audio files)
    partitions = ['train', 'devel', 'test']

    # Load file list
    label_paths = [Path(labels_file) / 'labels_{}.tsv'.format(part) for part in partitions]
    instances = pd.concat([pd.read_csv(path, sep='\t') for path in label_paths])['file_name']

    # Iterate through partitions and extract features
    for part in partitions:
        instances_part = instances[instances.str.startswith(part)]
        output_file      = features_folder + task_name + '.ComParE.'      + part + '.csv'
        output_file_lld  = features_folder + task_name + '.ComParE-LLD.'  + part + '.csv'
        if os.path.exists(output_file):
            os.remove(output_file)
        if os.path.exists(output_file_lld):
            os.remove(output_file_lld)
        # Extract openSMILE features for the whole partition (standard ComParE and LLD-only)
        for inst in instances_part:
            for d in Path(SMILE_CONF).parent.iterdir():
                if not d.name.endswith('.conf'):
                    continue
                os.system(SMILE_PATH + ' -C ' + str(d) + ' -I ' + audio_folder + inst + ' -instname ' + inst + ' -csvoutput '+ output_file + ' -timestampcsv 0 -lldcsvoutput ' + output_file_lld + ' -appendcsvlld 1')
            exit()

        # Compute BoAW representations from openSMILE LLDs
        # num_assignments = 10
        # for csize in [125, 250]:
        #     output_file_boaw = features_folder + task_name + '.BoAW-' + str(csize) + '.' + part + '.csv'
        #     xbow_config = '-i ' + output_file_lld + ' -attributes nt1[65]2[65] -o ' + output_file_boaw
        #     if part == 'train':
        #         xbow_config += ' -standardizeInput -size ' + str(csize) + ' -a ' + str(num_assignments) + ' -log -B codebook_' + str(csize)
        #     else:
        #         xbow_config += ' -b codebook_' + str(csize)
        #     os.system('java -Xmx12000m -jar openXBOW.jar -writeName ' + xbow_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    extract_conf = vars(extract_args(parser).parse_args())

    main(extract_conf)