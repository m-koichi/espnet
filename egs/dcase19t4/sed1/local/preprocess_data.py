import os
import pandas as pd
import numpy as np
import shutil
from glob import glob
import argparse
import sys
# work_dir = espnet/egs/dcase19t4/sed1
DATA_DIR = './DCASE2019_task4/dataset/metadata'
AUDIO_DIR = os.path.join(os.getcwd(), 'DCASE2019_task4/dataset/audio')
SOX_OPTION = '-r 16000 -c 1 -t wav'


parser = argparse.ArgumentParser()
# general configuration
parser.add_argument('--train-dir', default='train_nr', type=str)
parser.add_argument('--valid-dir', default='validation_nr', type=str)
parser.add_argument('--eval-dir', default='eval_nr', type=str)

args = parser.parse_args(sys.argv[1:])


def make_scps(scp_dir: str, filenames: np.ndarray, dataset: str, att: str) -> None:
    with open(os.path.join(scp_dir, 'text'), 'a') as text_f, \
         open(os.path.join(scp_dir, 'wav.scp'), 'a') as wav_scp_f, \
         open(os.path.join(scp_dir, 'utt2spk'), 'a') as utt2spk_f:

        if dataset == 'train':
            for filename in filenames:
                # import ipdb
                # ipdb.set_trace()
                if not os.path.exists(os.path.join(AUDIO_DIR, dataset, att, filename)):
                    continue
                wav_id = os.path.splitext(filename)[0]
                wav_scp_f.write(f'{att}-{wav_id} sox {os.path.join(AUDIO_DIR, dataset, att, filename)} {SOX_OPTION} - |\n')
                utt2spk_f.write(f'{att}-{wav_id} {att}\n')
                text_f.write(f'{att}-{wav_id} {wav_id}.json\n')

        elif dataset == 'validation':
            for filename in filenames:
                if not os.path.exists(os.path.join(AUDIO_DIR, dataset, filename)):
                    continue
                wav_id = os.path.splitext(filename)[0]
                wav_scp_f.write(f'{att}-{wav_id} sox {os.path.join(AUDIO_DIR, dataset, filename)} {SOX_OPTION} - |\n')
                utt2spk_f.write(f'{att}-{wav_id} {att}\n')
                text_f.write(f'{att}-{wav_id} {wav_id}.json\n')

        elif dataset == 'eval':
            for filename in filenames:
                if not os.path.exists(os.path.join(AUDIO_DIR, dataset, filename)):
                    continue
                wav_id = os.path.splitext(filename)[0]
                wav_scp_f.write(f'{att}-{wav_id} sox {os.path.join(AUDIO_DIR, dataset, filename)} {SOX_OPTION} - |\n')
                utt2spk_f.write(f'{att}-{wav_id} {att}\n')
                text_f.write(f'{att}-{wav_id} {wav_id}.json\n')


def remove_missing_file_label():
    """
    Remake matadata. Remove missing file label
    """
    train_set = ['synthetic', 'unlabel_in_domain', 'weak']
    validation_set = ['validation', 'test_dcase2018', 'eval_dcase2018']
    for x in ['train', 'validation']:
        if x == 'train':
            for train in train_set:
                metadata = os.path.join(DATA_DIR, x, train + '.csv')
                shutil.move(metadata, metadata.replace('csv', 'bak'))
                with open(metadata, 'w') as metadata_new, \
                     open(metadata.replace('csv', 'bak')) as metadata_org:
                    metadata_new.write(metadata_org.readline())
                    for line in metadata_org.readlines():
                        wav_file = line.strip().split('\t')[0]
                        if not os.path.exists(os.path.join(AUDIO_DIR, x, train, wav_file)):
                            continue
                        metadata_new.write(line)

        if x == 'validation':
            for validation in validation_set:
                metadata = os.path.join(DATA_DIR, x, validation + '.csv')
                shutil.move(metadata, metadata.replace('csv', 'bak'))
                with open(metadata, 'w') as metadata_new, \
                     open(metadata.replace('csv', 'bak')) as metadata_org:
                    metadata_new.write(metadata_org.readline())
                    for line in metadata_org.readlines():
                        index = line.strip().split('\t')
                        if len(line.strip().split('\t')) == 1:
                            continue
                        wav_file = index[0]
                        if not os.path.exists(os.path.join(AUDIO_DIR, x, wav_file)):
                            continue
                        metadata_new.write(line)


def make_eval_list():
    missing_files = ['6e0sgfmd.wav', '803o63xo.wav', 'czwbz8ym.wav', 'd01s5h2o.wav', 'ejb1szay.wav'
                     'm4fidt9q.wav', 'phwmle95.wav', 'sai7h25y.wav', 'th7_mnm2.wav']
    os.makedirs(os.path.join(DATA_DIR, 'eval'), exist_ok=True)
    wav_list = glob(os.path.join(AUDIO_DIR, 'eval/*.wav'))
    with open(os.path.join(DATA_DIR, 'eval/eval_dcase2019.csv'), 'w') as f:
        f.write('filename\n')
        for filename in wav_list:
            filename = os.path.basename(filename)
            if filename in missing_files:
                continue
            f.write(filename + '\n')


def main():

    os.makedirs(f'data/{args.train_dir}', exist_ok=True)
    os.makedirs(f'data/{args.valid_dir}', exist_ok=True)
    os.makedirs(f'data/{args.eval_dir}', exist_ok=True)

    # remove_missing_file_label()
    # make_eval_list()
    for x in [args.train_dir, args.valid_dir, args.eval_dir]:
        with open(os.path.join('data', x, 'text'), 'w') as text_f, \
             open(os.path.join('data', x, 'wav.scp'), 'w') as wav_scp_f, \
             open(os.path.join('data', x, 'utt2spk'), 'w') as utt2spk_f:

            text_f.truncate()
            wav_scp_f.truncate()
            utt2spk_f.truncate()

        if x == args.train_dir:

            df_synthetic = pd.read_csv(os.path.join(DATA_DIR, 'train', 'synthetic.csv'), delimiter='\t')
            df_unlabel = pd.read_csv(os.path.join(DATA_DIR, 'train', 'unlabel_in_domain.csv'), delimiter='\t')
            df_weak = pd.read_csv(os.path.join(DATA_DIR, 'train', 'weak.csv'), delimiter='\t')

            make_scps(os.path.join('data', x), df_synthetic.dropna()['filename'].unique(), 'train', 'synthetic')
            make_scps(os.path.join('data', x), df_unlabel.dropna()['filename'].unique(), 'train', 'unlabel_in_domain')
            make_scps(os.path.join('data', x), df_weak.dropna()['filename'].unique(), 'train', 'weak')

        elif x == args.valid_dir:
            df_validation = pd.read_csv(os.path.join(DATA_DIR, 'validation', 'validation.csv'), delimiter='\t')
            df_eval = pd.read_csv(os.path.join(DATA_DIR, 'validation', 'eval_dcase2018.csv'), delimiter='\t')
            df_test = pd.read_csv(os.path.join(DATA_DIR, 'validation', 'test_dcase2018.csv'), delimiter='\t')

            make_scps(os.path.join('data', x), df_validation.dropna()['filename'].unique(), 'validation', 'validation')
            make_scps(os.path.join('data', x), df_eval.dropna()['filename'].unique(), 'validation', 'eval_dcase2018')
            make_scps(os.path.join('data', x), df_test.dropna()['filename'].unique(), 'validation', 'test_dcase2018')

        elif x == args.eval_dir:
            df_eval = pd.read_csv(os.path.join(DATA_DIR, 'eval', 'eval.csv'), delimiter='\t')
            make_scps(os.path.join('data', x), df_eval.dropna()['filename'].unique(), 'eval', 'eval_dcase2019')


if __name__ == '__main__':
    main()
