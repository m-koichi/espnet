import os
import pandas as pd
import numpy as np
import argparse
from distutils.util import strtobool
import sys

parser = argparse.ArgumentParser()
# general configuration
parser.add_argument('--train', default=True, type=strtobool)
parser.add_argument('--valid', default=True, type=strtobool)
parser.add_argument('--eval', default=False, type=strtobool)
parser.add_argument('--public', default=False, type=strtobool)
parser.add_argument('--metadata-dir', default='./dcase20_task4/dataset/metadata', type=str)
parser.add_argument('--audio-dir', default=os.path.join(os.getcwd(), './dcase20_task4/dataset/audio'), type=str)
parser.add_argument('--sox-option', default='--norm=-3 -r 16000 -c 1 -t wav', type=str)
parser.add_argument('--remove-dc-component', default=True, type=strtobool)

args = parser.parse_args(sys.argv[1:])

DATA_DIR = args.metadata_dir
AUDIO_DIR = args.audio_dir
SOX_OPTION = args.sox_option
REMOVE_DC = 'highpass 10 ' if args.remove_dc_component else ''


def create_wav_scp(
    scp_dir: str,
    df_meta,
    dataset: str,
    label: str,
    label_type:str
    ) -> None:
    assert label_type in ["strong_label", "weak_label", "unlabeled"]

    filenames = df_meta.dropna()['filename'].unique()
    if label in ['synthetic20']:
        dataset = os.path.join(dataset, label, 'soundscapes')
    if label in ['weak', 'unlabel_in_domain']:
        dataset = os.path.join(dataset, label)
    with open(os.path.join(scp_dir, f'wav.scp'), 'a') as wav_scp_f, \
         open(os.path.join(scp_dir, f'utt2spk'), 'a') as utt2spk_f, \
         open(os.path.join(scp_dir, f'labels.txt'), 'a') as label_txt_f:
        for filename in filenames:
            if not os.path.exists(os.path.join(AUDIO_DIR, dataset, filename)):
                continue
            wav_id = os.path.splitext(filename)[0]
            wav_scp_f.write(
                f'{label}-{wav_id} sox {os.path.join(AUDIO_DIR, dataset, filename)} {SOX_OPTION} - {REMOVE_DC}|\n')
            utt2spk_f.write(f'{label}-{wav_id} {label}\n')
            if label_type == "strong_label":
                label_txt_f.write(f"{label}-{wav_id}")
                labels = df_meta[df_meta['filename'] == filename]
                for onset, offset, event_label in zip(labels.onset, labels.offset, labels.event_label):
                    label_txt_f.write(f" ({onset},{offset},{event_label})")
                label_txt_f.write(f"\n")
            elif label_type == "weak_label":
                labels = df_meta[df_meta['filename'] == filename]
                label_txt_f.write(f"{label}-{wav_id} {labels.event_labels.values[0]}\n")
            elif label_type == "unlabeled":
                label_txt_f.write(f"{label}-{wav_id} unlabeled\n")
            else:
                raise ValueError

def main() -> None:

    data = []
    if args.train: 
        data.append('train')
    if args.valid: 
        data.append('dev')
    if args.eval: 
        data.append('eval')
    assert data != []
    for x in data:
        os.makedirs(f'data/{x}', exist_ok=True)
        with open(os.path.join('data', x, 'wav.scp'), 'w') as wav_scp_f, \
             open(os.path.join('data', x, 'utt2spk'), 'w') as utt2spk_f, \
             open(os.path.join('data', x, 'labels.txt'), 'w') as labels_txt_f:
            wav_scp_f.truncate()
            utt2spk_f.truncate()
            labels_txt_f.truncate()
        if x == 'train':
            df_synthetic = pd.read_csv(os.path.join(DATA_DIR, 'train', 'synthetic20', 'soundscapes.tsv'), delimiter='\t')
            df_unlabel = pd.read_csv(os.path.join(DATA_DIR, 'train', 'unlabel_in_domain.tsv'), delimiter='\t')
            df_weak = pd.read_csv(os.path.join(DATA_DIR, 'train', 'weak.tsv'), delimiter='\t')
            create_wav_scp(os.path.join('data', x), df_synthetic, 'train', 'synthetic20', 'strong_label')
            create_wav_scp(os.path.join('data', x), df_weak, 'train', 'weak', 'weak_label')
            create_wav_scp(os.path.join('data', x), df_unlabel, 'train', 'unlabel_in_domain', 'unlabeled')

        elif x == 'dev':
            df_validation = pd.read_csv(os.path.join(DATA_DIR, 'validation', 'validation.tsv'), delimiter='\t')
            create_wav_scp(os.path.join('data', x), df_validation, 'validation', 'validation', 'strong_label')

        elif x == 'eval':
            df_eval = pd.read_csv(os.path.join(DATA_DIR, 'eval', 'eval.tsv'), delimiter='\t')
            create_wav_scp(os.path.join('data', x), df_eval, 'eval', 'eval', 'strong_label')

        elif x == 'public':
            df_public = pd.read_csv(os.path.join(DATA_DIR, 'public', 'public.tsv'), delimiter='\t')
            create_wav_scp(os.path.join('data', x), df_public, 'public', 'public', 'strong_label')


if __name__ == '__main__':
    main()
