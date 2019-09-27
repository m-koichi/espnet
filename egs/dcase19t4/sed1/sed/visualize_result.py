# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipdb


CLASSES = {
    'Alarm_bell_ringing'        : 0,
    'Blender'                   : 1,
    'Cat'                       : 2,
    'Dishes'                    : 3,
    'Dog'                       : 4,
    'Electric_shaver_toothbrush': 5,
    'Frying'                    : 6,
    'Running_water'             : 7,
    'Speech'                    : 8,
    'Vacuum_cleaner'            : 9
}


def get_frame_level_predictoin(df, fname, n_class=10, n_frames=62):
    res = np.zeros((n_class, n_frames))
    for onset, offset, event_label in zip(df[df.filename == fname]['onset'],
                                df[df.filename == fname]['offset'],
                                df[df.filename == fname]['event_label']):
        onset_frame = int(onset / (10 / n_frames))
        offset_frame = int(offset / (10 / n_frames))
        label = CLASSES[event_label]
        res[label, onset_frame:offset_frame] =1
    return res


def plot_result(predict, ground_truth, dest, ext='png'):
    result = predict + 2 * ground_truth
    plt.imshow(result)
    plt.savefig(f'{dest}.{ext}')
    


def main():
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument('--predictions', type=str, default='./result_epoch94.csv',
                        help='predictions csv file')
    parser.add_argument('--ground-truth', type=str, default='../DCASE2019_task4/dataset/metadata/validation/validation.csv',
                        help='ground trush csv file')
    parser.add_argument('--outdir', type=str, default='../exp/debug/img',
                        help='Output directory')
    parser.add_argument('--n_frames', type=int, default=62,
                        help='Output directory')
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    df_predictions = pd.read_csv(args.predictions, delimiter='\t')
    df_ground_truth = pd.read_csv(args.ground_truth, delimiter='\t')
    
    n_frames = args.n_frames
    wav_list = df_ground_truth['filename'].unique()
    
    for i, file_name in enumerate(wav_list):
        if i == 10:
            break
        predict = get_frame_level_predictoin(df_predictions, file_name, n_class=10, n_frames=n_frames)
        ground_truth = get_frame_level_predictoin(df_ground_truth, file_name, n_class=10, n_frames=n_frames)
        plot_result(predict, ground_truth, dest=os.path.join(args.outdir,file_name.replace('.wav', '')))
    



if __name__ == '__main__':
    main()
