#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

import numpy as np
import torch
import glob

import ipdb


def average_checkpoint(snapshots_dir, model, out='average.pth', num=10, last_from_best=True):
#     if args.log is not None:
#         with open(args.log) as f:
#             logs = json.load(f)
#         val_scores = []
#         for log in logs:
#             if "validation/main/acc" in log.keys():
#                 val_scores += [[log["epoch"], log["validation/main/acc"]]]
#             elif "val_perplexity" in log.keys():
#                 val_scores += [[log["epoch"], 1 / log["val_perplexity"]]]
#         if len(val_scores) == 0:
#             raise ValueError("`validation/main/acc` or `val_perplexity` is not found in log.")
#         val_scores = np.array(val_scores)
#         sort_idx = np.argsort(val_scores[:, -1])
#         sorted_val_scores = val_scores[sort_idx][::-1]
#         print("best val scores = " + str(sorted_val_scores[:args.num, 1]))
#         print("selected epochs = " + str(sorted_val_scores[:args.num, 0].astype(np.int64)))
#         last = [os.path.dirname(args.snapshots[0]) + "/snapshot.ep.%d" % (
#             int(epoch)) for epoch in sorted_val_scores[:args.num, 0]]
#     else:
#         last = sorted(args.snapshots, key=os.path.getmtime)
#         last = last[-args.num:]
#     print("average over", last)
    if model == 'crnn' or model == 'crnn_ema':
        snapshots = glob.glob(f'{snapshots_dir}/*.pth')
    elif model == 'transformer':
        snapshots = glob.glob(f'{snapshots_dir}/iteration_*.pth')
        snapshots.append(f'{snapshots_dir}/best_iteration.pth')
    elif model == 'transformer_ema':
        snapshots = glob.glob(f'{snapshots_dir}/ema_iteration_*.pth')
        snapshots.append(f'{snapshots_dir}/best_ema_iteration.pth')
    last = sorted(snapshots, key=os.path.getmtime)
    if last_from_best:
        if model == 'crnn' or model == 'crnn_ema':
            best_idx = last.index(f'{snapshots_dir}/best.pth')
        elif model == 'transformer':
            best_idx = last.index(f'{snapshots_dir}/best_iteration.pth')
        elif model == 'transformer_ema':
            best_idx = last.index(f'{snapshots_dir}/best_ema_iteration.pth')
        last = last[best_idx - num:best_idx]
    else:
        last = last[-num:]
    avg = None
 
    # sum
    for path in last:
        if model == 'crnn':
            states = torch.load(path, map_location=torch.device("cpu"))["model"]['state_dict']
        elif model == 'crnn_ema':
            states = torch.load(path, map_location=torch.device("cpu"))["ema_model"]['state_dict']
        elif model == 'transformer' or model == 'transformer_ema':
            states = torch.load(path, map_location=torch.device("cpu"))
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]

    # average
    for k in avg.keys():
        if avg[k] is not None:
            avg[k] /= args.num

    torch.save(avg, os.path.join(snapshots_dir, out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='average models from snapshot')
    parser.add_argument("--snapshots_dir", default='./exp3/baseline_16k_mel64_cbloss/model', type=str, nargs="+")
    parser.add_argument("--run-name", required=True, type=str)
    parser.add_argument("--model", required=True, type=str, choices=['crnn', 'crnn_ema', 'transformer', 'transformer_ema'])
    parser.add_argument("--out", default='average.pth', type=str)
    parser.add_argument("--num", default=10, type=int)
    parser.add_argument("--log", default=None, type=str, nargs="?")
    args = parser.parse_args()
    snapshots_dir = os.path.join('./exp3', args.run_name, 'model')
    average_checkpoint(snapshots_dir, args.model)
