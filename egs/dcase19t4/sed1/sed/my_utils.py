
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
from dcase_util.data import ProbabilityEncoder
import time
import torch
import torch.nn as nn
import re
from sklearn.metrics import confusion_matrix
import os
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

weak_samples = {
    'Alarm_bell_ringing'        : 192,
    'Blender'                   : 125,
    'Cat'                       : 164,
    'Dishes'                    : 177,
    'Dog'                       : 208,
    'Electric_shaver_toothbrush': 97,
    'Frying'                    : 165,
    'Running_water'             : 322,
    'Speech'                    : 522,
    'Vacuum_cleaner'            : 162
}

weak_samples_list = [192, 125, 164, 177, 208, 97, 165, 322, 522, 162]

def cycle_iteration(iterable):
    while True:
        for i in iterable:
            yield i
            
            
def get_sample_rate_and_hop_length(args):
    if args.n_frames == 864:
        sample_rate = 44100
        hop_length = 511
    elif args.n_frames == 605:
        sample_rate = 22050
        hop_length = 365
    elif args.n_frames == 496:
        sample_rate = 16000
        hop_length = 323
    else:
        raise ValueError
    
    return sample_rate, hop_length

def log_scalar(writer, name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    writer.add_scalar(name, value, step)
    mlflow.log_metric(name, value)
    
    

def mat_to_img(mat, dest, ext='png'):
    plt.figure(figsize=(14,1)) # arbitrary, looks good on my screen.
#     plt.gca().invert_yaxis()
    plt.imshow(mat, origin='lower')
    plt.savefig(f'{dest}.{ext}')
    
    
class ConfMat():
    def __init__(self):
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0
        
    def add_cf(self, tn, fp, fn ,tp):
        self.tn += tn
        self.fp += fp
        self.fn += fn
        self.tp += tp
        
    def calc_f1(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        f_score = 2 * precision * recall / (precision + recall + 1e-7)
        return precision, recall, f_score
    
    def reset(self):
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0
        

        
def get_batch_predictions(model, data_loader, decoder, post_processing=None,
                          save_predictions=None,
                          transforms=None, mode='validation', logger=None,
                          pooling_time_ratio=1., sample_rate=22050, hop_length=365,
                          threshold=0.5, binarization_type='global_threshold'):
    '''
    post_processing: e.g. [functools.partial(median_filt_1d, filt_span=39)]
    '''
    prediction_df = pd.DataFrame()
    avg_strong_loss = 0
    avg_weak_loss = 0
    
    # Flame level 
    frame_measure = [ConfMat() for i in range(len(CLASSES))]
    tag_measure = ConfMat()
    
    start = time.time()
    for batch_idx, (batch_input, target, data_ids) in enumerate(data_loader):

        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
        pred_strong, pred_weak = model(batch_input)
        
        target_np = target.numpy()

        if mode == 'validation':
            class_criterion = nn.BCELoss().cuda()
            target = target.cuda()
            strong_class_loss = class_criterion(pred_strong, target)
            weak_class_loss = class_criterion(pred_weak, target.max(-2)[0])
            avg_strong_loss += strong_class_loss.item() / len(data_loader)
            avg_weak_loss += weak_class_loss.item() / len(data_loader)

        pred_strong = pred_strong.cpu().data.numpy()
        pred_weak = pred_weak.cpu().data.numpy()
        
        
        if binarization_type == 'class_threshold':
            for i in range(pred_strong.shape[0]):
                pred_strong[i] = ProbabilityEncoder().binarization(pred_strong[i], binarization_type=binarization_type,
                                                                   threshold=threshold, time_axis=0)
        elif binarization_type == 'global_threshold':
            pred_strong = ProbabilityEncoder().binarization(pred_strong, binarization_type=binarization_type,
                                                            threshold=threshold)
        else:
            raise ValueError
        pred_weak = ProbabilityEncoder().binarization(pred_weak, binarization_type='global_threshold',
                                                        threshold=0.5)

        for pred, data_id in zip(pred_strong, data_ids):
            # pred = post_processing(pred)
            # ipdb.set_trace()
            if post_processing is not None:
                for i in range(pred_strong.shape[0]):
                    for post_process_fn in post_processing:
                        # ipdb.set_trace()
                        pred_strong[i] = post_process_fn(pred_strong[i])
            pred = decoder(pred)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = re.sub('^.*?-', '', data_id + '.wav')
            prediction_df = prediction_df.append(pred)
            
        for i in range(len(pred_strong)):
            tn, fp, fn, tp = confusion_matrix(target_np[i].max(axis=0), pred_weak[i], labels=[0, 1]).ravel()
            tag_measure.add_cf(tn, fp, fn, tp)
            for j in range(len(CLASSES)):
                tn, fp, fn, tp = confusion_matrix(target_np[i][:, j], pred_strong[i][:, j], labels=[0, 1]).ravel()
                frame_measure[j].add_cf(tn, fp, fn, tp)
        

    # In seconds
    prediction_df.onset = prediction_df.onset * pooling_time_ratio / (sample_rate / hop_length)
    prediction_df.offset = prediction_df.offset * pooling_time_ratio / (sample_rate / hop_length)
    
    # Compute frame level macro f1 score
    macro_f1 = 0
    ave_precision = 0
    ave_recall = 0
    for i in range(len(CLASSES)):
        ave_precision_, ave_recall_, macro_f1_ = frame_measure[i].calc_f1()
        ave_precision += ave_precision_
        ave_recall += ave_recall_
        macro_f1 += macro_f1_
    ave_precision /= len(CLASSES)
    ave_recall /= len(CLASSES)
    macro_f1 /= len(CLASSES)
    

    if save_predictions is not None:
        LOG.info("Saving predictions at: {}".format(save_predictions))
        prediction_df.to_csv(save_predictions, index=False, sep="\t")

    weak_f1 = tag_measure.calc_f1()[2]    
    if mode == 'validation' and logger is not None:
        logger.scalar_summary('valid_strong_loss', avg_strong_loss, global_step)
        logger.scalar_summary('valid_weak_loss', avg_weak_loss, global_step)
        logger.scalar_summary('frame_level_macro_f1', macro_f1, global_step)
        logger.scalar_summary('frame_level_ave_precision', ave_precision, global_step)
        logger.scalar_summary('frame_level_ave_recall', ave_recall, global_step)
        logger.scalar_summary('frame_level_weak_f1', weak_f1, global_step)
        
    elapsed_time = time.time() - start
    print(f'prediction finished. elapsed time: {elapsed_time}')
    print(f'valid_strong_loss: {avg_strong_loss}')
    print(f'valid_weak_loss: {avg_weak_loss}')
    print(f'frame level macro f1: {macro_f1}')
    print(f'frame level ave. precision: {ave_precision}')
    print(f'frame level ave. recall: {ave_recall}')
    print(f'weak f1: {weak_f1}')
    
    return prediction_df, ave_precision, ave_recall, macro_f1, weak_f1, frame_measure, tag_measure
        
    
def get_batch_predictions_trans(model, data_loader, decoder, post_processing=None,
                          save_predictions=None,
                          transforms=None, mode='validation', logger=None,
                          pooling_time_ratio=1., sample_rate=22050, hop_length=365,
                          threshold=0.5, binarization_type='global_threshold'):
    '''
    post_processing: e.g. [functools.partial(median_filt_1d, filt_span=39)]
    '''
    prediction_df = pd.DataFrame()
    avg_strong_loss = 0
    avg_weak_loss = 0
    
    # Flame level 
    frame_measure = [ConfMat() for i in range(len(CLASSES))]
    tag_measure = ConfMat()
    
    start = time.time()
    for batch_idx, (batch_input, target, data_ids, _) in enumerate(data_loader):

        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
        # pred_strong, pred_weak, _ = model(batch_input)
        pred_strong, pred_weak, attn_ws = model(batch_input)
#         attn_ws = attn_ws.cpu().detach().numpy()
#         for data_id, attn_w in zip(data_ids, attn_ws):
#             os.mkdir(f'img/attnw/{data_id}')
#             for i, head in enumerate(attn_w):
#                 plot_attention_weights(attn_ws=head, dest=f'img/attnw/{data_id}/head_{i}', ext='png')
#         ipdb.set_trace()
        target_np = target.numpy()

        if mode == 'validation':
            class_criterion = nn.BCELoss().cuda()
            target = target.cuda()
            strong_class_loss = class_criterion(pred_strong, target)
            weak_class_loss = class_criterion(pred_weak, target.max(-2)[0])
            avg_strong_loss += strong_class_loss.item() / len(data_loader)
            avg_weak_loss += weak_class_loss.item() / len(data_loader)

        pred_strong = pred_strong.cpu().data.numpy()
        pred_weak = pred_weak.cpu().data.numpy()
        
        
        if binarization_type == 'class_threshold':
            for i in range(pred_strong.shape[0]):
                pred_strong[i] = ProbabilityEncoder().binarization(pred_strong[i], binarization_type=binarization_type,
                                                                   threshold=threshold, time_axis=0)
        elif binarization_type == 'global_threshold':
            pred_strong = ProbabilityEncoder().binarization(pred_strong, binarization_type=binarization_type,
                                                            threshold=threshold)
        else:
            raise ValueError
        pred_weak = ProbabilityEncoder().binarization(pred_weak, binarization_type='global_threshold',
                                                        threshold=0.5)

        for pred, data_id in zip(pred_strong, data_ids):
            # pred = post_processing(pred)
            # ipdb.set_trace()
            if post_processing is not None:
                for i in range(pred_strong.shape[0]):
                    for post_process_fn in post_processing:
                        # ipdb.set_trace()
                        pred_strong[i] = post_process_fn(pred_strong[i])
            pred = decoder(pred)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = re.sub('^.*?-', '', data_id + '.wav')
            prediction_df = prediction_df.append(pred)
            
        for i in range(len(pred_strong)):
            tn, fp, fn, tp = confusion_matrix(target_np[i].max(axis=0), pred_weak[i], labels=[0, 1]).ravel()
            tag_measure.add_cf(tn, fp, fn, tp)
            for j in range(len(CLASSES)):
                tn, fp, fn, tp = confusion_matrix(target_np[i][:, j], pred_strong[i][:, j], labels=[0, 1]).ravel()
                frame_measure[j].add_cf(tn, fp, fn, tp)
        

    # In seconds
    prediction_df.onset = prediction_df.onset * pooling_time_ratio / (sample_rate / hop_length)
    prediction_df.offset = prediction_df.offset * pooling_time_ratio / (sample_rate / hop_length)
    
    # Compute frame level macro f1 score
    macro_f1 = 0
    ave_precision = 0
    ave_recall = 0
    for i in range(len(CLASSES)):
        ave_precision_, ave_recall_, macro_f1_ = frame_measure[i].calc_f1()
        ave_precision += ave_precision_
        ave_recall += ave_recall_
        macro_f1 += macro_f1_
    ave_precision /= len(CLASSES)
    ave_recall /= len(CLASSES)
    macro_f1 /= len(CLASSES)
    

    if save_predictions is not None:
        LOG.info("Saving predictions at: {}".format(save_predictions))
        prediction_df.to_csv(save_predictions, index=False, sep="\t")

    weak_f1 = tag_measure.calc_f1()[2]    
    if mode == 'validation' and logger is not None:
        logger.scalar_summary('valid_strong_loss', avg_strong_loss, global_step)
        logger.scalar_summary('valid_weak_loss', avg_weak_loss, global_step)
        logger.scalar_summary('frame_level_macro_f1', macro_f1, global_step)
        logger.scalar_summary('frame_level_ave_precision', ave_precision, global_step)
        logger.scalar_summary('frame_level_ave_recall', ave_recall, global_step)
        logger.scalar_summary('frame_level_weak_f1', weak_f1, global_step)
        
    elapsed_time = time.time() - start
    print(f'prediction finished. elapsed time: {elapsed_time}')
    print(f'valid_strong_loss: {avg_strong_loss}')
    print(f'valid_weak_loss: {avg_weak_loss}')
    print(f'frame level macro f1: {macro_f1}')
    print(f'frame level ave. precision: {ave_precision}')
    print(f'frame level ave. recall: {ave_recall}')
    print(f'weak f1: {weak_f1}')
    
    return prediction_df, ave_precision, ave_recall, macro_f1, weak_f1, frame_measure, tag_measure
        
            

def average_checkpoint(models):
    
    return ave_model


def score_fusion(models):
    pass

def plot_attention_weights(attn_ws, dest, ext='png'):
    plt.imshow(attn_ws, origin='lower')
    plt.xlabel('Query')
    plt.ylabel('Key')
    plt.savefig(f'{dest}.{ext}')