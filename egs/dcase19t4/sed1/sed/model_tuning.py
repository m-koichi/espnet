import sys

sys.path.append('./DCASE2019_task4/baseline')
import numpy as np
import json
from evaluation_measures import compute_strong_metrics
import torch
import pandas as pd
import re

from dataset import SEDDataset
from solver.transformer import Transformer
from utils.utils import ManyHotEncoder
from dcase_util.data import ProbabilityEncoder

from torch.utils.data import DataLoader
from scipy.signal import medfilt
import config as cfg
from transforms import Normalize, FrequencyMask, ApplyLog
import functools
import ipdb
from my_utils import get_batch_predictions

LABELS = ['Alarm_bell_ringing', 'Blender', 'Cat', 'Dishes', 'Dog',
          'Electric_shaver_toothbrush', 'Frying', 'Running_water', 'Speech', 'Vacuum_cleaner']


def search_best_threshold(model, valid_loader, validation_df, many_hot_encoder, step,
                          pooling_time_ratio, sample_rate, hop_length, target='Frame'):
    # Event = namedtuple('Event', ('thres', 'f1'))
    best_th = {k: 0 for k in LABELS}
    best_f1 = {k: 0 for k in LABELS}

    model.eval()
    for th in np.arange(step, 1, step):
        print('th:', th)
        predictions, _, _ , _, _, frame_measure, _ = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
                                                        post_processing=None, save_predictions=None,
                                                        pooling_time_ratio=pooling_time_ratio,
                                                        threshold=th, sample_rate=sample_rate,
                                                        hop_length=hop_length)
        valid_events_metric, valid_segments_metric = compute_strong_metrics(predictions, validation_df, pooling_time_ratio=None,
                                                     sample_rate=sample_rate, hop_length=hop_length)
        for i, label in enumerate(LABELS):
            if target == 'Event':
                f1 = valid_events_metric.class_wise_f_measure(event_label=label)['f_measure']
            elif target == 'Frame':
                f1 = frame_measure[i].calc_f1()[2]
            else:
                raise NotImplementedError
            if f1 > best_f1[label]:
                best_th[label] = th
                best_f1[label] = f1

    thres_list = [0.5] * len(LABELS)
    for i, label in enumerate(LABELS):
        thres_list[i] = best_th[label]

    predictions, _, _ , _, _, _, _ = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong, post_processing=None,
                                        threshold=thres_list, binarization_type='class_threshold',
                                        pooling_time_ratio=pooling_time_ratio,
                                        sample_rate=sample_rate,
                                        hop_length=hop_length
                                       )
    valid_events_metric, valid_segments_metric = compute_strong_metrics(predictions, validation_df, pooling_time_ratio=None,
                                                 sample_rate=sample_rate, hop_length=hop_length)
    # predictions = get_batch_predictions_tta(model, valid_loader, many_hot_encoder.decode_strong)
    # valid_events_metric = compute_strong_metrics(predictions, validation_df, pooling_time_ratio=1)

    print('best_th:', best_th)
    print('best_f1:', best_f1)
    return best_th, best_f1


#
# def search_best_post_process(predictions filt_span, accept_gap, reject_duration):
#     best_filt_span = {k: 0 for k in LABELS}
#     best_accept_gap = {k: 0 for k in LABELS}
#     best_reject_duration = {k: 0 for k in LABELS}


def get_batch_predictions_tta(model, data_loader, decoder, post_processing=False, save_predictions=None, tta=5,
                              transforms=[FrequencyMask()]):
    prediction_df = pd.DataFrame()
    for batch_idx, (batch_input, _, data_ids) in enumerate(data_loader):

        if tta != 1:
            assert transforms is not None
            mean_strong = None
            mean_weak = None
            for i in range(tta):
                batch_input_np = batch_input.numpy()
                for transform in transforms:
                    for j in range(batch_input.shape[0]):
                        batch_input_np[j] = transform(batch_input_np[j])
                batch_input_t = torch.from_numpy(batch_input_np)
                if torch.cuda.is_available():
                    batch_input_t = batch_input_t.cuda()
                strong, weak = model(batch_input_t)
                if mean_strong is None:
                    mean_strong = strong
                    mean_weak = weak
                else:
                    mean_strong += strong
                    mean_weak += weak
            pred_strong = mean_strong / tta
            pred_weak = mean_weak / tta
        else:
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            # strong, weak = model(batch_input)
            pred_strong, _ = model(batch_input)
        pred_strong = pred_strong.cpu().data.numpy()
        pred_strong = ProbabilityEncoder().binarization(pred_strong, binarization_type="global_threshold",
                                                        threshold=0.5)
        if post_processing:
            for i in range(pred_strong.shape[0]):
                pred_strong[i] = median_filt_1d(pred_strong[i])
                pred_strong[i] = fill_up_gap(pred_strong[i])
                pred_strong[i] = remove_short_duration(pred_strong[i])

        for pred, data_id in zip(pred_strong, data_ids):
            # pred = post_processing(pred)
            pred = decoder(pred)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = re.sub('^.*?-', '', data_id + '.wav')
            prediction_df = prediction_df.append(pred)

            # if batch_idx == 0:
            #     LOG.debug("predictions: \n{}".format(pred))
            #     LOG.debug("predictions strong: \n{}".format(pred_strong))
            #     prediction_df = pred.copy()
            # else:
    # pdb.set_trace()

    if save_predictions is not None:
        LOG.info("Saving predictions at: {}".format(save_predictions))
        prediction_df.to_csv(save_predictions, index=False, sep="\t")
    # ipdb.set_trace()
    return prediction_df


# def get_batch_predictions(model, data_loader, decoder, threshold=0.5, binarization_type='global_threshold',
#                           post_processing=None, save_predictions=None,
#                           pooling_time_ratio=1., sample_rate=22050, hop_length=365):
#     prediction_df = pd.DataFrame()

#     # validation = {}
#     # predictions = {}
#     # target = {}

#     for batch_idx, (batch_input, batch_target, data_ids) in enumerate(data_loader):
#         if torch.cuda.is_available():
#             batch_input = batch_input.cuda()

#         pred_strong, _ = model(batch_input)
#         pred_strong = pred_strong.cpu().data.numpy()

#         batch_target = batch_target.numpy()
#         #
#         # for i in range(pred_strong.shape[0]):
#         #     predictions[data_ids[i]] = pred_strong[i]
#         #     target[data_ids[i]] = batch_target[i]
#         # # import ipdb
#         # ipdb.set_trace()

#         # TODO: 全部推論してからパラメータ探索する

#         if binarization_type == 'class_threshold':
#             for i in range(pred_strong.shape[0]):
#                 pred_strong[i] = ProbabilityEncoder().binarization(pred_strong[i], binarization_type=binarization_type,
#                                                                    threshold=threshold, time_axis=0)
#         else:
#             pred_strong = ProbabilityEncoder().binarization(pred_strong, binarization_type=binarization_type,
#                                                             threshold=threshold)

#         # ipdb.set_trace()
#         if post_processing is not None:
#             for i in range(pred_strong.shape[0]):
#                 for post_process_fn in post_processing:
#                     # ipdb.set_trace()
#                     pred_strong[i] = post_process_fn(pred_strong[i])

#         for pred, data_id in zip(pred_strong, data_ids):
#             # pred = post_processing(pred)
#             pred = decoder(pred)
#             pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
#             pred["filename"] = re.sub('^.*?-', '', data_id + '.wav')
#             prediction_df = prediction_df.append(pred)

#             # if batch_idx == 0:
#             #     LOG.debug("predictions: \n{}".format(pred))
#             #     LOG.debug("predictions strong: \n{}".format(pred_strong))
#             #     prediction_df = pred.copy()
#             # else:
#     # pdb.set_trace()
#     # validation['prediction'] = predictions
#     # validation['target'] = target
#     # ipdb.set_trace()
#     #
#     # with open('weak_result.pkl', 'wb') as f:
#     #     pickle.dump(validation, f)
#     # ipdb.set_trace()

#     # In seconds
#     prediction_df.onset = prediction_df.onset * pooling_time_ratio / (sample_rate / hop_length)
#     prediction_df.offset = prediction_df.offset * pooling_time_ratio / (sample_rate / hop_length)

#     if save_predictions is not None:
#         LOG.info("Saving predictions at: {}".format(save_predictions))
#         prediction_df.to_csv(save_predictions, index=False, sep="\t")
#     return prediction_df


# def compute_strong_metrics(predictions, data_loader):
def median_filt_1d(event_roll, filt_span=7):
    """FUNCTION TO APPLY MEDIAN FILTER
    ARGS:
    --
    event_roll: event roll [T,C]
    filt_span: median filter span(integer odd scalar)
    RETURN:
    --
    event_roll : median filter applied event roll [T,C]
    """
    assert isinstance(filt_span, (int, list))
    if len(event_roll.shape) == 1:
        event_roll = medfilt(event_roll, filt_span)
    else:
        if isinstance(filt_span, int):
            for i in range(event_roll.shape[1]):
                event_roll[:, i] = medfilt(event_roll[:, i], filt_span)
        else:
            assert event_roll.shape[1] == len(filt_span)
            for i in range(event_roll.shape[1]):
                event_roll[:, i] = medfilt(event_roll[:, i], filt_span[i])

    return event_roll


def fill_up_gap(event_roll, accept_gap=5):
    """FUNCTION TO PERFORM FILL UP GAPS
    ARGS:
    --
    event_roll: event roll [T,C]
    accept_gap: number of accept gap to fill up (integer scalar)
    RETURN:
    --
    event_roll: processed event roll [T,C]
    """
    assert isinstance(accept_gap, (int, list))
    num_classes = event_roll.shape[1]
    event_roll_ = np.append(
            np.append(
                    np.zeros((1, num_classes)),
                    event_roll, axis=0),
            np.zeros((1, num_classes)),
            axis=0)
    aux_event_roll = np.diff(event_roll_, axis=0)

    for i in range(event_roll.shape[1]):
        onsets = np.where(aux_event_roll[:, i] == 1)[0]
        offsets = np.where(aux_event_roll[:, i] == -1)[0]
        for j in range(1, onsets.shape[0]):
            if isinstance(accept_gap, int):
                if onsets[j] - offsets[j - 1] <= accept_gap:
                    event_roll[offsets[j - 1]:onsets[j], i] = 1
            elif isinstance(accept_gap, list):
                if onsets[j] - offsets[j - 1] <= accept_gap[i]:
                    event_roll[offsets[j - 1]:onsets[j], i] = 1

    return event_roll


def remove_short_duration(event_roll, reject_duration=10):
    """Remove short duration
    ARGS:
    --
    event_roll: event roll [T,C]
    reject_duration: number of duration to reject as short section (int or list)
    RETURN:
    --
    event_roll: processed event roll [T,C]
    """
    assert isinstance(reject_duration, (int, list))
    num_classes = event_roll.shape[1]
    event_roll_ = np.append(
            np.append(
                    np.zeros((1, num_classes)),
                    event_roll, axis=0),
            np.zeros((1, num_classes)),
            axis=0)
    aux_event_roll = np.diff(event_roll_, axis=0)

    for i in range(event_roll.shape[1]):
        onsets = np.where(aux_event_roll[:, i] == 1)[0]
        offsets = np.where(aux_event_roll[:, i] == -1)[0]
        for j in range(onsets.shape[0]):
            if isinstance(reject_duration, int):
                if onsets[j] - offsets[j] <= reject_duration:
                    event_roll[offsets[j]:onsets[j], i] = 0
            elif isinstance(reject_duration, list):
                if onsets[j] - offsets[j] <= reject_duration[i]:
                    event_roll[offsets[j]:onsets[j], i] = 0

    return event_roll

def search_best_median(model, valid_loader, validation_df, many_hot_encoder, spans,
                       pooling_time_ratio, sample_rate, hop_length, best_th=None, target='Event'):
    # Event = namedtuple('Event', ('thres', 'f1'))
    best_span = {k: 1 for k in LABELS}
    best_f1 = {k: 0 for k in LABELS}

    model.eval()
    for span in spans:
        print('span:', span)
        post_process_fn = [functools.partial(median_filt_1d, filt_span=span)]
        if best_th is not None:
            predictions, _, _, _, _, frame_measure, _ = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
                                                            post_processing=post_process_fn, save_predictions=None,
                                                            threshold=best_th, binarization_type='class_threshold',
                                                            pooling_time_ratio=pooling_time_ratio,
                                                            sample_rate=sample_rate,
                                                            hop_length=hop_length)
        else:
            predictions, _, _, _, _, frame_measure, _ = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
                                                            post_processing=post_process_fn, save_predictions=None,
                                                            pooling_time_ratio=pooling_time_ratio,
                                                            sample_rate=sample_rate,
                                                            hop_length=hop_length)
        valid_events_metric, valid_segments_metric = compute_strong_metrics(predictions, validation_df, pooling_time_ratio=None,
                                                     sample_rate=sample_rate, hop_length=hop_length)
        for i, label in enumerate(LABELS):
            if target == 'Event':
                f1 = valid_events_metric.class_wise_f_measure(event_label=label)['f_measure']
            elif target == 'Frame':
                f1 = frame_measure[i].calc_f1()[2]
            else:
                raise NotImplementedError
            if f1 > best_f1[label]:
                best_span[label] = span
                best_f1[label] = f1

    # ipdb.set_trace()
    post_process_fn = [functools.partial(median_filt_1d, filt_span=list(best_span.values()))]
    if best_th is not None:
        predictions, _, _, _, _, _, _ = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
                                            post_processing=post_process_fn,
                                            threshold=best_th, binarization_type='class_threshold',
                                            pooling_time_ratio=pooling_time_ratio,
                                            sample_rate=sample_rate,
                                            hop_length=hop_length)
    else:
        predictions, _, _, _, _, _, _ = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
                                            post_processing=post_process_fn,
                                            pooling_time_ratio=pooling_time_ratio,
                                            sample_rate=sample_rate,
                                            hop_length=hop_length)
    valid_events_metric, valid_segments_metric = compute_strong_metrics(predictions, validation_df, pooling_time_ratio=None,
                                                 sample_rate=sample_rate, hop_length=hop_length)

    print('best_span:', best_span)
    print('best_f1:', best_f1)
    return best_span, best_f1


def search_best_accept_gap(model, valid_loader, validation_df, many_hot_encoder, gaps,
                           pooling_time_ratio, sample_rate, hop_length, best_th=None, target='Event'):
    # Event = namedtuple('Event', ('thres', 'f1'))
    best_gap = {k: 1 for k in LABELS}
    best_f1 = {k: 0 for k in LABELS}

    model.eval()
    for gap in gaps:
        print('gap:', gap)
        post_process_fn = [functools.partial(fill_up_gap, accept_gap=gap)]
        
        if best_th is not None:
            predictions, _, _, _, _, frame_measure, _ = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
                                                            post_processing=post_process_fn, save_predictions=None,
                                                            threshold=best_th, binarization_type='class_threshold',
                                                            pooling_time_ratio=pooling_time_ratio,
                                                            sample_rate=sample_rate,
                                                            hop_length=hop_length)
        else:
            predictions, _, _, _, _, frame_measure, _ = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
                                                            post_processing=post_process_fn, save_predictions=None,
                                                            pooling_time_ratio=pooling_time_ratio,
                                                            sample_rate=sample_rate,
                                                            hop_length=hop_length)
        valid_events_metric, valid_segments_metric = compute_strong_metrics(predictions, validation_df, pooling_time_ratio=None,
                                                     sample_rate=sample_rate, hop_length=hop_length)
        
        for i, label in enumerate(LABELS):
            if target == 'Event':
                f1 = valid_events_metric.class_wise_f_measure(event_label=label)['f_measure']
            elif target == 'Frame':
                f1 = frame_measure[i].calc_f1()[2]
            else:
                raise NotImplementedError
            if f1 > best_f1[label]:
                best_gap[label] = gap
                best_f1[label] = f1

    post_process_fn = [functools.partial(fill_up_gap, accept_gap=list(best_gap.values()))]
    if best_th is not None:
        predictions, _, _, _, _, _, _ = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
                                            post_processing=post_process_fn,
                                            threshold=best_th, binarization_type='class_threshold',
                                            pooling_time_ratio=pooling_time_ratio,
                                            sample_rate=sample_rate,
                                            hop_length=hop_length)
    else:
        predictions, _, _, _, _, _, _ = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
                                            post_processing=post_process_fn,
                                            pooling_time_ratio=pooling_time_ratio,
                                            sample_rate=sample_rate,
                                            hop_length=hop_length)
    valid_events_metric, valid_segments_metric = compute_strong_metrics(predictions, validation_df, pooling_time_ratio=None,
                                                 sample_rate=sample_rate, hop_length=hop_length)

    print('best_gap:', best_gap)
    print('best_f1:', best_f1)
    return best_gap, best_f1


def search_best_remove_short_duration(model, valid_loader, validation_df, many_hot_encoder, durations,
                                      pooling_time_ratio, sample_rate, hop_length, best_th=None, target='Event'):
    # Event = namedtuple('Event', ('thres', 'f1'))
    best_duration = {k: 1 for k in LABELS}
    best_f1 = {k: 0 for k in LABELS}

    model.eval()
    for duration in durations:
        print('duration:', duration)
        post_process_fn = [functools.partial(remove_short_duration, reject_duration=duration)]
        if best_th is not None:
            predictions, _, _, _, _, frame_measure, _ = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
                                                post_processing=post_process_fn,
                                                threshold=best_th, binarization_type='class_threshold',
                                                pooling_time_ratio=pooling_time_ratio,
                                                sample_rate=sample_rate,
                                                hop_length=hop_length)
        else:
            predictions, _, _, _, _, frame_measure, _ = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
                                                post_processing=post_process_fn,
                                                pooling_time_ratio=pooling_time_ratio,
                                                sample_rate=sample_rate,
                                                hop_length=hop_length)
        valid_events_metric, valid_segments_metric = compute_strong_metrics(predictions, validation_df, pooling_time_ratio=None,
                                                     sample_rate=sample_rate, hop_length=hop_length)
        for i, label in enumerate(LABELS):
            if target == 'Event':
                f1 = valid_events_metric.class_wise_f_measure(event_label=label)['f_measure']
            elif target == 'Frame':
                f1 = frame_measure[i].calc_f1()[2]
            else:
                raise NotImplementedError
            if f1 > best_f1[label]:
                best_duration[label] = duration
                best_f1[label] = f1

    post_process_fn = [functools.partial(remove_short_duration, reject_duration=list(best_duration.values()))]
    if best_th is not None:
        predictions, _, _, _, _, _, _ = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
                                            post_processing=post_process_fn,
                                            threshold=best_th, binarization_type='class_threshold',
                                            pooling_time_ratio=pooling_time_ratio,
                                            sample_rate=sample_rate,
                                            hop_length=hop_length)
    else:
        predictions, _, _, _, _, _, _ = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
                                            post_processing=post_process_fn,
                                            pooling_time_ratio=pooling_time_ratio,
                                            sample_rate=sample_rate,
                                            hop_length=hop_length)
    valid_events_metric, valid_segments_metric = compute_strong_metrics(predictions, validation_df, pooling_time_ratio=None,
                                                 sample_rate=sample_rate, hop_length=hop_length)

    print('best_duration:', best_duration)
    print('best_f1:', best_f1)
    return best_duration, best_f1


def show_best(model, valid_loader, validation_df, many_hot_encoder, pp_params,
              pooling_time_ratio, sample_rate, hop_length):
    best_th = list(pp_params[0].values())
    best_fs = list(pp_params[1].values())
    best_ag = list(pp_params[2].values())
    best_rd = list(pp_params[3].values())
    post_processing_fn = [functools.partial(median_filt_1d, filt_span=list(best_fs)),
                          functools.partial(fill_up_gap, accept_gap=list(best_ag)),
                          functools.partial(remove_short_duration, reject_duration=list(best_rd))]
    predictions, _, _, _, _, _, _ = get_batch_predictions(model, valid_loader, many_hot_encoder.decode_strong,
                                        post_processing=post_processing_fn,
                                        threshold=best_th, binarization_type='class_threshold',
                                        pooling_time_ratio=pooling_time_ratio,
                                        sample_rate=sample_rate,
                                        hop_length=hop_length)
    valid_events_metric, valid_segments_metric = compute_strong_metrics(predictions, validation_df, pooling_time_ratio=None,
                                                 sample_rate=sample_rate, hop_length=hop_length)


def model_ensemble(models, valid_loader, mode='majority_voting', majority_th=None, save_predictions=None):
    prediction_df = pd.DataFrame()

    if mode == 'majority_voting':
        if majority_th == None:
            majority_th = len(models) // 2
        for model in models:
            for batch_idx, (batch_input, _, data_ids) in enumerate(valid_loader):
                if torch.cuda.is_available():
                    batch_input = batch_input.cuda()

                mean_pred_strong = None
                for model in models:
                    pred_strong, _ = model(batch_input)
                    pred_strong = pred_strong.cpu().data.numpy()
                    if mean_pred_strong is None:
                        mean_pred_strong = pred_strong / len(models)
                    else:
                        mean_pred_strong += pred_strong / len(models)

            if binarization_type == 'class_threshold':
                for i in range(pred_strong.shape[0]):
                    mean_pred_strong[i] = ProbabilityEncoder().binarization(pred_strong[i],
                                                                            binarization_type=binarization_type,
                                                                            threshold=threshold, time_axis=0)
            else:
                pred_strong = ProbabilityEncoder().binarization(pred_strong, binarization_type=binarization_type,
                                                                threshold=threshold)
            if post_processing is not None:
                for i in range(pred_strong.shape[0]):
                    for post_process_fn in post_processing:
                        pred_strong[i] = post_process_fn(pred_strong[i])

            for pred, data_id in zip(pred_strong, data_ids):
                # pred = post_processing(pred)
                # ipdb.set_trace()
                pred = decoder(pred)
                pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
                pred["filename"] = re.sub('^.*?-', '', data_id + '.wav')
                prediction_df = prediction_df.append(pred)

    if mode == 'mean_posterior':
        for i in range(len(models)):
            for batch_idx, (batch_input, _, data_ids) in enumerate(valid_loader):
                if torch.cuda.is_available():
                    batch_input = batch_input.cuda()

                mean_pred_strong = None
                for model in models:
                    pred_strong, _ = model(batch_input)
                    pred_strong = pred_strong.cpu().data.numpy()
                    if mean_pred_strong is None:
                        mean_pred_strong = pred_strong / len(models)
                    else:
                        mean_pred_strong += pred_strong / len(models)

        if binarization_type == 'class_threshold':
            for i in range(pred_strong.shape[0]):
                mean_pred_strong[i] = ProbabilityEncoder().binarization(pred_strong[i],
                                                                        binarization_type=binarization_type,
                                                                        threshold=threshold, time_axis=0)
        else:
            pred_strong = ProbabilityEncoder().binarization(pred_strong, binarization_type=binarization_type,
                                                            threshold=threshold)
        if post_processing is not None:
            for i in range(pred_strong.shape[0]):
                for post_process_fn in post_processing:
                    pred_strong[i] = post_process_fn(pred_strong[i])

        for pred, data_id in zip(pred_strong, data_ids):
            # pred = post_processing(pred)
            # ipdb.set_trace()
            pred = decoder(pred)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = re.sub('^.*?-', '', data_id + '.wav')
            prediction_df = prediction_df.append(pred)

            # if batch_idx == 0:
            #     LOG.debug("predictions: \n{}".format(pred))
            #     LOG.debug("predictions strong: \n{}".format(pred_strong))
            #     prediction_df = pred.copy()
            # else:
            # pdb.set_trace()

    if save_predictions is not None:
        logging.info("Saving predictions at: {}".format(save_predictions))
        prediction_df.to_csv(save_predictions, index=False, sep="\t")
    return prediction_df


def test_komatsu(sample, validation_df, decoder):
    prediction_df = pd.DataFrame()
    # if binarization_type == 'class_threshold':

    pred = sample['new']
    ids = list(sample['new'].keys())
    for key in ids:
        pred[key] = ProbabilityEncoder().binarization(pred[key], binarization_type='global_threshold',
                                                      threshold=0.5, time_axis=0)

    for key in ids:
        # pred = post_processing(pred)
        pred[key] = decoder(pred[key])
        pred[key] = pd.DataFrame(pred[key], columns=["event_label", "onset", "offset"])
        pred[key]["filename"] = re.sub('^.*?-', '', key + '.wav')
        prediction_df = prediction_df.append(pred[key])
    # if save_predictions is not None:
    #     LOG.info("Saving predictions at: {}".format(save_predictions))

    prediction_df.to_csv('result_komatsu.csv', index=False, sep="\t")
    valid_events_metric = compute_strong_metrics(prediction_df, validation_df, pooling_time_ratio=None,
                                                 sample_rate=16000, hop_length=320)
    ipdb.set_trace()
    return prediction_df


import argparse
import os
import pickle
from distutils.util import strtobool

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True, type=str)
    parser.add_argument('--validation-meta', default='./DCASE2019_task4/dataset/metadata/validation/validation.csv')
    parser.add_argument('--input_layer_type', default=1)
    parser.add_argument('--mels', default=64)
    parser.add_argument('--n_frames', default=864)
    parser.add_argument("--transformer-init", type=str, default="pytorch",
                        choices=["pytorch", "xavier_uniform", "xavier_normal",
                                 "kaiming_uniform", "kaiming_normal"],
                        help='how to initialize transformer parameters')
    parser.add_argument("--transformer-input-layer", type=str, default="conv2d",
                        choices=["conv2d", "linear", "embed"],
                        help='transformer input layer type')
    parser.add_argument('--transformer-attn-dropout-rate', default=0.5, type=float,
                        help='dropout in transformer attention. use --dropout-rate if None is set')
    parser.add_argument('--transformer-lr', default=10.0, type=float,
                        help='Initial value of learning rate')
    parser.add_argument('--transformer-warmup-steps', default=25000, type=int,
                        help='optimizer warmup steps')
    parser.add_argument('--transformer-length-normalized-loss', default=True, type=strtobool,
                        help='normalize loss by length')
    parser.add_argument('--input-layer-type', default=2, type=int,
                        help='normalize loss by length')
    parser.add_argument('--adim', default=128, type=int)
    parser.add_argument('--aheads', default=4, type=int)
    parser.add_argument('--elayers', default=3, type=int)
    parser.add_argument('--eunits', default=512, type=int)
    parser.add_argument('--accum-grad', default=4, type=int)
    parser.add_argument('--dropout-rate', default=0.0, type=float,
                        help='Dropout rate for the encoder')
    parser.add_argument('--dropout', default=0.0, type=float,
                        help='Dropout rate for the encoder')

    args = parser.parse_args()

    # valid_meta = './DCASE2019_task4/dataset/metadata/validation/validation.csv'
    # validation_df = pd.read_csv(valid_meta, header=0, sep="\t")
    # many_hot_encoder = ManyHotEncoder(cfg.classes, n_frames=500)
    #
    # sample = pickle.load(open('sample.pickle', 'rb'))
    # test_komatsu(sample, validation_df, many_hot_encoder.decode_strong)

    # with open(os.path.join(args.model_dir, 'config.yml'), 'r') as f:
    #     config = yaml.load(f)

    # sr = '_16k' if config['n_frames'] == 500 else '_44k'
    # mels = '_mel64' if config['mels'] == 64 else '_mel128'
    # nr = '_nr' if config['test_data'] == 'noise_reduction' else ''

    valid_meta = './DCASE2019_task4/dataset/metadata/validation/validation.csv'
    valid_json = f'./data/validation_44k_mel64/data_validation.json'
    # valid_json = f'./data/train_16k_mel64/data_weak.json'
    # pooling_time_ratio = config['pooling_time_ratio']
    pooling_time_ratio = None

    print(valid_json)
    with open(valid_json, 'rb') as valid_json:
        valid_json = json.load(valid_json)['utts']

    # if config['log_mels'] == 1 and config['use_specaugment'] == 1:
    #     test_transforms = [ApplyLog(), Normalize(), FrequencyMask()]
    # elif config['log_mels'] == 1 and config['use_specaugment'] == 0:
    test_transforms = [ApplyLog(), Normalize()]

    valid_dataset = SEDDataset(valid_json, label_type='strong', sequence_length=864, transforms=test_transforms,
                               pooling_time_ratio=pooling_time_ratio)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    validation_df = pd.read_csv(valid_meta, header=0, sep="\t")
    many_hot_encoder = ManyHotEncoder(cfg.classes, n_frames=864)

    crnn_kwargs = cfg.crnn_kwargs
    # if pooling_time_ratio == 1:
    #     if config['mels'] == 64:
    #         crnn_kwargs['pooling'] = list(3 * ((1, 4),))
    #     elif config['mels'] == 128:
    #         crnn_kwargs['pooling'] = [(1, 4), (1, 4), (1, 8)]
    # elif pooling_time_ratio == 8:
    #     pass
    # else:
    #     raise ValueError
    # # crnn_kwargs['attention'] = False
    # args.input_layer_type = 1
    cnn_kwargs = {
        'pooling'   : [(1, 4), (1, 4), (1, 4)],
        'nb_filters': [64, 64, args.mels]
    }
    model = Transformer(
            input_dim=64,
            n_class=10,
            args=args,
            pooling='attention',
            input_conv=False,
            cnn_kwargs=cnn_kwargs)
    params = torch.load(os.path.join(args.model_dir, 'model', 'iteration_15600.pth'))
    model.load(parameters=params)
    model = model.cuda()
    #
    # ead_model = CRNN(**crnn_kwargs)
    # ead_params = torch.load()

    # predictions = get_batch_predictions(crnn, valid_loader, many_hot_encoder.decode_strong,
    #                                     post_processing=args.use_post_processing,
    #                                     save_predictions=os.path.join(exp_name, 'predictions', f'result_{epoch}.csv'))
    with torch.no_grad():
        best_th, best_f1 = search_best_threshold(model, valid_loader, validation_df, many_hot_encoder, step=0.1,
                                                 sample_rate=16000, hop_length=320)
        best_fs, best_f1 = search_best_median(model, valid_loader, validation_df, many_hot_encoder,
                                              spans=list(range(3, 31, 2)))
        best_ag, best_f1 = search_best_accept_gap(model, valid_loader, validation_df, many_hot_encoder,
                                                  gaps=list(range(3, 31, 2)))
        best_rd, best_f1 = search_best_remove_short_duration(model, valid_loader, validation_df, many_hot_encoder,
                                                             durations=list(range(3, 31, 2)))
        show_best(model, valid_loader, many_hot_encoder, [best_th, best_fs, best_ag, best_rd])

    best_setting = {
        'best_th': best_th,
        'best_fs': best_fs,
        'best_ag': best_ag,
        'best_rd': best_rd,
    }
    with open(os.path.join(args.model_dir, 'best_setting.pkl'), 'wb') as f:
        pickle.dump(best_setting, f)
