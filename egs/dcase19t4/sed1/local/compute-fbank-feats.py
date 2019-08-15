#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from distutils.util import strtobool
import logging

import kaldiio
import numpy
import pyroomacoustics

from espnet.transform.spectrogram import logmelspectrogram
from espnet.utils.cli_utils import FileWriterWrapper
from espnet.utils.cli_utils import get_commandline_args

import numpy as np
import librosa
from pyroomacoustics.denoise.iterative_wiener import apply_iterative_wiener

def calculate_mel_spec(x,
                fs,
                n_mels,
                n_fft,
                n_shift,
                win_length,
                fmin,
                fmax):
    """
    Calculate a mal spectrogram from raw audio waveform
    Note: The parameters of the spectrograms are in the config.py file.
    Args:
        audio : numpy.array, raw waveform to compute the spectrogram

    Returns:
        numpy.array
        containing the mel spectrogram
    """


    fmin = 0 if fmin is None else fmin
    fmax = fs / 2 if fmax is None else fmax
    # Compute spectrogram
    ham_win = np.hamming(n_fft)

    spec = librosa.stft(
            x,
            n_fft=n_fft,
            hop_length=n_shift,
            window=ham_win,
            center=True,
            pad_mode='reflect'
    )

    mel_spec = librosa.feature.melspectrogram(
            S=np.abs(spec),  # amplitude, for energy: spec**2 but don't forget to change amplitude_to_db.
            sr=fs,
            n_mels=n_mels,
            fmin=fmin, fmax=fmax,
            htk=False, norm=None)

    # if self.save_log_feature:
    #     mel_spec = librosa.amplitude_to_db(mel_spec)  # 10 * log10(S**2 / ref), ref default is 1
    mel_spec = mel_spec.T
    mel_spec = mel_spec.astype(np.float32)
    return mel_spec


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fs', type=int,
                        help='Sampling frequency')
    parser.add_argument('--fmax', type=int, default=None, nargs='?',
                        help='Maximum frequency')
    parser.add_argument('--fmin', type=int, default=None, nargs='?',
                        help='Minimum frequency')
    parser.add_argument('--n_mels', type=int, default=80,
                        help='Number of mel basis')
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT length in point')
    parser.add_argument('--n_shift', type=int, default=512,
                        help='Shift length in point')
    parser.add_argument('--win_length', type=int, default=None, nargs='?',
                        help='Analisys window length in point')
    parser.add_argument('--window', type=str, default='hann',
                        choices=['hann', 'hamming'],
                        help='Type of window')
    parser.add_argument('--write-num-frames', type=str,
                        help='Specify wspecifer for utt2num_frames')
    parser.add_argument('--mono', type=strtobool, default=False,
                        help='Convert to monophonic audio')
    parser.add_argument('--noise_reduction', type=strtobool, default=False,
                        help='Apply noise reduction')
    parser.add_argument('--filetype', type=str, default='mat',
                        choices=['mat', 'hdf5'],
                        help='Specify the file format for output. '
                             '"mat" is the matrix format in kaldi')
    parser.add_argument('--compress', type=strtobool, default=False,
                        help='Save in compressed format')
    parser.add_argument('--compression-method', type=int, default=2,
                        help='Specify the method(if mat) or gzip-level(if hdf5)')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--normalize', choices=[1, 16, 24, 32], type=int,
                        default=None,
                        help='Give the bit depth of the PCM, '
                             'then normalizes data to scale in [-1,1]')
    parser.add_argument('rspecifier', type=str, help='WAV scp file')
    parser.add_argument(
        '--segments', type=str,
        help='segments-file format: each line is either'
             '<segment-id> <recording-id> <start-time> <end-time>'
             'e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5')
    parser.add_argument('wspecifier', type=str, help='Write specifier')
    args = parser.parse_args()

    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    with kaldiio.ReadHelper(args.rspecifier,
                            segments=args.segments) as reader, \
            FileWriterWrapper(args.wspecifier,
                              filetype=args.filetype,
                              write_num_frames=args.write_num_frames,
                              compress=args.compress,
                              compression_method=args.compression_method
                              ) as writer:
        for utt_id, (rate, array) in reader:
            print(utt_id)
            assert rate == args.fs
            array = array.astype(numpy.float32)

            if args.mono and array.ndim > 1:
                array = numpy.mean(array, axis=1)

            if args.normalize is not None and args.normalize != 1:
                array = array / (1 << (args.normalize - 1))

            if args.noise_reduction:
                array = apply_iterative_wiener(array,
                                               frame_len=512,
                                               lpc_order=20,
                                               iterations=2,
                                               alpha=0.8,
                                               thresh=0.01)

            # lmspc = logmelspectrogram(
            #     x=array,
            #     fs=args.fs,
            #     n_mels=args.n_mels,
            #     n_fft=args.n_fft,
            #     n_shift=args.n_shift,
            #     win_length=args.win_length,
            #     window=args.window,
            #     fmin=args.fmin,
            #     fmax=args.fmax)
            lmspc = calculate_mel_spec(
                x=array,
                fs=args.fs,
                n_mels=args.n_mels,
                n_fft=args.n_fft,
                n_shift=args.n_shift,
                win_length=args.win_length,
                fmin=args.fmin,
                fmax=args.fmax)
            writer[utt_id] = lmspc


if __name__ == "__main__":
    main()
