"""Verbatim extract of the released ++ test loader.

Source: third_party/Bailando/tpami_bailandopp/utils/functional.py, load_test_data_aist.
That module imports cv2, smplx, and PIL at import time. This file keeps the loader
callable without those packages. The body is the released function, including its
root-relative conversion and zero padding. Do not "clean up" the loop: it is part
of the numerical preprocessing the checkpoint was trained against.
"""
import json
import os

import numpy as np


def load_test_data_aist(data_dir, rotmat, move, external_wav=None, external_wav_rate=1, music_normalize=False, wav_padding=0):

    tot = 0
    input_names = []

    music_data, dance_data = [], []
    fnames = sorted(os.listdir(data_dir))
    #print(fnames)
    # fnames = fnames[:60]  # For debug
    for fname in fnames:
        path = os.path.join(data_dir, fname)
        with open(path) as f:
            #print(path)
            sample_dict = json.loads(f.read())
            np_music = np.array(sample_dict['music_array'])
            if external_wav is not None:
                # print('load from external wav!')
                wav_path = os.path.join(external_wav, fname.split('_')[-2] + '.json') if 'demo' not in data_dir else os.path.join(external_wav, fname)
                with open(wav_path) as ff:
                    sample_dict_wav = json.loads(ff.read())
                    np_music = np.array(sample_dict_wav['music_array'])

            if 'dance_array' in sample_dict:
                np_dance = np.array(sample_dict['dance_array'])
                if not rotmat:
                    root = np_dance[:, :3]  # the root
                    np_dance = np_dance - np.tile(root, (1, 24))  # Calculate relative offset with respect to root
                    np_dance[:, :3] = root

                for kk in range((len(np_dance) // move + 1) * move - len(np_dance) ):
                    np_dance = np.append(np_dance, np_dance[-1:], axis=0)

                dance_data.append(np_dance)
            # print('Music data shape: ', np_music.shape)
            else:
                np_dance = None
                dance_data = None
            music_move = external_wav_rate if external_wav is not None else move

            # print('Here!', wav_padding, flush=True)
            # zero padding left
            for kk in range(wav_padding):
                np_music = np.append(np.zeros_like(np_music[-1:]), np_music, axis=0)
            # fully devisable
            for kk in range((len(np_music) // music_move + 1) * music_move - len(np_music) ):
                np_music = np.append(np_music, np_music[-1:], axis=0)
            # zero padding right
            for kk in range(wav_padding):
                np_music = np.append(np_music, np.zeros_like(np_music[-1:]), axis=0)

            music_data.append(np_music)
            input_names.append(fname)
            # tot += 1
            # if tot == 3:
            #     break
    # if music_normalize:

    music_data_norm = music_data

    return music_data_norm, dance_data, input_names
