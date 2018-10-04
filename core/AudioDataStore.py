#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 27.09.18
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : AudioDataStore
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import librosa
import numpy as np
import threading

from core.DataStore import DataStore


class AudioDataStore(DataStore):
    def __init__(self, *args, **kwargs):
        # TODO, extend to more than one extension
        self.extension = "wav"
        super(AudioDataStore, self).__init__(self, extension=self.extension, *args, **kwargs)
        self.target_fs = kwargs.get('target_fs', 22000)
        self.labels = kwargs.get('labels', None)


    def read(self, *args, **kwargs):
        index = kwargs.get('index', -1)
        if index == -1:
            audio, fs = librosa.load(self.files[self.read_pointer], dtype='float32', sr=self.target_fs, mono=True)

            file_name = os.path.basename(self.files[self.read_pointer])
            labels_arousal = self.labels_arousal.read(file_name=file_name)
            labels_valence = self.labels_valence.read(file_name=file_name)

            self.read_pointer = self.read_pointer + 1
            return audio, fs, labels_arousal, labels_valence
        elif index >= 0:
            audio, fs = librosa.load(self.files[index], dtype='float32', sr=self.target_fs, mono=True)

            file_name = os.path.basename(self.files[index])
            labels_arousal = self.labels_arousal.read(file_name=file_name)
            labels_valence = self.labels_valence.read(file_name=file_name)
            return audio, fs, labels_arousal, labels_valence
        return -1

    def subset(self, contain_str=""):
        subset = []
        for file_name in self.files:
            if contain_str in file_name:
                subset.append(file_name)
        return self.__class__(files=subset, labels_arousal=self.labels_arousal, labels_valence=self.labels_valence)

    def batch_raw_audio_generator(self, window_size=5, hop_size=0.4, batch_size=128):
        num_files = self.get_number_files()
        while (1):
            idx_perm = np.random.permutation(num_files)
            for i in range(num_files):
                audio, fs, labels_arousal, labels_valence = self.read(index=idx_perm[i])
                window_pointer = 0
                audio_length = np.shape(audio)[0]

                time_steps = int(window_size / hop_size)
                feature_shape = int(fs * hop_size)
                features_for_batch = np.zeros((batch_size, time_steps, feature_shape))
                labels_for_batch = np.zeros((batch_size, 2))

                for j in range(batch_size):
                    if window_pointer > audio_length - fs * window_size:
                        break
                    else:
                        audio_matrix = np.reshape(audio[window_pointer:window_pointer + int(fs * window_size)], (time_steps, feature_shape))
                        features_for_batch[j, :, :] = audio_matrix

                        label_pointer = int(window_pointer / fs * hop_size + window_size / hop_size)
                        labels_for_batch[j, 0] = np.reshape(labels_arousal[label_pointer], (1,))
                        labels_for_batch[j, 1] = np.reshape(labels_valence[label_pointer], (1,))
                        window_pointer = window_pointer + int(fs * hop_size)

                features_for_batch = np.expand_dims(features_for_batch, 4)
                # print("Features shape:" + str(np.shape(features_for_batch)))
                # print("Labels shape:" + str(np.shape(labels_for_batch)))
                yield (features_for_batch, labels_for_batch)