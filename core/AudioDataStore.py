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
            labels = self.labels[self.read_pointer]
            self.read_pointer = self.read_pointer + 1
            return audio, fs, labels
        elif index >= 0:
            audio, fs = librosa.load(self.files[index], dtype='float32', sr=self.target_fs, mono=True)
            labels = self.labels[index]
            return audio, fs, labels
        return -1

    def subset(self, file_list=[], contain_str=""):
        subset = []
        labels = []
        if not file_list and contain_str is not "":
            for idx, file_name in enumerate(self.files):
                if contain_str in file_name:
                    subset.append(file_name)
                    if self.labels is not None:
                        labels.append(self.labels[idx])
            return self.__class__(files=subset, labels=labels)

        else:
            for file in file_list:
                idx_in_all_file = self.files.index(file)
                if self.labels is not None:
                    labels.append(self.labels[idx_in_all_file])
            return self.__class__(files=file_list, labels=labels)

    def batch_raw_audio_generator(self, window_size=5, hop_size=1, batch_size=128):
        num_files = self.get_number_files()
        while (1):
            idx_perm = np.random.permutation(num_files)
            for i in range(num_files):
                audio, fs, labels = self.read(index=idx_perm[i])
                label_size = np.shape(labels)[1]
                window_pointer = 0
                audio_length = np.shape(audio)[0]

                time_steps = int(window_size / hop_size)
                feature_shape = int(fs * hop_size)
                features_for_batch = np.zeros((batch_size, time_steps * feature_shape))
                labels_for_batch = np.zeros((batch_size, label_size))

                for j in range(batch_size):
                    if window_pointer > audio_length - fs * window_size:
                        break
                    else:
                        audio_matrix = np.reshape(audio[window_pointer:window_pointer + int(fs * window_size)], (time_steps * feature_shape))
                        features_for_batch[j, :] = audio_matrix
                        labels_for_batch[j, :] = np.reshape(labels, (-1, ))
                        window_pointer = window_pointer + int(fs * hop_size)

                features_for_batch = np.expand_dims(features_for_batch, 4)
                # print("Features shape:" + str(np.shape(features_for_batch)))
                # print("Labels shape:" + str(np.shape(labels_for_batch)))
                yield (features_for_batch, labels_for_batch)