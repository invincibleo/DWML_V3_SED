#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 27.09.18 14:23
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : Dataset
# @Software: PyCharm Community Edition
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import abstractmethod

import os
import tensorflow as tf


class DataStore(object):
    def __init__(self, *args, **kwargs):
        self.ds_address = kwargs.get('ds_address', "")
        self.extension = kwargs.get('extension', None)
        files = kwargs.get('files', None)
        if files is None:
            file_glob = os.path.join(self.ds_address, '*.' + self.extension)
            self.files = tf.gfile.Glob(file_glob)
        else:
            self.files = files

        # sort the file list
        self.files.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))

        self.read_pointer = 0

    def subset(self, contain_str=""):
        subset = []
        for file_name in self.files:
            if contain_str in file_name:
                subset.append(file_name)
        return self.__class__(files=subset)

    def reset(self):
        self.read_pointer = 0

    def get_number_files(self):
        return len(self.files)

    @abstractmethod
    def read(self, *args, **kwargs):
        pass
