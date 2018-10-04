#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25.08.17 11:09
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : Dataset_Youtube8M
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tqdm import tqdm
import pickle
import numpy as np
import csv
import pandas
import hashlib

from core.Dataset_V2 import Dataset
from core.GeneralFileAccessor import GeneralFileAccessor
from core.TimeseriesPoint import *
from core.Preprocessing import *
from core.util import *
from core.ontologyProcessing import *
from core.AudioDataStore import *

class Dataset_Youtube8M(Dataset):
    
    def __init__(self, *args, **kwargs):
        dataset_name = 'Dataset_Youtube8M'
        super(Dataset_Youtube8M, self).__init__(self, dataset_name=dataset_name, *args, **kwargs)
        self.aso = {}
        self.using_existing_features = kwargs.get('using_existing_features', True)
        self.if_second_level_labels = kwargs.get('if_second_level_labels', False)
        self.data_list_meta = []
        self.label_list = self.get_label_list()
        self.num_classes = len(self.label_list)
        self.extensions = ['wav']
        self.data_list = self.create_data_list()

        if self.normalization:
            self.dataset_normalization()


    def get_label_list(self):
        ontology_file_addr = os.path.join(self.FLAGS.parameter_dir, 'ontology.json')

        csv_file_addr = os.path.join(self.FLAGS.parameter_dir, 'balanced_train_segments.csv')
        data_list_meta = pandas.read_csv(filepath_or_buffer=csv_file_addr,
                                         sep=' ',
                                         header=None,
                                         skiprows=3
                                         )
        self.data_list_meta = dict(zip(data_list_meta[0].str.replace(',', ''), data_list_meta[3].str.split(',')))

        if self.if_second_level_labels:
            second_level_class, self.aso = OntologyProcessing.get_2nd_level_label_name_list(ontology_file_addr)
            return sorted(second_level_class)
        else:
            self.aso = OntologyProcessing.get_label_name_list(ontology_file_addr)
            label_list_set = set()
            for values in self.data_list_meta.values():
                for label in values:
                    label_list_set.add(label)
            return sorted(list(label_list_set))

    def get_khot_label(self, audio_file_addr):
        file_name = os.path.basename(audio_file_addr).split('.')[0]
        label_name = self.data_list_meta[file_name]
        if self.if_second_level_labels:
            label_name_list = OntologyProcessing.get_2nd_level_class_label_index(label_name,
                                                                                         self.aso,
                                                                                         self.label_list)
        else:
            label_name_list = self.data_list_meta[file_name]

        label_name = ''
        label_content = np.zeros((1, self.num_classes))
        for class_name in label_name_list:
            label_content[0, self.label_list.index(class_name)] = 1
            label_name = label_name + self.aso[class_name]['name'] + ','

        return label_content, label_name

    def create_data_list(self):
        datalist_pickle_file = self.get_dataset_file_addr()
        if not tf.gfile.Exists(datalist_pickle_file) or not tf.gfile.Exists(self.feature_dir):
            datastore = AudioDataStore(ds_address=self.dataset_dir,
                                       target_fs=22000)
            audio_file_list = datastore.files
            datastore.labels = [self.get_khot_label(x)[0] for x in audio_file_list]

            num_files = len(audio_file_list)

            np.random.seed(1234)
            idx_perm = np.random.permutation(num_files)

            validation_file_idx = idx_perm[:int(num_files * self.FLAGS.validation_percentage / 100)]
            train_file_idx = idx_perm[int(num_files * self.FLAGS.validation_percentage / 100):]
            test_file_idx = []

            validation_file_list = [datastore.files[x] for x in validation_file_idx]
            train_file_list = [datastore.files[x] for x in train_file_idx]
            test_file_list = []

            dev_ds = datastore.subset(file_list=validation_file_list)
            train_ds = datastore.subset(file_list=train_file_list)
            test_ds = datastore.subset(file_list=[])


            data_list = {'validation': dev_ds, 'testing': test_ds, 'training': train_ds}

            pickle.dump(data_list, open(datalist_pickle_file, 'wb'), 2)
        else:
            data_list = pickle.load(open(datalist_pickle_file, 'rb'))

        # count data point
        self.num_training_data = len(data_list['training'].files)
        self.num_validation_data = len(data_list['validation'].files)
        self.num_testing_data = len(data_list['testing'].files)
        return data_list


