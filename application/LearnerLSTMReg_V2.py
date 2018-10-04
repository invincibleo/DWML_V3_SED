#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 16.11.17 10:12
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : LearnerLSTMReg
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os
import tensorflow as tf
import keras
import shutil
import math
import matplotlib.pyplot as plt

from scipy.linalg import toeplitz
from keras.layers import Input, LSTM, Lambda
from core.Models import *
from core.Learner import Learner
from core.Metrics import *

class LearnerLSTMReg(Learner):
    def learn(self):
        model_json_file_addr, model_h5_file_addr = self.generate_pathes()
        continue_training = False
        if not os.path.exists(model_json_file_addr) or continue_training:
            self.copy_configuration_code()  # copy the configuration code so that known in which condition the model is trained
            input_shape = (5*22000, 1)
            # input_shape = self.input_shape
            # self.dataset.training_total_features = np.reshape(self.dataset.training_total_features, (-1,) + input_shape)
            # self.dataset.validation_total_features = np.reshape(self.dataset.validation_total_features, (-1,) + input_shape)

            model = SoundNet(input_shape)
            model.add(Flatten())

            model.add(Dense(1024))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(self.FLAGS.drop_out_rate))

            model.add(Dense(self.dataset.num_classes))
            model.add(Activation('sigmoid'))

            if continue_training:
                model.load_weights("tmp/model/" + self.hash_name_hashed + "/model.h5")  # load weights into new model

            if self.FLAGS.learning_rate_decay:
                decay_factor = self.FLAGS.learning_rate / self.FLAGS.epochs
            else:
                decay_factor = 0
            # Compile model
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(lr=self.FLAGS.learning_rate, decay=decay_factor),
                          metrics=['binary_accuracy', 'categorical_accuracy', top3_accuracy, 'top_k_categorical_accuracy'])

            # callbacks
            if tf.gfile.Exists('tmp/logs/tensorboard/' + str(self.hash_name_hashed)):
                shutil.rmtree('tmp/logs/tensorboard/' + str(self.hash_name_hashed))
            tensorboard = keras.callbacks.TensorBoard(
                log_dir='tmp/logs/tensorboard/' + str(self.hash_name_hashed),
                histogram_freq=0, write_graph=True, write_images=False, batch_size=self.FLAGS.train_batch_size)
            model_check_point = keras.callbacks.ModelCheckpoint(
                filepath='tmp/model/' + str(self.hash_name_hashed) + '/checkpoints/' + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                save_best_only=True,
                save_weights_only=True,
                mode='min')
            reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                     factor=0.1,
                                                                     patience=10,
                                                                     epsilon=0.0005)

            model.summary()

            train_generator = self.dataset.data_list['training'].batch_raw_audio_generator(window_size=5,
                                                                                           hop_size=1,
                                                                                           batch_size=self.FLAGS.train_batch_size)
            validation_generator = self.dataset.data_list['validation'].batch_raw_audio_generator(window_size=5,
                                                                                                  hop_size=1,
                                                                                                  batch_size=self.FLAGS.train_batch_size)
            hist = model.fit_generator(
                            generator=train_generator,
                            steps_per_epoch=int(self.dataset.num_training_data * 5 / self.FLAGS.train_batch_size),
                            epochs=self.FLAGS.epochs,
                            verbose=1,
                            callbacks=[tensorboard],
                            validation_data=validation_generator,
                            validation_steps=int(self.dataset.num_validation_data * 5 / self.FLAGS.train_batch_size),
                            shuffle=True,
                            )

            # save the model and training history
            self.save_model(hist, model)

    def predict(self):
        model = self.load_model_from_file()

        model_json_file_addr, model_h5_file_addr = self.generate_pathes()

        # load weights into new model
        model.load_weights(model_h5_file_addr)

        predictions_all = model.predict(self.dataset.validation_total_features,
                                        batch_size=self.FLAGS.test_batch_size,
                                        verbose=0)

        Y_all = self.dataset.validation_total_labels

        Y_all = np.reshape(Y_all, (-1, np.shape(Y_all)[-1]))
        predictions_all = np.reshape(predictions_all, (-1, np.shape(predictions_all)[-1]))

        return Y_all, predictions_all
