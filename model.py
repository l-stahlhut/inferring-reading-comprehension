"""
Building, training and evaluating a BiLSTM for the classification of scanpaths.
This code is based on https://github.com/aeye-lab/etra-reading-comprehension/blob/master/nn/model.py.
I intorduced a new model architecture and adapted the code to work for my problem setting.
"""
import argparse
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import sys
sys.path.append(os.getcwd())

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalMaxPool1D
from keras.layers import Input
from keras.layers import LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard


# This is the model I ended up using (worked best on SB-SAT dataset with all features)
def get_nn_model_lstm9(
        n_fixations,
        n_features,
):
    # model architecture
    model = Sequential()
    model.add(Bidirectional(LSTM(75, return_sequences=True), input_shape=(n_fixations, n_features))) # 394, 37
    model.add(Bidirectional(LSTM(75)))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = Adam()
    model.compile(
        optimizer=opt, loss='binary_crossentropy',
        metrics=['accuracy', auroc],
    )
    # return model.summary()
    return model


# from utils import feature_extraction as feature_extraction
# Model variations used for tuning
def get_nn_model_lstm(
        n_fixations,
        n_features,
):
    # model architecture
    model = Sequential()
    model.add(Bidirectional(LSTM(25, return_sequences=True), input_shape=(n_fixations, n_features))) # 394, 37
    model.add(Bidirectional(LSTM(25)))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = Adam()
    model.compile(
        optimizer=opt, loss='binary_crossentropy',
        metrics=['accuracy', auroc],
    )
    # return model.summary()
    return model

def get_nn_model_lstm2(
        n_fixations,
        n_features,
):
    # model architecture
    model = Sequential()
    model.add(Bidirectional(LSTM(25, return_sequences=True), input_shape=(n_fixations, n_features))) # 394, 37
    model.add(Bidirectional(LSTM(25)))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = Adam()
    model.compile(
        optimizer=opt, loss='binary_crossentropy',
        metrics=['accuracy', auroc],
    )
    # return model.summary()
    return model

def get_nn_model_lstm4(n_fixations, n_features, num_units=50):
    # model architecture
    model = Sequential()
    model.add(Bidirectional(LSTM(num_units, return_sequences=True), input_shape=(n_fixations, n_features)))  # 394, 37
    model.add(Bidirectional(LSTM(num_units)))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = Adam()
    model.compile(
        optimizer=opt, loss='binary_crossentropy',
        metrics=['accuracy', auroc],
    )

    return model

def get_nn_model_lstm5(n_fixations, n_features, num_units=50):
    # model architecture
    model = Sequential()
    model.add(Bidirectional(LSTM(num_units, return_sequences=True), input_shape=(n_fixations, n_features)))  # 394, 37
    model.add(Bidirectional(LSTM(num_units)))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = Adam()
    model.compile(
        optimizer=opt, loss='binary_crossentropy',
        metrics=['accuracy', auroc],
    )

    return model

def get_nn_model_lstm6(
        n_fixations,
        n_features,
):
    # model architecture
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(n_fixations, n_features))) # 394, 37
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(20, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = Adam()
    model.compile(
        optimizer=opt, loss='binary_crossentropy',
        metrics=['accuracy', auroc],
    )
    # return model.summary()
    return model

def get_nn_model_lstm7(
        n_fixations,
        n_features,
):
    # model architecture
    model = Sequential()
    model.add(Bidirectional(LSTM(25, return_sequences=True), input_shape=(n_fixations, n_features))) # 394, 37
    model.add(Bidirectional(LSTM(25)))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = Adam()
    model.compile(
        optimizer=opt, loss='binary_crossentropy',
        metrics=['accuracy', auroc],
    )
    # return model.summary()
    return model

def get_nn_model_lstm8(
        n_fixations,
        n_features,
):
    # model architecture
    model = Sequential()
    model.add(Bidirectional(LSTM(25, return_sequences=True), input_shape=(n_fixations, n_features))) # 394, 37
    model.add(Bidirectional(LSTM(25)))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = Adam()
    model.compile(
        optimizer=opt, loss='binary_crossentropy',
        metrics=['accuracy', auroc],
    )
    # return model.summary()
    return model



def get_nn_model_lstm10(
        n_fixations,
        n_features,
):
    # model architecture
    model = Sequential()
    model.add(Bidirectional(LSTM(125, return_sequences=True), input_shape=(n_fixations, n_features))) # 394, 37
    model.add(Bidirectional(LSTM(125)))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = Adam()
    model.compile(
        optimizer=opt, loss='binary_crossentropy',
        metrics=['accuracy', auroc],
    )

    return model

def help_roc_auc(y_true, y_pred):
    if len(np.unique(y_true)) == 1:
        return .5
    else:
        return roc_auc_score(y_true, y_pred)

# calculate the roc-auc as a metric

def auroc(y_true, y_pred):
    return tf.py_function(help_roc_auc, (y_true, y_pred), tf.double)


def train_nn(
    spit_criterions, label, label_dict, # book, page, subj; binary_score
    feature_names_per_word, # depends on dataset (boolean flags, argparse
    model_name, # todo
    dataset_name,
    dataset_version,
    s1_rm1_lf1=True,
    s1_rm1_lf0=True,
    s1_rm1_lf1_pos_cont=True,
    s1_rm0_lf0=True,
    s0_rm1_lf0=True,
    s0_rm0_lf1=True,
    flag_redo=False,
    normalize_flag=True,
    patience=50, # todo
    batch_size=256,
    epochs=1000,
    save_dir='nn/results',  # todo
    weights_dir='nn/model_weights',
    save_csv=True,
    save_joblib=False,
):
    for split_criterion in spit_criterions:
        print("Training for split criterion ", split_criterion)
        # for label in labels:

        model_prefix = str(s1_rm1_lf1) +\
            '_' + str(s1_rm1_lf0) +\
            '_' + str(s1_rm1_lf1_pos_cont) +\
            '_' + str(s1_rm0_lf0) +\
            '_' + str(s0_rm1_lf0) +\
            '_' + str(s0_rm0_lf1) +\
            '_'
        csv_save_path = f'{save_dir}/{dataset_name}_{dataset_version}_{split_criterion}_{label}_64_tanh.csv'
        # csv_save_path = f'{save_dir}{model_prefix}{model_name}_{split_criterion}_text_sequence_{label}.csv'  # noqa: E501
        joblib_save_path = csv_save_path.replace('.csv', '.joblib')
        if not flag_redo and save_csv and os.path.exists(csv_save_path):
            continue
        if not flag_redo and save_joblib and os.path.exists(joblib_save_path):
            continue

        DATA_PATH = os.path.join('nn', dataset_name, "data_" + dataset_version, split_criterion)
        # SB_SAT_PATH = f'paper_splits/{split_criterion}/' # todo

        split_criterion_dict = {
            'subj': 0,
            'book': 1,
            'subj-book': 0, # todo?? book-page?
        }

        if (split_criterion == 'book') or (split_criterion == 'subj-book'):
            num_folds = 4
        else:
            num_folds = 5 # book page, subj

        # initialize dataframe for results
        pd_init = pd.DataFrame(
            columns=[
                'ahn_baseline', # TODO why ahn
                'fold0_auc', 'fold1_auc', 'fold2_auc', 'fold3_auc', 'fold4_auc',
                'fold0_tpr', 'fold1_tpr', 'fold2_tpr', 'fold3_tpr', 'fold4_tpr',
                'fold0_fpr', 'fold1_fpr', 'fold2_fpr', 'fold3_fpr', 'fold4_fpr',
                'fold0_y_pred', 'fold1_y_pred', 'fold2_y_pred', 'fold3_y_pred', 'fold4_y_pred',
                'fold0_y_test', 'fold1_y_test', 'fold2_y_test', 'fold3_y_test', 'fold4_y_test',
                'avg_auc', 'std_auc',
            ],
        )
        out_dict = dict()

        pd_init['ahn_baseline'] = [model_name] # TODO

        for fold in range(num_folds):
            print("fold: ", fold)
            # set seed value of numpy random number generator and the built-in python random number generator to fixed
            # value -> the seq of random numbers will always be the same. Useful in ML since it allows for results to be
            # reproduced and compared accross different runs of the model
            # each fold in the cross validation process will use a different seed value to ensure that the train and val
            # sets are different for each fold, but by using the same seed value for each folds, the results can still
            # be reproduced and compared across different folds.
            np.random.seed(fold)
            random.seed(fold)

            # collect the inputs for train, validation and test

            # train and val data
            X_train_path = os.path.join(
                DATA_PATH, f'X_train_{split_criterion}_{fold}.npy',
            )

            y_train_path = os.path.join(
                DATA_PATH, f'y_train_{split_criterion}_{fold}.npy',
            )
            # load np arrays and assign to variables
            x_train_all, y_train_all = np.load(X_train_path), np.load(
                y_train_path, allow_pickle=True,
            )
            # print(x_train_all.shape, "\n", y_train_all.shape)  # (1672, 394, 37); (1672, 3) -> the last one is the score


            # x_train_fix_all = np.load(X_train_fix_path)
            # x_train_fix_postions = x_train_fix_all[:, :, 4]
            # normalize the input features of the ml model using the MinMaxScaler class -> scale the input features to a fixed range between 0 and 1. Input features may have different scales and ranges
            # which could cause poor performance.
            if normalize_flag:
                scaler = MinMaxScaler()
                fix_scaler = MinMaxScaler()
                x_train_all = scaler.fit_transform(
                    x_train_all.reshape(-1, x_train_all.shape[-1]),
                ).reshape(x_train_all.shape)


            if split_criterion != 'book':
                outer_cv = KFold(
                    n_splits=4, shuffle=True,
                    random_state=fold,
                )
            else:
                outer_cv = KFold(
                    n_splits=3, shuffle=True,
                    random_state=fold,
                )
            # print(split_criterion) # book-page
            # print(y_train_all)
            # print(y_train_all.shape) # 1672, 3
            # print("Label dict:", label_dict) # {'subject_id': 0, 'text_id': 1, 'binary_score': 2}
            # print("label:", label) # binary_score
            # print("label_dict[label]", label_dict[label]) # 2
            # print(y_train_all[:, label_dict[label]].shape) # array with all labels -> shape (1672,)
            # print("Unique values: ", np.unique(y_train_all[:, label_dict[label]], return_counts=True)) #  array([1092,  580]))

            # print(y_train_all)
            # print(split_criterion_dict[split_criterion])
            # return None

            # split criterion book: splitkeys is book id (e.g. 1 for dickens)
            # splikeys = [1. 2. 3. 4.]
            # split crierion subject: splitkeys is subject id (e.g. 1 for subj 1)
            # splikeys = [  1.   2.   3.   4.   5.  ... 106. 107.]
            # split crterion book-page: splitkeys is an array of all labels [0. 1. 0. 0. ....0] for binary_score
            # splitkeys = [0. 0. 0. ... 0. 0. 1.]
            if split_criterion != 'book-page':
                splitkeys = np.array(
                    sorted(
                        list(
                            set(
                                y_train_all[
                                    :,
                                    split_criterion_dict[split_criterion],  # subj:0, book: 1
                                ],
                            ),
                        ),
                    ),
                )
            else:
                splitkeys = y_train_all[:, label_dict[label]] # array for all scores, eg shape 1672

            for train_idx, val_idx in outer_cv.split(splitkeys):
                break

            if split_criterion != 'book-page':
                N_train_sub = splitkeys[train_idx]
                N_test_sub = splitkeys[val_idx]

                train_idx = np.where(
                    np.isin(
                        y_train_all[
                            :, split_criterion_dict[split_criterion],
                        ], N_train_sub,
                    ),
                )[0]
                val_idx = np.where(
                    np.isin(
                        y_train_all[
                            :, split_criterion_dict[split_criterion],
                        ], N_test_sub,
                    ),
                )[0]
            x_train = x_train_all[train_idx]
            y_train = y_train_all[train_idx]
            x_val = x_train_all[val_idx]
            y_val = y_train_all[val_idx]
            y_train_all[val_idx]

            # train + val = 1672 -> val is 25%
            # print("x_val: ",x_val.shape)  # (418, 394, 37)
            # print("y_val: ",y_val.shape)  # (418, 3)
            # print("x_train:", x_train.shape)  # (1254, 394, 37)
            # print("y_train: ", y_train.shape)  # (1254, 3)
            # return None


            y_train = np.array(y_train[:, label_dict[label]], dtype=int)  # shape (1254,)
            y_val = np.array(y_val[:, label_dict[label]], dtype=int)  # shape (418,)

            # print("x_val: ", x_val.shape)  # (418, 394, 37)
            # print("y_val: ", y_val.shape)  # (418,)
            # print("x_train:", x_train.shape)  # (1254, 394, 37)
            # print("y_train: ", y_train.shape)  # (1254,)

            # Test Data
            X_test_path = os.path.join(
                DATA_PATH,
                f'X_test_{split_criterion}_{fold}.npy',
            )
            # X_test_fix_path = os.path.join(
            #     DATA_PATH,
            #     f'X_test_{split_criterion}_{fold}_fix_data.npy',
            # )
            y_test_path = os.path.join(
                DATA_PATH,
                f'y_test_{split_criterion}_{fold}.npy',
            )
            x_test_all, y_test_all = np.load(X_test_path), np.load( # shapes (418, 394, 37) and (418, 3) respectively
                y_test_path, allow_pickle=True,
            )

            if normalize_flag:
                x_test_all = scaler.transform(
                    x_test_all.reshape(-1, x_test_all.shape[-1]),
                ).reshape(x_test_all.shape)

            y_test = np.array(y_test_all[:, label_dict[label]], dtype=int) # (418,)

            n_samples_train, n_timesteps, n_features = x_train.shape
            x_train_2d = x_train.reshape(n_samples_train, n_timesteps * n_features)

            n_samples_val, n_timesteps, n_features = x_val.shape
            x_val_2d = x_val.reshape(n_samples_val, n_timesteps * n_features)

            n_samples_test, n_timesteps, n_features = x_test_all.shape
            x_test_2d = x_test_all.reshape(n_samples_test, n_timesteps * n_features)

            input_scaler = MinMaxScaler()
            x_train_scaled = input_scaler.fit_transform(x_train_2d)
            x_val_scaled = input_scaler.transform(x_val_2d)
            x_test_scaled = input_scaler.transform(x_test_2d)

            # reshape the scaled data back to the original shape
            x_train = x_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
            x_val = x_val_scaled.reshape(n_samples_val, n_timesteps, n_features)
            x_test = x_test_scaled.reshape(n_samples_test, n_timesteps, n_features)

            # print(x_train.shape)
            # print("x_train:", x_train.shape)
            # print("y_train:", y_train.shape)
            # print("x_val: ", x_val.shape)
            # print("y_val: ", y_val.shape)
            # print("x test:", x_test.shape)
            # print("y test:", y_test.shape)

            n_fixations = x_train.shape[1]
            n_features = x_train.shape[2]

            model = get_nn_model_lstm9(n_fixations, n_features)  # todo

            #Set up keras session for training nn
            # clear current keras session (free up resources, help pavoid issues with variable name)
            tf.keras.backend.clear_session()
            # monitor validation loss during training; stop training process if validation loss does not imporve fro patience number of epochs
            weights_directory =f'{weights_dir}/{dataset_name}/{dataset_name}_{dataset_version}_{split_criterion}_{label}_256weights.h5'
            callbacks = [
                ModelCheckpoint(weights_directory, save_best_only=True),
                # ModelCheckpoint(os.path.join('nn/model_weights/SBSAT', 'SBSAT_s1_rm1_lf1.h5'), save_best_only=True),
                # ModelCheckpoint(os.path.join('srv/scratch3/laura/nn/model_weights/', dataset_name, dataset_version, 'model7_weights.h5'), save_best_only=True),
                EarlyStopping(
                    monitor='val_loss', patience=patience,
                ),
                TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
            ]
            history = model.fit(  # noqa: F841
                x_train, y_train,
                validation_data=(
                    x_val,
                    y_val,
                ),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=0,
            )

            num_epochs_trained = len(history.epoch)
            print("Number of epochs trained:", num_epochs_trained)

            y_pred = model.predict(
                x_test,
                batch_size=batch_size,
            )
            try:
                fpr, tpr, _ = metrics.roc_curve(
                    y_test,
                    y_pred,
                    pos_label=1,
                )
                auc = metrics.auc(fpr, tpr)
                print("auc: ", auc)
                pd_init[f'fold{fold}_auc'] = auc
                pd_init[f'fold{fold}_tpr'] = [tpr]
                pd_init[f'fold{fold}_fpr'] = [fpr]
                pd_init[f'fold{fold}_y_test'] = [y_test]
                pd_init[f'fold{fold}_y_pred'] = [y_pred]

                out_dict[f'fold{fold}_auc'] = auc
                out_dict[f'fold{fold}_tpr'] = [tpr]
                out_dict[f'fold{fold}_fpr'] = [fpr]
                out_dict[f'fold{fold}_y_test'] = [y_test]
                out_dict[f'fold{fold}_y_pred'] = [y_pred]
            except KeyError:
                try:
                    fpr, tpr, _ = metrics.roc_curve(
                        y_test,
                        y_pred,
                        pos_label=1,
                    )
                    auc = metrics.auc(fpr, tpr)
                    print("auc: ", auc)
                    pd_init[f'fold{fold}_auc'] = auc
                    pd_init[f'fold{fold}_tpr'] = [tpr]
                    pd_init[f'fold{fold}_fpr'] = [fpr]
                    pd_init[f'fold{fold}_y_test'] = y_test
                    pd_init[f'fold{fold}_y_pred'] = y_pred

                    out_dict[f'fold{fold}_auc'] = auc
                    out_dict[f'fold{fold}_tpr'] = [tpr]
                    out_dict[f'fold{fold}_fpr'] = [fpr]
                    out_dict[f'fold{fold}_y_test'] = y_test
                    out_dict[f'fold{fold}_y_pred'] = y_pred
                except KeyError as e:
                    raise e

        pd_init['avg_auc'] = 0
        out_dict['avg_auc'] = 0
        for i in range(num_folds):
            pd_init['avg_auc'] += pd_init[f'fold{i}_auc']
            out_dict['avg_auc'] += out_dict[f'fold{i}_auc']
        pd_init['avg_auc'] /= num_folds
        out_dict['avg_auc'] /= num_folds

        pd_init['std_auc'] = 0
        out_dict['std_auc'] = 0
        for i in range(0, num_folds):
            pd_init['std_auc'] += (pd_init[f'fold{i}_auc'] - pd_init['avg_auc'])**2
            out_dict['std_auc'] += (
                out_dict[f'fold{i}_auc'] - out_dict['avg_auc']
            )**2
        pd_init['std_auc'] = (pd_init['std_auc'] / num_folds)**(1 / 2)
        out_dict['std_auc'] = (out_dict['std_auc'] / num_folds)**(1 / 2)
        if save_csv:
            pd_init.to_csv(csv_save_path, index=None)
        if save_joblib:
            joblib.dump(out_dict, joblib_save_path, compress=3, protocol=2)
        print('mean auc: ' + str(pd_init['avg_auc']))


def convert_string_to_boolean(input_string):
    if input_string == 'True':
        return True
    return False


def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-GPU', '--GPU', type=int, default=4)
    parser.add_argument('-s1_rm1_lf1', '--s1_rm1_lf1', action='store_true', default='True') # todo see if it still works in train_model
    parser.add_argument('-s1_rm1_lf0', '--s1_rm1_lf0', action='store_true', default='True')
    parser.add_argument('-s1_rm1_lf1_pos_cont', '--s1_rm1_lf1_pos_cont', action='store_true', default='True')
    parser.add_argument('-s1_rm0_lf0', '--s1_rm0_lf0', action='store_true', default='True')
    parser.add_argument('-s0_rm1_lf0', '--s0_rm1_lf0', action='store_true', default='True')
    parser.add_argument('-s0_rm0_lf1', '--s0_rm0_lf1', action='store_true', default='True')

    parser.add_argument('-save_dir', '--save_dir', type=str, default='True')
    parser.add_argument('-weights_dir', '--weights_dir', type=str, default='True')
    parser.add_argument('--SBSAT', action='store_true', help='English version: If argument is given, model '
                                                             'will be trained with SB-SAT dataset.')
    parser.add_argument('--InDiCo', action='store_true', help='German version: If argument is given, model '
                                                              'model will be trained with InDiCo dataset.')

    args = parser.parse_args()

    # set configuraitons for the model training process
    GPU = args.GPU
    s1_rm1_lf1 = convert_string_to_boolean(args.s1_rm1_lf1)
    s1_rm1_lf0 = convert_string_to_boolean(args.s1_rm1_lf0)
    s1_rm1_lf1_pos_cont = convert_string_to_boolean(args.s1_rm1_lf1_pos_cont)
    s1_rm0_lf0 = convert_string_to_boolean(args.s1_rm0_lf0)
    s0_rm1_lf0 = convert_string_to_boolean(args.s0_rm1_lf0)
    s0_rm0_lf1 = convert_string_to_boolean(args.s0_rm0_lf1)

    save_dir = args.save_dir
    weights_dir = args.weights_dir

    # select graphic card
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # TODO
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    tf_session = tf.compat.v1.Session(config=config)  # noqa: F841

    # determine which dataset to use
    if args.SBSAT is True:
        dataset_name = 'sbsat_splits'
    elif args.InDiCo is True:
        dataset_name = 'indico_splits'
    # determine which version of the features to use
    if args.s1_rm1_lf1 is True:
        dataset_version = "s1_rm1_lf1"
    elif args.s1_rm1_lf0 is True:
        dataset_version = "s1_rm1_lf0"
    elif args.s1_rm1_lf1_pos_cont is True:
        dataset_version = "s1_rm1_lf1_pos_cont"
    elif args.s1_rm0_lf0 is True:
        dataset_version = "s1_rm0_lf0"
    elif args.s0_rm1_lf0 is True:
        dataset_version = "s0_rm1_lf0"
    elif args.s0_rm0_lf1 is True:
        dataset_version = "s0_rm0_lf1"

    normalize_flag = True

    flag_redo = True  # todo
    patience = 50  # todo
    # patience = 5
    # batch_size = 256  # todo
    # batch_size = 128
    batch_size = 64
    epochs = 1000  # todo
    # epochs = 30

    spit_criterions = ['book-page', 'subj', 'book']
    # labels = ['subj_acc_level', 'acc_level', 'native', 'difficulty'] # todo
    label = "binary_score"
    # model_name = 'nn_paul' # todo
    model_name = 'nn_laura'

    # all features (RQ1)
    if args.s1_rm1_lf1 is True:
        feature_names_per_word = ['CURRENT_FIX_X', 'CURRENT_FIX_Y', 'CURRENT_FIX_PUPIL', 'CURRENT_FIX_DURATION', 'ffd',
                                  'tfd', 'n_fix', 'fpr', 'incoming_sac_regressive_norm', 'outgoing_sac_regressive_norm',
                                  'outgoing_sac_progressive_norm', 'simplified_pos_A', 'simplified_pos_N',
                                  'simplified_pos_FUNC', 'simplified_pos_VERB', 'content_word', 'synonym_homonym',
                                  'NE_IOB', 'n_rights', 'n_lefts', 'dep_distance', 'synt_surprisal', 'surprisal',
                                  'wordfreq_lemma', 'tf_idf', 'lex_overlap', 'pron_det_to_noun_ratio', 'voice',
                                  'word_n_char', 'sent_n_words', 't_n_phrases', 'sent_mean_word_length',
                                  'sent_lexical_density_tokens', 't_lexical_density', 't_lemma_TTR', 't_content_w_TTR',
                                  't_function_w_TTR','semantic_similarity_adjacent_sent', 'sent_cut', 't_genre']
    # no linguistic features (RQ1)
    elif args.s1_rm1_lf0 is True:
        feature_names_per_word = ['CURRENT_FIX_X', 'CURRENT_FIX_Y', 'CURRENT_FIX_PUPIL', 'CURRENT_FIX_DURATION', 'ffd',
                                  'tfd', 'n_fix', 'fpr', 'incoming_sac_regressive_norm', 'outgoing_sac_regressive_norm',
                                  'outgoing_sac_progressive_norm']
    # only simplified POS and content word indicator as features (RQ2)
    elif args.s1_rm1_lf1_pos_cont is True:
        feature_names_per_word = ['CURRENT_FIX_X', 'CURRENT_FIX_Y', 'CURRENT_FIX_PUPIL', 'CURRENT_FIX_DURATION', 'ffd',
                                  'tfd', 'n_fix', 'fpr', 'incoming_sac_regressive_norm', 'outgoing_sac_regressive_norm',
                                  'outgoing_sac_progressive_norm', 'simplified_pos_A', 'simplified_pos_N',
                                  'simplified_pos_FUNC', 'simplified_pos_VERB', 'content_word']
    # Only sequence features (ablation study)
    elif args.s1_rm0_lf0 is True:
        feature_names_per_word = ['CURRENT_FIX_X', 'CURRENT_FIX_Y', 'CURRENT_FIX_PUPIL', 'CURRENT_FIX_DURATION']
    # Only reading measures (ablation study)
    elif args.s0_rm1_lf0 is True:
        feature_names_per_word = ['ffd', 'tfd', 'n_fix', 'fpr', 'incoming_sac_regressive_norm',
                                  'outgoing_sac_regressive_norm', 'outgoing_sac_progressive_norm']
    # Only linguistic features (ablation study)
    elif args.s0_rm0_lf1 is True:
        feature_names_per_word = ['CURRENT_FIX_X', 'CURRENT_FIX_Y', 'CURRENT_FIX_PUPIL', 'CURRENT_FIX_DURATION', 'ffd',
                                  'tfd', 'n_fix', 'fpr', 'incoming_sac_regressive_norm', 'outgoing_sac_regressive_norm',
                                  'outgoing_sac_progressive_norm', 'simplified_pos_A', 'simplified_pos_N',
                                  'simplified_pos_FUNC', 'simplified_pos_VERB', 'content_word', 'synonym_homonym',
                                  'NE_IOB', 'n_rights', 'n_lefts', 'dep_distance', 'synt_surprisal', 'surprisal',
                                  'wordfreq_lemma', 'tf_idf', 'lex_overlap', 'pron_det_to_noun_ratio', 'voice',
                                  'word_n_char', 'sent_n_words', 't_n_phrases', 'sent_mean_word_length',
                                  'sent_lexical_density_tokens', 't_lexical_density', 't_lemma_TTR', 't_content_w_TTR',
                                  't_function_w_TTR','semantic_similarity_adjacent_sent', 'sent_cut', 't_genre']

    #label ditionary 0: subject, 1: text, 2: score
    if args.SBSAT is True:
        with open('nn/sbsat_splits/labels_dict.json') as fp:
            label_dict = json.load(fp)
    elif args.InDiCo is True:
        with open('nn/indico_splits/labels_dict.json') as fp:
            label_dict = json.load(fp)


    train_nn(
        spit_criterions=spit_criterions, # book-page, subj, book
        label=label, # binary_score
        label_dict=label_dict,
        feature_names_per_word=feature_names_per_word,  # e.g. s1_rm1_lf1
        model_name=model_name,# todo
        dataset_name=dataset_name, #indico or sbsat
        dataset_version=dataset_version,  # e.g. data_s1_rm1_lf1
        s1_rm1_lf1=s1_rm1_lf1,
        s1_rm1_lf0=s1_rm1_lf0,
        s1_rm1_lf1_pos_cont=s1_rm1_lf1_pos_cont,
        s1_rm0_lf0=s1_rm0_lf0,
        s0_rm1_lf0=s0_rm1_lf0,
        s0_rm0_lf1=s0_rm0_lf1,
        flag_redo=flag_redo,
        normalize_flag=normalize_flag,
        # use_gaze_entropy_features=use_gaze_entropy_features,#todo
        patience=patience, # number of epochs wherein val loss does not improve before train process is stopped
        batch_size=batch_size,
        epochs=epochs,
        save_dir=save_dir,
        weights_dir=weights_dir,
        save_csv=True,
        save_joblib=True,
    )


if __name__ == '__main__':
    raise SystemExit(main())
