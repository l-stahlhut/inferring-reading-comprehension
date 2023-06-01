"""
Transform data as nn input: Encode data and generate split in 3 cross validation settings.
The funcitons 'extract_features' & 'write_npys' are taken from the following script:
https://github.com/aeye-lab/etra-reading-comprehension/blob/master/utils/generate_text_sequence_splits.py
Adapted and extendend by me.

How to use the scrpit: Chose the dataset and the amount of features to include in the reuting array with the binary
command line arguments. Example:
$ python3 src/encode_data.pa --SBSAT --s1_rm1_lf1
$ python3 src/encode_data.pa --InDiCo --s0_rm1_lf0

1) How good are my linguistic features? s1_rm1_lf1 vs. s1_rm1_lf0
1, Ablation study: s1_rm0_lf0, s0_rm1_lf0, s0_rm0_lf1
2) How good is my model? s1_rm1_lf1_pos_cont

example for how to use the script:
$ python nn/model.py -GPU 4 --s1_rm0_lf0 --SBSAT --save_dir nn/results/SBSAT/s1_rm0_lf0

"""


import pandas as pd
from typing import Dict, Collection, List, Tuple
from tqdm import tqdm
import re
import numpy as np
import torch
import argparse
import os
from sklearn.model_selection import KFold
import random
import json
import joblib

parser = argparse.ArgumentParser(
    prog='encode_data',
    description='Transform data as NN input')
parser.add_argument('--SBSAT', action='store_true', help='English version: If argument is given, SB-SAT '
                                                         'stimulus texts will be encoded.')
parser.add_argument('--InDiCo', action='store_true', help='German version: If argument is given, InDiCo '
                                                          'stimulus texts will be encoded.')
#RQ 1: Do the linguistic features improve the performance of my model?
parser.add_argument('--s1_rm1_lf1', action='store_true',
                    help='Data split. Creates an np array of data with the specified features: '
                         'Scanpath features: True, reading measures: True, linguistic features: True.')
parser.add_argument('--s1_rm1_lf0', action='store_true',
                    help='Data split. Creates an np array of data with the specified features: '
                         'scanpath features: True, reading measures: True, linguistic features: False')
# Ablation study: What is the impact of the components (scanpath, reading measures, linguistic features?)
parser.add_argument('--s1_rm0_lf0', action='store_true',
                    help='Data split. Creates an np array of data with the specified features: '
                         'scanpath features: True, reading measures: False, linguistic features: False')
parser.add_argument('--s0_rm1_lf0', action='store_true',
                    help='Data split. Creates an np array of data with the specified features: '
                         'scanpath features: False, reading measures: True, linguistic features: False')
parser.add_argument('--s0_rm0_lf1', action='store_true',
                    help='Data split. Creates an np array of data with the specified features: '
                         'scanpath features: False, reading measures: False, linguistic features: True')
# RQ 2: Is it better to use one lstm than davids architecture with multiple subnets?
parser.add_argument('--s1_rm1_lf1_pos_cont', action='store_true',
                    help='Data split. Creates an np array of data with the specified features: '
                         'scanpath features: True, reading measures: True, linguistic features: simplified PoS and '
                         'Content Word Feature only.')

args = parser.parse_args()


class EncodedData():
    def __init__(self, path, label_path):
        self.path = path
        self.data = self.load_data(path)[0]  # columns before feature engineering
        self.features = self.load_data(path)[1]
        self.labels = self.get_label(label_path)[0]
        self.label_dict = self.get_label(label_path)[1]
        # self.data_encoded = self.encode_features()  # everything encoded except for word
        pass

    def get_label(self, label_path):
        """Create a dictionary that maps (subject_id, text_id) to the number of correctly answered questions."""
        columns = ['subject_id', 'text_id', 'binary_score']
        df = pd.read_csv(label_path, sep='\t', usecols=columns)

        if args.SBSAT is True:
            # transfrom subject_id label msd001 -> 1
            df['subject_id'] = df['subject_id'].apply(lambda x: int(re.findall('\d+', x)[0]))

        label_dict = {label: idx for idx, label in enumerate(df.columns.tolist())}
        # return label_dict
        #
        # return df.head()
        labels = {}
        for index, row in df.iterrows():
            subject_id = int(row['subject_id'])
            text_id = int(row['text_id'])
            # n_correct_answers = int(row['n_correct_answers'])
            binary_score = int(row['binary_score'])

            label = (subject_id, text_id)

            if label not in labels:
                labels[label] = binary_score

        l = [[k[0], k[1], v] for k, v in labels.items()]   # subject_id, text_id, label
        d_labels = {k: v for k, v in zip(labels.keys(), l)}

        return d_labels, label_dict

    def load_data(self, path: str) -> pd.DataFrame:
        """Load linguistically annotated fixation dataset. Encode nominal and binary features."""
        # RQ1: without linguistic features
        if args.s1_rm1_lf0 is True:
            # for baseline: classification without linguistic features
            columns = ['subject_id', 'text_id', 'screen_id', 'fixation_id', 'word', 'CURRENT_FIX_X', 'CURRENT_FIX_Y',
                       'CURRENT_FIX_PUPIL', 'CURRENT_FIX_DURATION', 'ffd', 'tfd', 'n_fix', 'fpr',
                       'incoming_sac_regressive_norm', 'outgoing_sac_regressive_norm', 'outgoing_sac_progressive_norm']
            df = pd.read_csv(path, usecols=columns)
            # add text_screen id
            df['text_screen_id'] = df['text_id'].astype(str) + '_' + df['screen_id'].astype(str)
            columns_reordered = ['subject_id', 'text_id', 'screen_id', 'text_screen_id', 'fixation_id', 'word',
                                 'CURRENT_FIX_X', 'CURRENT_FIX_Y', 'CURRENT_FIX_PUPIL', 'CURRENT_FIX_DURATION', 'ffd',
                                 'tfd', 'n_fix', 'fpr', 'incoming_sac_regressive_norm', 'outgoing_sac_regressive_norm',
                                 'outgoing_sac_progressive_norm']

            df = df.reindex(columns=columns_reordered)

        # RQ2: only pos and content word feature
        elif args.s1_rm1_lf1_pos_cont is True:
            columns = ['subject_id', 'text_id', 'screen_id', 'fixation_id', 'word', 'CURRENT_FIX_X', 'CURRENT_FIX_Y',
                       'CURRENT_FIX_PUPIL', 'CURRENT_FIX_DURATION', 'ffd', 'tfd', 'n_fix', 'fpr',
                       'incoming_sac_regressive_norm', 'outgoing_sac_regressive_norm', 'outgoing_sac_progressive_norm',
                       'simplified_pos', 'content_word']

            dtypes = {'content_word': 'int'}

            df = pd.read_csv(path, dtype=dtypes, usecols=columns)


            # One hot encode other nominal features
            nominal_features = ['simplified_pos']
            df = self.one_hot_encode(df, nominal_features)


            # add text_screen id
            df['text_screen_id'] = df['text_id'].astype(str) + '_' + df['screen_id'].astype(str)

            columns_reordered = ['subject_id', 'text_id', 'screen_id', 'text_screen_id', 'fixation_id', 'word',
                                 'CURRENT_FIX_X', 'CURRENT_FIX_Y', 'CURRENT_FIX_PUPIL', 'CURRENT_FIX_DURATION', 'ffd',
                                 'tfd', 'n_fix', 'fpr', 'incoming_sac_regressive_norm', 'outgoing_sac_regressive_norm',
                                 'outgoing_sac_progressive_norm', 'simplified_pos_A', 'simplified_pos_N',
                                 'simplified_pos_FUNC', 'simplified_pos_VERB', 'content_word']

            df = df.reindex(columns=columns_reordered)

        # RQ1: all linguistic features / ablation study
        elif args.s1_rm1_lf1 is True or args.s1_rm0_lf0 is True or args.s0_rm1_lf0 or args.s0_rm0_lf1 is True:
            columns = ['subject_id', 'text_id', 'screen_id', 'fixation_id', 'word', 'CURRENT_FIX_X', 'CURRENT_FIX_Y',
                       'CURRENT_FIX_PUPIL', 'CURRENT_FIX_DURATION', 'ffd', 'tfd', 'n_fix', 'fpr',
                       'incoming_sac_regressive_norm', 'outgoing_sac_regressive_norm', 'outgoing_sac_progressive_norm',
                       'simplified_pos', 'content_word','synonym_homonym', 'NE_IOB',
                       'n_rights', 'n_lefts', 'dep_distance', 'synt_surprisal', 'surprisal', 'wordfreq_lemma', 'tf_idf',
                       'lex_overlap', 'semantic_similarity_adjacent_sent', 'pron_det_to_noun_ratio', 'voice', 'word_n_char',
                       'sent_n_words', 't_n_phrases', 'sent_mean_word_length', 'sent_lexical_density_tokens',
                       't_lexical_density', 't_lemma_TTR', 't_content_w_TTR', 't_function_w_TTR',
                       'semantic_similarity_adjacent_sent', 'sent_cut', 't_genre']


            dtypes = {'word_n_char': 'int',
                      'content_word': 'int',
                      'synonym_homonym': 'int',
                      'lex_overlap': 'int',
                      'sent_cut': 'int'}

            df = pd.read_csv(path, dtype=dtypes, usecols=columns)

            # transform IOB tags into binary feature
            df['NE_IOB'] = df['NE_IOB'].map({'I': 1, 'O': 0, 'B': 1})
            # One hot encode other nominal features
            nominal_features = ['simplified_pos']
            df = self.one_hot_encode(df, nominal_features)


            # transform values for binary features if not already done
            binary_features = ['voice', 't_genre']
            df = self.encode_binary_features(df, binary_features)

            # add text_screen id
            df['text_screen_id'] = df['text_id'].astype(str) + '_' + df['screen_id'].astype(str)

            columns_reordered = ['subject_id', 'text_id', 'screen_id', 'text_screen_id', 'fixation_id', 'word',
                                 'CURRENT_FIX_X', 'CURRENT_FIX_Y', 'CURRENT_FIX_PUPIL', 'CURRENT_FIX_DURATION', 'ffd',
                                 'tfd', 'n_fix', 'fpr', 'incoming_sac_regressive_norm', 'outgoing_sac_regressive_norm',
                                 'outgoing_sac_progressive_norm', 'simplified_pos_A', 'simplified_pos_N',
                                 'simplified_pos_FUNC', 'simplified_pos_VERB', 'content_word', 'synonym_homonym',
                                 'NE_IOB', 'n_rights', 'n_lefts', 'dep_distance', 'synt_surprisal',
                                 'surprisal', 'wordfreq_lemma', 'tf_idf', 'lex_overlap', 'pron_det_to_noun_ratio',
                                 'voice', 'word_n_char', 'sent_n_words', 't_n_phrases', 'sent_mean_word_length',
                                 'sent_lexical_density_tokens', 't_lexical_density', 't_lemma_TTR', 't_content_w_TTR',
                                 't_function_w_TTR', 'semantic_similarity_adjacent_sent', 'sent_cut', 't_genre']

            df = df.reindex(columns=columns_reordered)

            # ablation study
            if args.s1_rm0_lf0 is True:
                df = df.drop(['ffd','tfd', 'n_fix', 'fpr', 'incoming_sac_regressive_norm',
                              'outgoing_sac_regressive_norm', 'outgoing_sac_progressive_norm', 'simplified_pos_A',
                              'simplified_pos_N', 'simplified_pos_FUNC', 'simplified_pos_VERB',
                              'content_word', 'synonym_homonym', 'NE_IOB', 'n_rights', 'n_lefts', 'dep_distance',
                              'synt_surprisal','surprisal', 'wordfreq_lemma', 'tf_idf', 'lex_overlap',
                              'pron_det_to_noun_ratio', 'voice', 'word_n_char', 'sent_n_words', 't_n_phrases',
                              'sent_mean_word_length', 'sent_lexical_density_tokens', 't_lexical_density',
                              't_lemma_TTR', 't_content_w_TTR', 't_function_w_TTR', 'semantic_similarity_adjacent_sent',
                              'sent_cut', 't_genre'], axis=1)

            elif args.s0_rm1_lf0 is True:
                df = df.drop(['CURRENT_FIX_X', 'CURRENT_FIX_Y', 'CURRENT_FIX_PUPIL', 'CURRENT_FIX_DURATION',
                              'simplified_pos_A', 'simplified_pos_N', 'simplified_pos_FUNC', 'simplified_pos_VERB',
                              'content_word', 'synonym_homonym','NE_IOB', 'n_rights', 'n_lefts', 'dep_distance',
                              'synt_surprisal', 'surprisal', 'wordfreq_lemma', 'tf_idf', 'lex_overlap',
                              'pron_det_to_noun_ratio', 'voice', 'word_n_char', 'sent_n_words', 't_n_phrases',
                              'sent_mean_word_length', 'sent_lexical_density_tokens', 't_lexical_density', 't_lemma_TTR',
                              't_content_w_TTR', 't_function_w_TTR', 'semantic_similarity_adjacent_sent', 'sent_cut',
                              't_genre'], axis=1)

            elif args.s0_rm0_lf1 is True:
                df = df.drop(['CURRENT_FIX_X', 'CURRENT_FIX_Y', 'CURRENT_FIX_PUPIL', 'CURRENT_FIX_DURATION', 'ffd',
                              'tfd', 'n_fix', 'fpr', 'incoming_sac_regressive_norm', 'outgoing_sac_regressive_norm',
                              'outgoing_sac_progressive_norm'], axis=1)

        feature_names = df.columns[6:]

        return df, feature_names

    # encode data with varying input type
    def one_hot_encode(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """One Hot Encoding nominal features: Grab a column with a nominal feature and return ohe columns
        Features: simplified_pos, NE_label, deps. The feature values may differ between the datasets.
        """
        for feature in features:
            # get label names
            unique_tags = df[feature].unique()
            for tag in unique_tags:
                df[str(feature) + "_" + str(tag)] = (df[feature] == tag).astype(int)

            # df.drop(feature, axis=1, inplace=True)

        return df

    def encode_binary_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Binary features: 'voice', 't_genre'
        # the other binary features are already encoded as (0/1): fpr, 'content_word', 'synonym_homonym',
        # 'lex_overlap', 'sent_cut'
        """
        for feature in features:
            unique_vals = df[feature].unique()
            if len(unique_vals) == 2:
                # If feature is binary, map one value to 1 and the other to 0
                mapping = {unique_vals[0]: 1, unique_vals[1]: 0}
                df[feature] = df[feature].map(mapping)
            else:
                # If feature is nominal, create one-hot encoded columns
                for val in unique_vals:
                    df[val] = (df[feature] == val).astype(int)
                df.drop(feature, axis=1, inplace=True)

        return df

    def extract_features(self) -> Tuple[np.ndarray]:
        """Extract features from csv and transform to np arrays.
        Return an array for features, size (#samples, #max fixations, #features) and
        Return an array for labels, size (#samples, 1)
        Indico: features (3280, 581, 37); labels (3280, 1)
        SBSAR: features (2090, 394, 37); labels (2090, 1)
        Function originally from David, adapted by me.
        The function is taken from this script:
        https://github.com/aeye-lab/etra-reading-comprehension/blob/master/utils/generate_text_sequence_splits.py
        """
        subjects = list(self.data.subject_id.unique()) # len 95
        text_screens = list(self.data.text_screen_id.unique())  # len 22
        max_num_fixations = self.data.groupby(["subject_id", "text_id", "screen_id"]).size().max()  # for padding: 394 or 581
        num_features = self.data.columns

        # initialize empty arrays as containers
        data_arr = np.empty((0, max_num_fixations, len(self.features)))  # [], indico shape (0, 581, 39); sbsat (0, 394, 39)
        label_arr = np.empty((0, 3))  # shape (0, 1) -> 1 label in my dataset columns in labels df (questionnnaire) plus subj and book

        texts_read = {}

        # get a dictionary with possible combinations (who has read which texts?)
        # keys: text_screen_id, value: list of subjects who read that page
        for text_screen_id in text_screens:
            # texts_read.append(text_screen_id)
            texts_read[text_screen_id] = []
            for subject in subjects:
                texts_read[text_screen_id].append(subject)

        # sort dictionary by key value
        texts_read = dict(sorted(texts_read.items()))

        # iterate over each text page
        for text_screen_id in texts_read:
            subjects_who_read_text = texts_read[text_screen_id]
            print(f'Calculating for {text_screen_id=}....')

            text_id = int(text_screen_id[:-2])
            screen_id = int(text_screen_id[-1])

            # iterate over subjects who have read the text and get an array for data and label
            for subject in tqdm(subjects_who_read_text):

                try:
                    tmp_label = self.labels[(int(subject), int(text_id))]  # subj_id, text_id, label (score)


                except KeyError:
                    pass

                fix_subdf = self.data.loc[(self.data["subject_id"] == subject) & (self.data["text_id"] == text_id) & (
                        self.data["screen_id"] == screen_id), self.features]


                # transform features df into np array, one row per feature
                features = fix_subdf.T.values  # shape (n featues, n fixations), e.g. (43, 108)
                # transpose features into shape (n fixations, n features), e.g. (108, 43), one row per fixation
                features = features.transpose()
                # features = features.reshape((1, features.shape[0], features.shape[1])) # (1, num feat, num fix)
                features = torch.from_numpy(features).unsqueeze(0)
                tmp_len = features.shape[1]  # num fixations in this scanpath (page), e.g. 108 or 61 fixations (=rows)

                # stack array and pad to max number of fixations on any page in this dataset
                data_arr = np.vstack(
                    [
                        data_arr,
                        np.pad(
                            features,
                            pad_width=((0, 0), (0, max_num_fixations - tmp_len), (0, 0)),
                        ),
                    ],
                )

                label_arr = np.vstack([label_arr, tmp_label])  # shape (1, 1), e.g. [6.]
        # with open(f'nn/arrays_lstm/label_arr.npy','wb') as f:
        #     np.save(f, label_arr)
        # with open(f'nn/arrays_lstm/data_arr.npy','wb') as f:
        #     np.save(f, data_arr)

        return label_arr, data_arr

    def write_npys(self,
            label_arr: np.array,
            data_arr: np.array,
            split_criterion: str,
            save_path: str = '',
            ) -> int:
        """This function is taken from this script:
        https://github.com/aeye-lab/etra-reading-comprehension/blob/master/utils/generate_text_sequence_splits.py"""
        if split_criterion != 'book':
            n_folds = 5
        else:
            n_folds = 4

        outer_cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(data_arr)):

            random.seed(fold)
            np.random.seed(fold)

            print(f'Writing fold {fold}...')

            X_train, X_test = data_arr[train_idx], data_arr[test_idx]
            y_train, y_test = label_arr[train_idx], label_arr[test_idx]

            with open(f'{save_path}/{split_criterion}/X_train_{split_criterion}_{fold}.npy', 'wb') as f:  # noqa: E501
                            np.save(f, X_train)
            with open(f'{save_path}/{split_criterion}/X_test_{split_criterion}_{fold}.npy', 'wb') as f:  # noqa: E501
                            np.save(f, X_test)
            with open(f'{save_path}/{split_criterion}/y_train_{split_criterion}_{fold}.npy', 'wb') as f:  # noqa: E501
                            np.save(f, y_train)
            with open(f'{save_path}/{split_criterion}/y_test_{split_criterion}_{fold}.npy', 'wb') as f:  # noqa: E501
                            np.save(f, y_test)

        return 0


def main():
    if args.InDiCo is True:
        path = "data/InDiCo/processed/indico_fix_lexfeats_final.csv"
        label_path = "data/InDiCo/interim/labels/indico_labels.csv"
        label_dictionary_path = "nn/indico_splits/labels_dict.json"
        # RQ 1: all linguistic features
        if args.s1_rm1_lf1 is True:
            save_path = 'nn/indico_splits/data_s1_rm1_lf1'
        # RQ 1: without linguistic features
        elif args.s1_rm1_lf0 is True:
            save_path = 'nn/indico_splits/data_s1_rm1_lf0'
        # RQ 2: only pos and content word
        elif args.s1_rm1_lf1_pos_cont is True:
            save_path = 'nn/indico_splits/data_s1_rm1_lf1_pos_cont'
        # Ablation study
        # only scanpath
        elif args.s1_rm0_lf0 is True:
            save_path = 'nn/indico_splits/data_s1_rm0_lf0'
        # only reading measures
        elif args.s0_rm1_lf0 is True:
            save_path = 'nn/indico_splits/data_s0_rm1_lf0'
        # only linguistic features
        elif args.s0_rm0_lf1 is True:
            save_path = 'nn/indico_splits/data_s0_rm0_lf1'


    elif args.SBSAT is True:
        path = "data/SB-SAT/processed/sbsat_fix_lexfeats_final.csv"
        label_path ="data/SB-SAT/interim/labels/sbsat_labels.csv"
        label_dictionary_path = "nn/sbsat_splits/labels_dict.json"
        # RQ 1: all linguistic features
        if args.s1_rm1_lf1 is True:
            save_path = 'nn/sbsat_splits/data_s1_rm1_lf1'
        # RQ 1: without linguistic features
        elif args.s1_rm1_lf0 is True:
            save_path = 'nn/sbsat_splits/data_s1_rm1_lf0'
        # RQ 2: only pos and content word
        elif args.s1_rm1_lf1_pos_cont is True:
            save_path = 'nn/sbsat_splits/data_s1_rm1_lf1_pos_cont'
        # Ablation study
        # only scanpath
        elif args.s1_rm0_lf0 is True:
            save_path = 'nn/sbsat_splits/data_s1_rm0_lf0'
        # only reading measures
        elif args.s0_rm1_lf0 is True:
            save_path = 'nn/sbsat_splits/data_s0_rm1_lf0'
        # only linguistic features
        elif args.s0_rm0_lf1 is True:
            save_path = 'nn/sbsat_splits/data_s0_rm0_lf1'

    # -----------------------------
    E = EncodedData(path, label_path)
    # print(E.data['simplified_pos'].unique())
    # print(E.data['NE_IOB'].unique())

    # write labels dictionary to json
    with open(label_dictionary_path, "w") as outfile:
        json.dump(E.label_dict, outfile)
    #
    # # print(E.labels)
    # # print(E.label_dict)
    # # # print(E.load_data(path)[0].columns)
    # # print(E.extract_features())
    # #
    label_arr, data_arr = E.extract_features()
    # print(np.isnan(data_arr).any())
    # print(data_arr)
    # print(E.label_dict)
    # print("label_arr", label_arr)
    # print("data_arr", data_arr)
    # print("label_arr shape", label_arr.shape)
    # print("data_arr shape", data_arr.shape)

    # save_path = 'nn/data_with_features'
    os.makedirs(save_path, exist_ok=True)
    for split_criterion in ['subj', 'book', 'book-page']:
        os.makedirs(os.path.join(save_path, split_criterion), exist_ok=True)
        print(f'Creating files for split {split_criterion}...')
        E.write_npys(
            label_arr=label_arr,
            data_arr=data_arr,
            split_criterion=split_criterion,
            save_path=save_path,
        )
    return 0

if __name__ == "__main__":
    main()