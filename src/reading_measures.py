import pandas as pd
from typing import Dict, Collection, List, Tuple
import glob
import sys
import string
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    prog='preprocess_indico_fixations',
    description='Preprocess raw fixations from indico (Dataviewer output): Clean & map interest areas')

parser.add_argument('--InDiCo', action='store_true', help='Process the txt file containing all sessions')
parser.add_argument('--SBSAT', action='store_true', help='Process the txt files containing individual '
                                                                       'sessions')
args = parser.parse_args()


class FixationDataset:
    def __init__(
        self,
        type,
    ):
        self.correction = type
        self.readerIds: Collection[int] = []
        self.readerTextDict: Dict = {}
        if type == "indico":
            self.lexical_features_file = "data/InDiCo/interim/stimuli/annotated_texts/InDiCo_annotated.csv"
            self.fixations_file = "data/InDiCo/interim/fixation/indico_fix_preprocessed.csv"
            self.data: pd.DataFrame = self.read_fixations()
            self.problems = [(1, 2), (2, 3), (4, 4), (16, 3), (16, 4)]  # (text, screen)
        elif type == "sbsat":
            self.lexical_features_file = "data/SB-SAT/interim/stimuli/annotated_texts/SBSAT_annotated.csv"
            self.fixations_file = "data/SB-SAT/interim/fixation/sbsat_fix_preprocessed.csv"  # apostrophe is stripped
            self.data: pd.DataFrame = self.read_fixations()
            self.problems = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 1), (2, 2), (2, 3), (2, 5), (2, 6), (3, 1),
                             (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (4, 1), (4, 2), (4, 3)]


        self.lexical_featues_list = ['text_number', 'screen_number', 'sentence_number', 'word_in_screen_id', 'word',
                                     'lemma', 'word_n_char', 'pos', 'simplified_pos', 'wordfreq_lemma', 'NE_IOB',
                                     'NE_label', 'content_word', 'technical_term', 'synonym_homonym', 'synt_surprisal', 'surprisal_bert', 'deps',
                                     'rights', 'lefts', 'n_rights', 'n_lefts', 'dep_distance', 'sent_n_words',
                                     'sent_n_char', 'sent_mean_word_length', 'sent_lexical_density_tokens', 'sent_cut',
                                     't_n_char', 't_n_words', 't_n_phrases', 't_mean_word_length', 't_lexical_density',
                                     't_lemma_TTR', 't_content_w_TTR', 't_function_w_TTR', 't_nouns_TTR',
                                     't_verbs_TTR', 't_adj_TTR', 't_adv_TTR', 't_genre', 'tf', 'idf', 'tf_idf',
                                     'lex_overlap', 'semantic_similarity_adjacent_sent', 'pron_det_to_noun_ratio',
                                     'voice']


        self.lexFeat: pd.DataFrame = pd.read_csv(
            self.lexical_features_file, header="infer", encoding="utf8", usecols=self.lexical_featues_list
        )

        self.lexFeat = self.lexFeat.rename(columns={"word_in_screen_id": "w_in_screen_id"})
        # self.lexFeat = self.lexFeat.rename(columns={"text_number": "text_id",
        #                                             "screen_number": "screen_id"})

        self.lexFeat.index = [
            self.lexFeat.text_number, # rename to id
            self.lexFeat.screen_number,
            self.lexFeat.w_in_screen_id,
        ]

        # dep_file = open("../data/test_daf_dependencies.json")
        # self.deps = json.load(dep_file)

    def read_fixations(self) -> pd.DataFrame:
        """Read in preprocessed fixation report, calculate reading measures, export df with reading measures"""
        # 003, 005 und 014 haben Deutsch nicht als Muttersprache
        fixations_df: pd.DataFrame = pd.read_csv(
            self.fixations_file,
            header="infer",
            decimal=",",
            sep=",",
            encoding="utf8",
            # dtype={'PARTICIPANT_AGE': str}  # specified to avoid DtypeWarning (the column will be droped anyway)
            # TOTO: delete
        )

        # add fixation id: number fixations within a text
        fixations_df['fixation_id'] = fixations_df.groupby('text_id')['screen_id'].cumcount() + 1


        # Calculate reading measures
        grouped_fixations = fixations_df.groupby(['subject_id', 'session_id', 'text_id', 'screen_id',
                                                  'word_in_screen_id'], group_keys=True)
        grouped_fixations1 = fixations_df.groupby(['subject_id', 'session_id', 'text_id', 'screen_id'], group_keys=True)
        grouped_fixations2 = fixations_df.groupby(['subject_id', 'session_id', 'text_id'], group_keys=True)

        # calculate ffd and tfd, n_fix
        first_fixation_duration = grouped_fixations['CURRENT_FIX_DURATION'].first()
        total_fixation_duration = grouped_fixations['CURRENT_FIX_DURATION'].sum()
        number_fixations_on_word = grouped_fixations['CURRENT_FIX_INTEREST_AREA_RUN_ID'].max()
        # Map the results for ffd, tfd, n_fix to the original DataFrame
        fixations_df['ffd'] = fixations_df.apply(
            lambda x: first_fixation_duration[x['subject_id'], x['session_id'], x['text_id'], x['screen_id'], x[
                'word_in_screen_id']], axis=1)
        fixations_df['tfd'] = fixations_df.apply(
            lambda x: total_fixation_duration[x['subject_id'], x['session_id'], x['text_id'], x['screen_id'], x[
                'word_in_screen_id']], axis=1)
        fixations_df['n_fix'] = fixations_df.apply(
            lambda x: number_fixations_on_word[x['subject_id'], x['session_id'], x['text_id'], x['screen_id'], x[
                'word_in_screen_id']], axis=1)

        # # ----------------
        # calculate normalized incoming regression count, normalized outgoing regression/progression, fpr
        # 1) calculate for every wort whether incoming/outgoing regression/progression
        # outgoing progressive sac: word id in current line < word id in line below
        grouped_fixations2 = fixations_df.groupby(['subject_id', 'session_id', 'text_id'], group_keys=True)
        shifted_word_id_down = grouped_fixations1['word_in_screen_id'].shift(periods=-1) # word id of the row below current id
        diff_word_id = fixations_df['word_in_screen_id'] - shifted_word_id_down
        fixations_df['outgoing_sac_progressive'] = diff_word_id.apply(lambda x: 1 if x < 0 else 0)
        # Fill the `NaN` values with `0` in the first row of each group
        fixations_df['outgoing_sac_progressive'] = (fixations_df['outgoing_sac_progressive'].fillna(0))
        # Make sure the values in the 'outgoing_sac_progressive' column are integers
        fixations_df['outgoing_sac_progressive'] = fixations_df['outgoing_sac_progressive'].astype(int)

        # outgoing regressive sac: word id in current line > word id in line below
        fixations_df['outgoing_sac_regressive'] = diff_word_id.apply(lambda x: 1 if x > 0 else 0)
        fixations_df['outgoing_sac_regressive'] = (fixations_df['outgoing_sac_regressive'].fillna(0))
        fixations_df['outgoing_sac_regressive'] = fixations_df['outgoing_sac_regressive'].astype(int)

        # incoming regressive sac: word id in current line < word id in line above
        shifted_word_id_up = grouped_fixations1['word_in_screen_id'].shift(periods=1)  # word id of the row above current row
        diff_word_id_upwards = fixations_df['word_in_screen_id'] - shifted_word_id_up
        fixations_df['incoming_sac_regressive'] = diff_word_id_upwards.apply(lambda x: 1 if x < 0 else 0)
        # fixations_df['incoming_sac_regressive'] = diff_word_id_upwards.apply(lambda x: 1 if x < 0 else 0 if x != 0 else 0)
        fixations_df['incoming_sac_regressive'] = (fixations_df['incoming_sac_regressive'].fillna(0))
        fixations_df['incoming_sac_regressive'] = fixations_df['incoming_sac_regressive'].astype(int)

        # fpr: if the outgoing saccade is regressive and the RUN ID is 1
        if fixations_df['CURRENT_FIX_INTEREST_AREA_RUN_ID'].dtype != 'int64':
            fixations_df['CURRENT_FIX_INTEREST_AREA_RUN_ID'] = fixations_df['CURRENT_FIX_INTEREST_AREA_RUN_ID'].astype(
                float).astype(int)
        fixations_df['fpr'] = ((fixations_df['outgoing_sac_regressive'] == 1) & (
                fixations_df['CURRENT_FIX_INTEREST_AREA_RUN_ID'] == 1)).astype(int)

        # normalized outgoing progressive/regressive saccades
        # sum up outgoing progr/regr saccades on individual word
        outgoing_progressive_sac_on_word = grouped_fixations['outgoing_sac_progressive'].sum()
        outgoing_regressive_sac_on_word = grouped_fixations['outgoing_sac_regressive'].sum()
        incoming_regressive_sac_on_word = grouped_fixations['incoming_sac_regressive'].sum()
        fixations_df['outgoing_progressive_sac_on_word'] = fixations_df.apply(
            lambda x: outgoing_progressive_sac_on_word[x['subject_id'], x['session_id'], x['text_id'], x['screen_id'], x[
                'word_in_screen_id']], axis=1)
        fixations_df['outgoing_regressive_sac_on_word'] = fixations_df.apply(
            lambda x: outgoing_regressive_sac_on_word[
                x['subject_id'], x['session_id'], x['text_id'], x['screen_id'], x[
                    'word_in_screen_id']], axis=1)
        fixations_df['incoming_regressive_sac_on_word'] = fixations_df.apply(
            lambda x: incoming_regressive_sac_on_word[
                x['subject_id'], x['session_id'], x['text_id'], x['screen_id'], x[
                    'word_in_screen_id']], axis=1)

        # sum of regressive saccades in scanpath
        regressions_total = grouped_fixations2['outgoing_sac_regressive'].sum()
        fixations_df['outgoing_regressive_sac_scanpath'] = fixations_df.apply(
            lambda x: regressions_total[x['subject_id'], x['session_id'], x['text_id']], axis=1)
        # sum of progressive saccades in scanpath
        progressions_total = grouped_fixations2['outgoing_sac_progressive'].sum()
        fixations_df['outgoing_progressive_sac_scanpath'] = fixations_df.apply(
            lambda x: progressions_total[x['subject_id'], x['session_id'], x['text_id']], axis=1)

        # Normalize scores on word level
        fixations_df["outgoing_sac_regressive_norm"] = round((fixations_df['outgoing_regressive_sac_on_word'] /
                                                              fixations_df['outgoing_regressive_sac_scanpath']), 6)
        fixations_df["outgoing_sac_progressive_norm"] = round((fixations_df['outgoing_progressive_sac_on_word'] /
                                                               fixations_df['outgoing_progressive_sac_scanpath']), 6)
        fixations_df["incoming_sac_regressive_norm"] = round((fixations_df['incoming_regressive_sac_on_word'] /
                                                              fixations_df['outgoing_regressive_sac_scanpath']), 6)

        fixations_df["word"] = fixations_df["word"].replace("â€ž", "„", regex=True)
        fixations_df["word"] = fixations_df["word"].replace("â€œ", "“", regex=True)
        fixations_df["word"] = fixations_df["word"].replace("â€�", "”", regex=True)

        # dtype conversion for sbsat data
        if fixations_df['n_fix'].dtype != 'int64':
            fixations_df['n_fix'] = fixations_df['n_fix'].astype(float).astype(int)

        fixations_df.sort_values(
            by=["subject_id", "text_id", "screen_id", "fixation_id"], inplace=True
        )

        fixations_df.reset_index(drop=True, inplace=True)

        self.readerIds = fixations_df.subject_id.unique()
        self.readerTextDict = dict.fromkeys(self.readerIds)

        tIds = fixations_df.groupby("subject_id")["text_id"].unique()

        for i in range(0, len(tIds)):
            self.readerTextDict[tIds.index[i]] = tIds[tIds.index[i]]

        if args.SBSAT is True:
            fixations_df["word"] = fixations_df["word"].replace("’", "'")
            fixations_df["word"] = fixations_df["word"].apply(lambda x: x.rstrip())


        return fixations_df[['subject_id', 'fixation_id', 'session_id', 'text_id', 'screen_id', 'CURRENT_FIX_X',
                             'CURRENT_FIX_Y', 'CURRENT_FIX_PUPIL', 'CURRENT_FIX_DURATION', 'ffd', 'tfd', 'n_fix',
                             'word_in_screen_id', 'fpr', 'incoming_sac_regressive_norm',
                             'outgoing_sac_regressive_norm', 'outgoing_sac_progressive_norm',
                             'word', 'CURRENT_FIX_INTEREST_AREA_RUN_ID']]

    def clean_optimized(self, type: str):
        """Optimized clean function for the ET data only.
        If a word falsely got split into two words (problems 1&2), the lines get
        merged and the wordIDs after the word get updated. If a word got falsely
        merged (problems 3,4&5), the line gets dropped and the wordIDs after the word
        get updated.
        In SB-SAT, there are no alignment issues, just encoding errors (apostrophe)"""

        if type == "indico":
            # problems: lexical features vs. reading measures
            # problem 1: lexical features word Eiszeiten (86) vs. reading measures word Eis
            self.clean_merge_lines(1, 2, 86, 87, "Eiszeiten")
            # problem 2: lexical features word Mutter-Kind-Situation (82) vs. reading measures word Mutter-
            self.clean_merge_lines(2, 3, 82, 83, "Mutter-Kind-Situation.")
            # # problem 3: lexical features word zu (79) vs. reading measures word zukanalisieren
            self.clean_remove_line(4, 4, 79, 80, "zu", 1)
            # # problem 4: lexical features word hatten (85) vs. reading measures word hatten.Kleine
            self.clean_remove_line(16, 3, 85, 86, "hatten", 1)
            # # problem 5: lexical features word Pflanzenwelt (6) vs. reading measures word Pflanzenwelt.Es
            self.clean_remove_line(16, 4, 6, 7, "Pflanzenwelt", 1)
            # # problem 6: lexical features word VHB (2) vs. reading measures word www.vhb.org
            self.clean_rename_line(9, 4, 2, 3, "VHB")

            self.data.reset_index(drop=True, inplace=True)

    def clean_optimized_sbsat(self):

        print("My name is sunny")
        print("Are we here nooow")
        # text 1, screen 1, wordid 95: lexical features word didn’t vs. reading measures word didn t
        self.clean_rename_line_sbsat(1, 1, 95, 96, "didn’t")

        self.data.reset_index(drop=True, inplace=True)

            # self.data.loc[(self.data['text_id'] == 1) & (self.data['screen_id'] == 1) & (self.data['word_in_screen_id'] == 95), 'word'] = "didn’t"

    def clean_merge_lines(self, text, screen, line1, line2, merged_word):
        """Function to fix ET problem 1 (Eiszeiten) and problem 2 (Mutter-Kind-
        Situation) where the words  get merged & the IDs after the merged
        word get updated.
        note: in the comments i'm using l1 and l2 because i'm referring to the lines in the dataframe
        l1 is actually the first part of the split word (Eis or Mutter) and l2 is the second part of the split word
        (-zeiten or -Kind-Situation). If there's two fixations on Eis, i'll call it l1[1] and l1[2]
        """
        # get problematic lines:
        problem_subjs = self.get_problematic_fixations(text, screen, line1, line2)
        # print('problem_subjs: ', problem_subjs)

        line3 = line2 + 1
        # print("Line1 ", line1)
        # print("Line2 ", line2)
        # print("Line3 ", line3)
        # 1. l1 fixated, l2 not fixated (Eis-):
        # adjust the word in l1, otherwise leave l1 alone. Move idx of all lines after l2 down by 1
        for subj in problem_subjs[0]:
            # print("subj problem 1: ", subj)
            # get the first line (which got fixated)
            line_to_merge = self.get_line_by_wordid(subj, text, screen, line1)
            ids_merge = line_to_merge.index

            for id_merge in ids_merge:  # loop in case of multiple fixations by the same subject

                # merge the word
                self.data.at[id_merge, "word"] = merged_word
                # fix word IDs of lines that come after the merged line
                lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line3)
                id_fix_wordids = lines_to_fix_wordid.index
                for id_fix_wordid in id_fix_wordids: #in case of multiple lines
                    self.data.at[id_fix_wordid, "word_in_screen_id"] = (
                            self.data.loc[id_fix_wordid, "word_in_screen_id"] - 1)

        # 2. l1 not fixated, l2 fixated (-zeiten):
        # change all ids from l2 onwards
        for subj in problem_subjs[1]:
            # print("subj. problem 2: ",subj)
            # get the second line (which got fixated) -> here, the word & ID needs to be adjusted
            line_to_adjust_word = self.get_line_by_wordid(subj, text, screen, line2)
            ids_adjust_word = line_to_adjust_word.index.tolist()

            for id_adjust_word in ids_adjust_word:
                self.data.at[id_adjust_word, "word"] = merged_word
                self.data.at[id_adjust_word, "word_in_screen_id"] = (
                        self.data.loc[id_adjust_word, "word_in_screen_id"] - 1)

            # get the lines to fix word id: move ids down for all of the following words
            lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line3)
            ids_fix_wordid = lines_to_fix_wordid.index

            for id_fix_wordid in ids_fix_wordid:  # loop in case of multiple fixations by the same subject
                # only one fixation on l2: only change l2
                self.data.at[id_fix_wordid, "word_in_screen_id"] = (
                        self.data.loc[id_fix_wordid, "word_in_screen_id"] - 1)



            # # print("problem_subjs", subj)
            # # get the line that got fixated (l2)
            # lines_to_drop = self.get_line_by_wordid(subj, text, screen, line2)
            # ids_drop = lines_to_drop.index
            # # merge the word
            #
            # for id_drop in ids_drop: # in case of multiple fixations
            #     self.data.at[id_drop, "word"] = merged_word
            #     # update word IDs after the merged word
            #     if text == 1 and screen == 2 and line1 == 86 and line2 == 87:  # eiszeiten
            #         lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line3)
            #     elif text == 2 and screen == 3 and line1 == 82 and line2 == 83:  # mutter-kind
            #         lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line2)
            #     ids_fix_wordid = lines_to_fix_wordid.index
            #     for id_fix_wordid in ids_fix_wordid:
            #         self.data.at[id_fix_wordid, "word_in_screen_id"] = (
            #                 self.data.loc[id_fix_wordid, "word_in_screen_id"] - 1)

        # ----------------------------------------------------------------------------------¨
        # ----------------------------------------------------------------------------------
        # --------------------------CASE 3I - OLD BEGIN ------------------------------
        # ------------------------orignially wordsw were merged--------------------------------------------
        # ----------------------------------------------------------------------------------

#         # 3. l1 fixated, l2 fixated:
#         for subj in problem_subjs[2]:
#             # print("subj problem 3: ", subj)
#             lines_to_merge = self.get_line_by_wordid(subj, text, screen, line1)  # can be one or multiple lines
#             lines_to_drop = self.get_line_by_wordid(subj, text, screen, line2)  # can be one or multiple lines
#
#             # print("l to merge", lines_to_merge)
#             # print("l to drop", lines_to_drop)
#
#
#             # 3i) l1 fixated once, l2 fixated twice. There are 2 possibilities:
#             # 3ia) l2[1] is subsequent to l1; l2[2] is a standalone fixation -> treat l2[2] like l1 not fixated, l2 fixated
#             # 3ib) l2[2] is subsequent to l1; l2[1] is a standalone fixation -> treat l2[1] like l1 not fixated, l2 fixated
#
#             if len(lines_to_merge) == 1 and len(lines_to_drop) == 2:
#
#                 lines_to_drop1 = lines_to_drop.iloc[[0]]
#                 lines_to_drop2 = lines_to_drop.iloc[[1]]
#                 id_merge = lines_to_merge.index[0].item()
#
#                 # ia) l2[1] subsequent to l1 -> merge
#                 if lines_to_merge.index == lines_to_drop1.index -1:  # subsequent fixations
#                     # print("First line is part of merged") # we didn't have that yet
#                     id_merge = lines_to_merge.index[0].item()
#                     id_drop = lines_to_drop1.index[0].item()
#
#                     self.data.at[id_merge, "ffd"] = (int(line_to_merge.loc[:, "ffd"].values[0])
#                                                      + int(lines_to_drop1.loc[:, "ffd"].values[0]))
#                     self.data.at[id_merge, "tfd"] = (int(line_to_merge.loc[:, "tfd"].values[0])
#                                                      + int(lines_to_drop1.loc[:, "tfd"].values[0]))
#                     self.data.at[id_merge, "n_fix"] = (int(line_to_merge.loc[:, "n_fix"].values[0])
#                                                        + int(lines_to_drop1.loc[:, "n_fix"].values[0]))
#                    # self.data.at[id_merge, "n_regressions_word_norm"] = (int(
#                     #    line_to_merge.loc[:, "n_regressions_word_norm"].values[0])
#                     #                                                     + int(
#                     #            lines_to_drop1.loc[:, "n_regressions_word_norm"].values[0]))
#
#                     self.data.at[id_merge, "fpr"] = int(lines_to_drop1.loc[:, "fpr"].values[0])
#                     self.data.loc[id_merge, "word"] = merged_word
#
#                     lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line3)
#
#                     id_fix_wordid = lines_to_fix_wordid.index
#
#                     for id in id_fix_wordid:
#                         self.data.loc[id, "word_in_screen_id"] = (self.data.loc[id, "word_in_screen_id"]) - 1
#                     self.data.drop(id_drop, axis=0, inplace=True)
#
#
#
#                 # 1b) Treat l2[1] as standalone fixation
#                 else:
#                     # this fixation is not part of another one -> treat it as if only the second part of the word had been fixated (see above)
#                     idx_to_adjust = lines_to_drop1.index[0].item()
#                     # merge the word
#                     self.data.at[idx_to_adjust, "word"] = merged_word
#                     # fix word IDs of lines that come after the merged line
#                     lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line3)
#                     # fix word id
#                     self.data.at[idx_to_adjust, "word_in_screen_id"] = (self.data.loc[idx_to_adjust, "word_in_screen_id"] - 1)
#
#                 # ib) l2[2] subsequent to l1 -> merge
#                 if lines_to_merge.index == lines_to_drop2.index -1:  # subsequent fixations, e.g. Eiszeiten, subject 4
#                     id_merge = lines_to_merge.index[0].item()
#                     id_drop = lines_to_drop2.index[0].item()
#
#                     self.data.at[id_merge, "ffd"] = (int(line_to_merge.loc[:, "ffd"].values[0])
#                                                      + int(lines_to_drop2.loc[:, "ffd"].values[0]))
#                     self.data.at[id_merge, "tfd"] = (int(line_to_merge.loc[:, "tfd"].values[0])
#                                                     + int(lines_to_drop2.loc[:, "tfd"].values[0]))
#                     self.data.at[id_merge, "n_fix"] = (int(line_to_merge.loc[:, "n_fix"].values[0])
#                                                      + int(lines_to_drop2.loc[:, "n_fix"].values[0]))
#                    # self.data.at[id_merge, "n_regressions_word_norm"] = (int(
#                     #    line_to_merge.loc[:, "n_regressions_word_norm"].values[0])
#                     #                                 + int(lines_to_drop2.loc[:, "n_regressions_word_norm"].values[0]))
#
#                     self.data.at[id_merge, "fpr"] = int(lines_to_drop2.loc[:, "fpr"].values[0])
#                     self.data.loc[id_merge, "word"] = merged_word
#
#                     lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line3)
#
#                     id_fix_wordid = lines_to_fix_wordid.index
#
#                     for id in id_fix_wordid:
#                         self.data.loc[id, "word_in_screen_id"] = (self.data.loc[id, "word_in_screen_id"])-1
#                     self.data.drop(id_drop, axis=0, inplace=True)
#
#
#                 # ia) Treat l2[2] as standalone fixation
#                 else:
#                     # this fixation is not part of another one -> treat it as if only the second part of the word had been fixated (see above)
#                     idx_to_adjust = lines_to_drop2.index[0].item()
#                     # merge the word
#                     self.data.at[idx_to_adjust, "word"] = merged_word
#                     # fix word IDs of lines that come after the merged line
#                     lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line3)
#                     # fix word id
#                     self.data.at[idx_to_adjust, "word_in_screen_id"] = (
#                                 self.data.loc[idx_to_adjust, "word_in_screen_id"] - 1)
#
#             # ----------------------
#             # 3ii) l1 more than once , l2 fixated once, all 3 fixations subsequent -> merge l2 into l1[2], leave l1[1]
#             elif len(lines_to_merge) >= 2 and len(lines_to_drop) == 1:
#
#                 line_to_merge1 = lines_to_merge.iloc[[0]]  # first fixation on first part of word
#                 id_merge1 = line_to_merge1.index[0].item()
#                 line_to_merge2 = lines_to_merge.iloc[[1]]  # second fixation on first part of word
#                 id_merge2 = line_to_merge2.index[0].item()
#                 id_drop = lines_to_drop.index[0].item()
#
#                 self.data.at[id_merge2, "ffd"] = (int(line_to_merge2.loc[:, "ffd"].values[0])
#                                                  + int(lines_to_drop.loc[:, "ffd"].values[0]))
#                 self.data.at[id_merge2, "tfd"] = (int(line_to_merge2.loc[:, "tfd"].values[0])
#                                                  + int(lines_to_drop.loc[:, "tfd"].values[0]))
#                 self.data.at[id_merge2, "n_fix"] = (int(line_to_merge2.loc[:, "n_fix"].values[0])
#                                                    + int(lines_to_drop.loc[:, "n_fix"].values[0]))
#                # self.data.at[id_merge2, "n_regressions_word_norm"] = (int(
#                 #    line_to_merge2.loc[:, "n_regressions_word_norm"].values[0])
#                 #                                                     + int(
#                   #          lines_to_drop.loc[:, "n_regressions_word_norm"].values[0]))
# #
#                 self.data.at[id_merge2, "fpr"] = int(lines_to_drop.loc[:, "fpr"].values[0])
#                 self.data.loc[id_merge2, "word"] = merged_word
#
#                 lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line3)
#
#                 id_fix_wordid = lines_to_fix_wordid.index
#
#                 for id in id_fix_wordid:
#                     self.data.loc[id, "word_in_screen_id"] = (self.data.loc[id, "word_in_screen_id"]) - 1
#                 self.data.drop(id_drop, axis=0, inplace=True)
#
#                 # treat l1[1] like case l1 fixated, l2 not fixated BUT don't move word id up!
#                 # merge the word
#                 self.data.at[id_merge1, "word"] = merged_word
#
#
#
#             # -----------------------------------------
#             # 3iii) l1 fixated twice, l2 fixated twice.
#             # l1[1] & l2[1] and l1[2] & l2[2] are subsequent fixations.
#             # -> merge l2[1] into l1[1] and merge l2[2] into l1[1]
#             elif len(lines_to_merge) == 2 and len(lines_to_drop) == 2:
#                 line_to_merge1 = lines_to_merge.iloc[[0]]  # l1[1]
#                 line_to_merge2 = lines_to_merge.iloc[[1]]  # l1[2]
#                 line_to_drop1 = lines_to_drop.iloc[[0]]  # l2[1]
#                 line_to_drop2 = lines_to_drop.iloc[[1]]  # l2[2]
#                 id_merge1 = line_to_merge1.index[0].item()
#                 id_merge2 = line_to_merge2.index[0].item()
#                 id_drop1 = line_to_drop1.index[0].item()
#                 id_drop2 = line_to_drop2.index[0].item()
#
#
#                 # first pair: merge l2[1] into l1[1]
#
#                 self.data.at[id_merge1, "ffd"] = (int(line_to_merge1.loc[:, "ffd"].values[0])
#                                                  + int(line_to_drop1.loc[:, "ffd"].values[0]))
#                 self.data.at[id_merge1, "tfd"] = (int(line_to_merge1.loc[:, "tfd"].values[0])
#                                                  + int(line_to_drop1.loc[:, "tfd"].values[0]))
#                 self.data.at[id_merge1, "n_fix"] = (int(line_to_merge1.loc[:, "n_fix"].values[0])
#                                                    + int(line_to_drop1.loc[:, "n_fix"].values[0]))
#                 #self.data.at[id_merge1, "n_regressions_word_norm"] = (int(
#                 #    line_to_merge1.loc[:, "n_regressions_word_norm"].values[0])
#                  #                                                    + int(
#                  #           line_to_drop1.loc[:, "n_regressions_word_norm"].values[0]))
#
#                 self.data.at[id_merge1, "fpr"] = int(line_to_drop1.loc[:, "fpr"].values[0])
#                 self.data.loc[id_merge1, "word"] = merged_word
#
#                 # drop lines for first pair
#                 lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line3)
#                 id_fix_wordid = lines_to_fix_wordid.index
#
#                 for id in id_fix_wordid:
#                     self.data.loc[id, "word_in_screen_id"] = (self.data.loc[id, "word_in_screen_id"]) - 1
#                 self.data.drop(id_drop1, axis=0, inplace=True)
#
#                 # if they directly follow one another
#                 if id_merge2 == id_drop2 - 1:
#                     # second pair: merge l2[2] into l1[1]
#                     self.data.at[id_merge2, "ffd"] = (int(line_to_merge2.loc[:, "ffd"].values[0])
#                                                      + int(line_to_drop2.loc[:, "ffd"].values[0]))
#                     self.data.at[id_merge2, "tfd"] = (int(line_to_merge2.loc[:, "tfd"].values[0])
#                                                      + int(line_to_drop2.loc[:, "tfd"].values[0]))
#                     self.data.at[id_merge2, "n_fix"] = (int(line_to_merge2.loc[:, "n_fix"].values[0])
#                                                        + int(line_to_drop2.loc[:, "n_fix"].values[0]))
#                     #self.data.at[id_merge2, "n_regressions_word_norm"] = (int(
#                     #    line_to_merge2.loc[:, "n_regressions_word_norm"].values[0])
#                      #                                                    + int(
#                      #           line_to_drop2.loc[:, "n_regressions_word_norm"].values[0]))
#
#                     self.data.at[id_merge2, "fpr"] = int(line_to_drop2.loc[:, "fpr"].values[0])
#                     self.data.loc[id_merge2, "word"] = merged_word
#
#                     # drop lines for second pair
#                     for id in id_fix_wordid:
#                         self.data.loc[id, "word_in_screen_id"] = (self.data.loc[id, "word_in_screen_id"]) - 1
#                     self.data.drop(id_drop2, axis=0, inplace=True)
#
#             # -----------------------------------------
#             # 3iii) l1 fixated twice, l2 fixated twice.
#             # l1[1] & l2[1] and l1[2] & l2[2] are subsequent fixations.
#             # -> merge l2[1] into l1[1] and merge l2[2] into l1[1]
#             elif len(lines_to_merge) == 1 and len(lines_to_drop) == 1:
#                 line_to_merge1 = lines_to_merge.iloc[[0]]  # l1[1]
#                 line_to_drop1 = lines_to_drop.iloc[[0]]  # l2[1]
#                 id_merge1 = line_to_merge1.index[0].item()
#                 id_drop1 = line_to_drop1.index[0].item()
#
#                 # first pair: merge l2[1] into l1[1]
#                 self.data.at[id_merge1, "ffd"] = (int(line_to_merge1.loc[:, "ffd"].values[0])
#                                                   + int(line_to_drop1.loc[:, "ffd"].values[0]))
#                 self.data.at[id_merge1, "tfd"] = (int(line_to_merge1.loc[:, "tfd"].values[0])
#                                                   + int(line_to_drop1.loc[:, "tfd"].values[0]))
#                 self.data.at[id_merge1, "n_fix"] = (int(line_to_merge1.loc[:, "n_fix"].values[0])
#                                                     + int(line_to_drop1.loc[:, "n_fix"].values[0]))
#                 #self.data.at[id_merge1, "n_regressions_word_norm"] = (int(
#                   #  line_to_merge1.loc[:, "n_regressions_word_norm"].values[0])
#                    #                                                   + int(
#                     #        line_to_drop1.loc[:, "n_regressions_word_norm"].values[0]))
#
#                 self.data.at[id_merge1, "fpr"] = int(line_to_drop1.loc[:, "fpr"].values[0])
#                 self.data.loc[id_merge1, "word"] = merged_word
#
#                 # drop lines for first pair
#                 lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line3)
#                 id_fix_wordid = lines_to_fix_wordid.index
#
#                 for id in id_fix_wordid:
#                     self.data.loc[id, "word_in_screen_id"] = (self.data.loc[id, "word_in_screen_id"]) - 1
#                 self.data.drop(id_drop1, axis=0, inplace=True)

        # ----------------------------------------------------------------------------------¨
        # ----------------------------------------------------------------------------------
        # --------------------------CASE 3I - OLD ENDE ------------------------------
        # ----------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------¨
        # ----------------------------------------------------------------------------------
        # --------------------------CASE 3I - NEW START ------------------------------
        # ---------------------------------here i dont merge lines but count as 2 fix on same word-----------------
        # ----------------------------------------------------------------------------------
        # 3. l1 fixated, l2 fixated:
        for subj in problem_subjs[2]:
            # print("subj problem 3: ", subj)
            lines_to_merge = self.get_line_by_wordid(subj, text, screen, line1)  # can be one or multiple lines
            lines_to_drop = self.get_line_by_wordid(subj, text, screen, line2)  # can be one or multiple lines

            # print("l to merge", lines_to_merge)
            # print("l to drop", lines_to_drop)

            # 3i) l1 fixated once, l2 fixated twice. There are 2 possibilities:
            # 3ia) l2[1] is subsequent to l1; l2[2] is a standalone fixation -> treat l2[2] like l1 not fixated, l2 fixated
            # 3ib) l2[2] is subsequent to l1; l2[1] is a standalone fixation -> treat l2[1] like l1 not fixated, l2 fixated
            if len(lines_to_merge) == 1 and len(lines_to_drop) == 2:

                lines_to_drop1 = lines_to_drop.iloc[[0]]
                lines_to_drop2 = lines_to_drop.iloc[[1]]
                id_merge = lines_to_merge.index[0].item()

                # ia) l2[1] subsequent to l1 -> adapt reading measures, count as 2 fixations on the same word
                if lines_to_merge.index == lines_to_drop1.index - 1:  # subsequent fixations
                    # print("First line is part of merged") # we didn't have that yet
                    id_merge = lines_to_merge.index[0].item() # WORD 1
                    id_drop = lines_to_drop1.index[0].item() # WORD 2

                    self.data.loc[id_merge, "word"] = merged_word
                    self.data.loc[id_drop, "word"] = merged_word
                    # n fix -> add the ones from word 2 to word 1
                    self.data.loc[id_merge, "n_fix"] = self.data.loc[id_merge, "n_fix"] + self.data.loc[id_drop, "n_fix"]
                    self.data.loc[id_drop, "n_fix"] = self.data.loc[id_merge, "n_fix"] + self.data.loc[id_drop, "n_fix"]
                    # adjust ffd of word 2 with ffd of word 2
                    self.data.loc[id_drop, "ffd"] = self.data.loc[id_merge, "ffd"]
                    # outgoing progressive and regressive saccade:
                    self.data.loc[id_merge, "outgoing_progressive_sac_on_word"] = 0
                    self.data.loc[id_merge, "outgoing_regressive_sac_on_word"] = 0
                    # incoming regressive saccade
                    self.data.loc[id_drop, "incoming_regressive_sac_on_word"] = 0


                    lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line3)

                    id_fix_wordid = lines_to_fix_wordid.index

                    for id in id_fix_wordid:
                        self.data.loc[id, "word_in_screen_id"] = (self.data.loc[id, "word_in_screen_id"]) - 1
                    # self.data.drop(id_drop, axis=0, inplace=True)

                # 1b) Treat l2[1] as standalone fixation
                else:
                    # this fixation is not part of another one -> treat it as if only the second part of the word had been fixated (see above)
                    idx_to_adjust = lines_to_drop1.index[0].item()
                    # merge the word
                    self.data.at[idx_to_adjust, "word"] = merged_word
                    # fix word IDs of lines that come after the merged line
                    lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line3)
                    # fix word id
                    self.data.at[idx_to_adjust, "word_in_screen_id"] = (
                                self.data.loc[idx_to_adjust, "word_in_screen_id"] - 1)

                # ib) l2[2] subsequent to l1 -> merge
                if lines_to_merge.index == lines_to_drop2.index - 1:  # subsequent fixations, e.g. Eiszeiten, subject 4
                    id_merge = lines_to_merge.index[0].item()
                    id_drop = lines_to_drop2.index[0].item()

                    self.data.loc[id_merge, "word"] = merged_word
                    self.data.loc[id_drop, "word"] = merged_word
                    # n fix -> add the ones from word 2 to word 1
                    self.data.loc[id_merge, "n_fix"] = self.data.loc[id_merge, "n_fix"] + self.data.loc[
                        id_drop, "n_fix"]
                    self.data.loc[id_drop, "n_fix"] = self.data.loc[id_merge, "n_fix"] + self.data.loc[id_drop, "n_fix"]
                    # adjust ffd of word 2 with ffd of word 2
                    self.data.loc[id_drop, "ffd"] = self.data.loc[id_merge, "ffd"]


                    lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line3)

                    id_fix_wordid = lines_to_fix_wordid.index

                    for id in id_fix_wordid:
                        self.data.loc[id, "word_in_screen_id"] = (self.data.loc[id, "word_in_screen_id"]) - 1
                    # self.data.drop(id_drop, axis=0, inplace=True)


                # ia) Treat l2[2] as standalone fixation
                else:
                    # this fixation is not part of another one -> treat it as if only the second part of the word had been fixated (see above)
                    idx_to_adjust = lines_to_drop2.index[0].item()
                    # merge the word
                    self.data.at[idx_to_adjust, "word"] = merged_word
                    # fix word IDs of lines that come after the merged line
                    lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line3)
                    # fix word id
                    self.data.at[idx_to_adjust, "word_in_screen_id"] = (
                            self.data.loc[idx_to_adjust, "word_in_screen_id"] - 1)

            # ----------------------
            # 3ii) l1 more than once , l2 fixated once, all 3 fixations subsequent -> merge l2 into l1[2], leave l1[1]
            elif len(lines_to_merge) >= 2 and len(
                    lines_to_drop) == 1:

                line_to_merge1 = lines_to_merge.iloc[[0]]  # first fixation on first part of word
                id_merge1 = line_to_merge1.index[0].item()
                line_to_merge2 = lines_to_merge.iloc[[1]]  # second fixation on first part of word
                id_merge2 = line_to_merge2.index[0].item()
                id_drop = lines_to_drop.index[0].item()


                self.data.loc[id_merge2, "word"] = merged_word
                self.data.loc[id_drop, "word"] = merged_word
                # n fix -> add the ones from word 2 to word 1
                self.data.loc[id_merge2, "n_fix"] = self.data.loc[id_merge2, "n_fix"] + self.data.loc[
                    id_drop, "n_fix"]
                self.data.loc[id_drop, "n_fix"] = self.data.loc[id_merge2, "n_fix"] + self.data.loc[id_drop, "n_fix"]
                # adjust ffd of word 2 with ffd of word 2
                self.data.loc[id_drop, "ffd"] = self.data.loc[id_merge2, "ffd"]


                lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line3)

                id_fix_wordid = lines_to_fix_wordid.index

                for id in id_fix_wordid:
                    self.data.loc[id, "word_in_screen_id"] = (self.data.loc[id, "word_in_screen_id"]) - 1
                # self.data.drop(id_drop, axis=0, inplace=True)

                # treat l1[1] like case l1 fixated, l2 not fixated BUT don't move word id up!
                # merge the word
                self.data.at[id_merge1, "word"] = merged_word



            # -----------------------------------------
            # 3iii) l1 fixated twice, l2 fixated twice.
            # l1[1] & l2[1] and l1[2] & l2[2] are subsequent fixations.
            # -> merge l2[1] into l1[1] and merge l2[2] into l1[1]
            elif len(lines_to_merge) == 2 and len(lines_to_drop) == 2:
                line_to_merge1 = lines_to_merge.iloc[[0]]  # l1[1]
                line_to_merge2 = lines_to_merge.iloc[[1]]  # l1[2]
                line_to_drop1 = lines_to_drop.iloc[[0]]  # l2[1]
                line_to_drop2 = lines_to_drop.iloc[[1]]  # l2[2]
                id_merge1 = line_to_merge1.index[0].item()
                id_merge2 = line_to_merge2.index[0].item()
                id_drop1 = line_to_drop1.index[0].item()
                id_drop2 = line_to_drop2.index[0].item()

                # first pair: merge l2[1] into l1[1]

                self.data.loc[id_merge1, "word"] = merged_word
                self.data.loc[id_drop1, "word"] = merged_word
                # n fix -> add the ones from word 2 to word 1
                self.data.loc[id_merge1, "n_fix"] = self.data.loc[id_merge1, "n_fix"] + self.data.loc[id_drop1, "n_fix"]
                self.data.loc[id_drop1, "n_fix"] = self.data.loc[id_merge1, "n_fix"] + self.data.loc[id_drop1, "n_fix"]
                # adjust ffd of word 2 with ffd of word 2
                self.data.loc[id_drop1, "ffd"] = self.data.loc[id_merge1, "ffd"]


                # drop lines for first pair
                lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line3)
                id_fix_wordid = lines_to_fix_wordid.index

                for id in id_fix_wordid:
                    self.data.loc[id, "word_in_screen_id"] = (self.data.loc[id, "word_in_screen_id"]) - 1
                # self.data.drop(id_drop1, axis=0, inplace=True)

                # if they directly follow one another
                if id_merge2 == id_drop2 - 1:
                    # second pair: merge l2[2] into l1[1]
                    self.data.loc[id_merge2, "word"] = merged_word
                    self.data.loc[id_drop2, "word"] = merged_word
                    # n fix -> add the ones from word 2 to word 1
                    self.data.loc[id_merge2, "n_fix"] = self.data.loc[id_merge2, "n_fix"] + self.data.loc[
                        id_drop, "n_fix"]
                    self.data.loc[id_drop2, "n_fix"] = self.data.loc[id_merge2, "n_fix"] + self.data.loc[id_drop2, "n_fix"]
                    # adjust ffd of word 2 with ffd of word 2
                    self.data.loc[id_drop2, "ffd"] = self.data.loc[id_merge, "ffd"]


                    # drop lines for second pair
                    for id in id_fix_wordid:
                        self.data.loc[id, "word_in_screen_id"] = (self.data.loc[id, "word_in_screen_id"]) - 1
                    # self.data.drop(id_drop2, axis=0, inplace=True)

            # -----------------------------------------
            # 3iii) l1 fixated twice, l2 fixated twice.
            # l1[1] & l2[1] and l1[2] & l2[2] are subsequent fixations.
            # -> merge l2[1] into l1[1] and merge l2[2] into l1[1]
            elif len(lines_to_merge) == 1 and len(lines_to_drop) == 1:
                line_to_merge1 = lines_to_merge.iloc[[0]]  # l1[1]
                line_to_drop1 = lines_to_drop.iloc[[0]]  # l2[1]
                id_merge1 = line_to_merge1.index[0].item()
                id_drop1 = line_to_drop1.index[0].item()

                self.data.loc[id_merge1, "word"] = merged_word
                self.data.loc[id_drop1, "word"] = merged_word
                # n fix -> add the ones from word 2 to word 1
                self.data.loc[id_merge1, "n_fix"] = self.data.loc[id_merge1, "n_fix"] + self.data.loc[id_drop1, "n_fix"]
                self.data.loc[id_drop1, "n_fix"] = self.data.loc[id_merge1, "n_fix"] + self.data.loc[id_drop1, "n_fix"]
                # adjust ffd of word 2 with ffd of word 2
                self.data.loc[id_drop1, "ffd"] = self.data.loc[id_merge1, "ffd"]


                # drop lines for first pair
                lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line3)
                id_fix_wordid = lines_to_fix_wordid.index

                for id in id_fix_wordid:
                    self.data.loc[id, "word_in_screen_id"] = (self.data.loc[id, "word_in_screen_id"]) - 1
                # self.data.drop(id_drop1, axis=0, inplace=True)
        # ----------------------------------------------------------------------------------¨
        # ----------------------------------------------------------------------------------
        # --------------------------CASE 3I - NEW ENDE ------------------------------
        # ----------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------

        # 4. l1 not fixated, l2 not fixated:
        for subj in problem_subjs[3]:
            # print("subj problem 4: ", subj)
            # subj 4, 11, 12, 14, 19,  16, 23, 26, 28, 33, 34, 35, 41, 42, 49, 50, 56
            # update word ID after line 2
            lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line3)
            ids_fix_wordid = lines_to_fix_wordid.index
            for id_fix_wordid in ids_fix_wordid:
                self.data.at[id_fix_wordid, "word_in_screen_id"] = (self.data.loc[id_fix_wordid, "word_in_screen_id"] - 1)

        # ------------------------------------------------------------------------------------------------------------
        # calculate reading measures for regressive outgoing/incoming sac and progressive outgoing sac after merge
        # because it depends on word id which gets changed by cleanin (merging split ids)
        # ----------------
        # calculate normalized incoming regression count, normalized outgoing regression/progression, fpr
        # 1) calculate for every wort whether incoming/outgoing regression/progression
        # outgoing progressive sac: word id in current line < word id in line below
        # grouped_fixations1 = self.data.groupby(['subject_id', 'session_id', 'text_id', 'screen_id'],
        #                                           group_keys=True)
        # grouped_fixations2 = self.data.groupby(['subject_id', 'session_id', 'text_id'], group_keys=True)
        # shifted_word_id_down = grouped_fixations1['word_in_screen_id'].shift(
        #     periods=-1)  # word id of the row below current id
        # diff_word_id = self.data['word_in_screen_id'] - shifted_word_id_down
        # self.data['outgoing_sac_progressive'] = diff_word_id.apply(lambda x: 1 if x < 0 else 0)
        # # Fill the `NaN` values with `0` in the first row of each group
        # self.data['outgoing_sac_progressive'] = (self.data['outgoing_sac_progressive'].fillna(0))
        # # Make sure the values in the 'outgoing_sac_progressive' column are integers
        # self.data['outgoing_sac_progressive'] = self.data['outgoing_sac_progressive'].astype(int)
        #
        # # outgoing regressive sac: word id in current line > word id in line below
        # self.data['outgoing_sac_regressive'] = diff_word_id.apply(lambda x: 1 if x > 0 else 0)
        # self.data['outgoing_sac_regressive'] = (self.data['outgoing_sac_regressive'].fillna(0))
        # self.data['outgoing_sac_regressive'] = self.data['outgoing_sac_regressive'].astype(int)
        #
        # # incoming regressive sac: word id in current line < word id in line above
        # shifted_word_id_up = grouped_fixations1['word_in_screen_id'].shift(
        #     periods=1)  # word id of the row above current row
        # diff_word_id_upwards = self.data['word_in_screen_id'] - shifted_word_id_up
        # self.data['incoming_sac_regressive'] = diff_word_id_upwards.apply(lambda x: 1 if x < 0 else 0)
        # # fixations_df['incoming_sac_regressive'] = diff_word_id_upwards.apply(lambda x: 1 if x < 0 else 0 if x != 0 else 0)
        # self.data['incoming_sac_regressive'] = (self.data['incoming_sac_regressive'].fillna(0))
        # self.data['incoming_sac_regressive'] = self.data['incoming_sac_regressive'].astype(int)
        #
        # # fpr: if the outgoing saccade is regressive and the RUN ID is 1
        # if self.data['CURRENT_FIX_INTEREST_AREA_RUN_ID'].dtype != 'int64':
        #     self.data['CURRENT_FIX_INTEREST_AREA_RUN_ID'] = self.data[
        #         'CURRENT_FIX_INTEREST_AREA_RUN_ID'].astype(
        #         float).astype(int)
        # self.data['fpr'] = ((self.data['outgoing_sac_regressive'] == 1) & (
        #         self.data['CURRENT_FIX_INTEREST_AREA_RUN_ID'] == 1)).astype(int)
        #
        # # normalized outgoing progressive/regressive saccades
        # # sum up outgoing progr/regr saccades on individual word
        # outgoing_progressive_sac_on_word = self.data['outgoing_sac_progressive'].sum()
        # outgoing_regressive_sac_on_word = self.data['outgoing_sac_regressive'].sum()
        # incoming_regressive_sac_on_word = self.data['incoming_sac_regressive'].sum()
        # self.data['outgoing_progressive_sac_on_word'] = self.data.apply(
        #     lambda x: outgoing_progressive_sac_on_word[
        #         x['subject_id'], x['session_id'], x['text_id'], x['screen_id'], x[
        #             'word_in_screen_id']], axis=1)
        # self.data['outgoing_regressive_sac_on_word'] = self.data.apply(
        #     lambda x: outgoing_regressive_sac_on_word[
        #         x['subject_id'], x['session_id'], x['text_id'], x['screen_id'], x[
        #             'word_in_screen_id']], axis=1)
        # self.data['incoming_regressive_sac_on_word'] = self.data.apply(
        #     lambda x: incoming_regressive_sac_on_word[
        #         x['subject_id'], x['session_id'], x['text_id'], x['screen_id'], x[
        #             'word_in_screen_id']], axis=1)
        #
        # # sum of regressive saccades in scanpath
        # regressions_total = grouped_fixations2['outgoing_sac_regressive'].sum()
        # self.data['outgoing_regressive_sac_scanpath'] = self.data.apply(
        #     lambda x: regressions_total[x['subject_id'], x['session_id'], x['text_id']], axis=1)
        # # sum of progressive saccades in scanpath
        # progressions_total = grouped_fixations2['outgoing_sac_progressive'].sum()
        # self.data['outgoing_progressive_sac_scanpath'] = self.data.apply(
        #     lambda x: progressions_total[x['subject_id'], x['session_id'], x['text_id']], axis=1)
        #
        # # Normalize scores on word level
        # self.data["outgoing_sac_regressive_norm"] = round((self.data['outgoing_regressive_sac_on_word'] /
        #                                                       self.data['outgoing_regressive_sac_scanpath']),
        #                                                      6)
        # self.data["outgoing_sac_progressive_norm"] = round(
        #     (self.data['outgoing_progressive_sac_on_word'] /
        #      self.data['outgoing_progressive_sac_scanpath']), 6)
        # self.data["incoming_sac_regressive_norm"] = round((self.data['incoming_regressive_sac_on_word'] /
        #                                                       self.data['outgoing_regressive_sac_scanpath']),
        #                                                      6)

    def clean_remove_line(self, text, screen, line1, line2, seperated_word, n_lines_rmv):
        """Function to fix ET problems where two words falesly got merged into 1 line.
        Remove the entry and update wordIDs.
        - problem 3: zukanalisieren, (79) (fix) vs. zu + kanalisieren,
        - problem 4: hatten.Kleine (85)(fix) vs. hatten. + Kleine
        - problem 5: Pflanzenwelt.Es (fix) vs. Pflanzenwelt.
        line 1 = line where word that should be split is
        line 2 = line from which word IDs have to be updated
        n_lines_rmv is the number of lines that need to be removed -> 1 for cases in InDiCo where 2 words got merged
        together and 2 for cases in sbsat where 3 tokens them-apart got merged (dash)
        """
        # get problematic lines: Which subjects fixated on the line and which didn't?
        problem_subjs = self.get_problematic_fixations(text, screen, line1, line2)
        word_fixated = problem_subjs[0] + problem_subjs[2]  # subj 6, 12, 25, 30, 33, 37, 42, 18, 18, 55, 57
        word_not_fixated = problem_subjs[1] + problem_subjs[3]  # subj 7, 19, 21, 27, 36, 49, 54, 61

        # subject fixated on the words that should be split -> move subsequent lines 1 idx up & rename line
        for subj in word_fixated:
            line_to_rename = self.get_line_by_wordid(subj, text, screen, line1)
            id_rename = line_to_rename.index[0].item()
            lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line2)
            id_fix_wordid = lines_to_fix_wordid.index
            # move index
            for i in id_fix_wordid:
                self.data.at[i, "word_in_screen_id"] = (self.data.loc[i, "word_in_screen_id"] + n_lines_rmv)
            # rename cell word
            self.data.at[id_rename, "word"] = seperated_word

        # subject didn't fixate on the words that should be split -> move up the indeces by 1
        for subj in word_not_fixated:
            lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line2)
            id_fix_wordid = lines_to_fix_wordid.index
            for i in id_fix_wordid:
                self.data.at[i, "word_in_screen_id"] = (self.data.loc[i, "word_in_screen_id"] + n_lines_rmv)

        # check if it worked (use id 6 for word fixated; 7 for word not fixated)
        # df_filtered = self.data[
        #     (self.data['word_in_screen_id'] >= 76) & (self.data['word_in_screen_id'] <= 86)]
        # df_filtered = df_filtered[
        #     (df_filtered['subject_id'] == 6) & (df_filtered['text_id'] == 4) & (df_filtered['screen_id'] == 4)]
        # print(df_filtered.head(100))

    def clean_rename_line(self, text, screen, line1, line2, seperated_word):

        """Function to change word in ET data
        Remove the entry and update wordIDs.
        - problem 4: www.vhb.org vs. VHB
        line 1 = word to change
        """

        # get problematic lines: Which subjects fixated on the line and which didn't?
        problem_subjs = self.get_problematic_fixations(text, screen, line1, line2)


        word_fixated = problem_subjs[0] + problem_subjs[2]  # subj 6, 12, 25, 30, 33, 37, 42, 18, 18, 55, 57
        word_not_fixated = problem_subjs[1] + problem_subjs[3]  # subj 7, 19, 21, 27, 36, 49, 54, 61

        # subject fixated on the words that should be split -> move subsequent lines 1 idx up & rename line
        for subj in word_fixated:
            line_to_rename = self.get_line_by_wordid(subj, text, screen, line1)
            ids_rename = line_to_rename.index.tolist()
            for id_rename in ids_rename:
                self.data.at[id_rename, "word"] = seperated_word
        # print("blubbblubbblubb: ")
        # # df_filtered = self.data[
        # #     (self.data['word_in_screen_id'] >= 92) & (self.data['word_in_screen_id'] <= 96)]
        # # df_filtered = df_filtered[(df_filtered['text_id'] == 1) & (df_filtered['screen_id'] == 1)]
        # # print(df_filtered.head(20))

    def clean_adjust_ids(self, text, screen, line1, line2, seperated_word):
        """in this example: text 1, screen 3, wordid 43: lexical features word & vs. reading measures word Sons
        & is missing -> move everyting after Sempre one line up because & is not there in Sempre & sons
        """
        # get problematic lines: Which subjects fixated on the line and which didn't?
        problem_subjs = self.get_problematic_fixations(text, screen, line1, line2)
        problem_subjs = sum(problem_subjs, []) # flatten to one list

        # subject fixated on the words that should be split -> move subsequent lines 1 idx up & rename line
        for subj in problem_subjs:
            lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line2)
            id_fix_wordid = lines_to_fix_wordid.index.tolist() # everything after line 1
            for i in id_fix_wordid:
                self.data.at[i, "word_in_screen_id"] = (self.data.loc[i, "word_in_screen_id"] + 1)

    def clean_remove_line_v2(self, text, screen, line1, line2, seperated_word):
        """Second version for removing multiple lines because in case of hyphen, it merged together multiple lines
        in fixation data, when in lexfeats, it's seperate lines, e.g. text 1, screen 1, wordid 107: lexical features
        word them vs. reading measures word them, -, apart
        """
        # get problematic lines: Which subjects fixated on the line and which didn't?
        problem_subjs = self.get_problematic_fixations(text, screen, line1, line2)
        word_fixated = problem_subjs[0] + problem_subjs[2]  # subj 6, 12, 25, 30, 33, 37, 42, 18, 18, 55, 57
        word_not_fixated = problem_subjs[1] + problem_subjs[3]  # subj 7, 19, 21, 27, 36, 49, 54, 61

        # subject fixated on the words that should be split -> move subsequent lines 1 idx up & rename line
        for subj in word_fixated:
            line_to_rename = self.get_line_by_wordid(subj, text, screen, line1)
            id_rename = line_to_rename.index.tolist()
            lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line2)
            id_fix_wordid = lines_to_fix_wordid.index
            # move index
            for i in id_fix_wordid:
                self.data.at[i, "word_in_screen_id"] = (self.data.loc[i, "word_in_screen_id"] + 2)
            # rename cell word
            for id in id_rename:
                self.data.at[id, "word"] = seperated_word

        # subject didn't fixate on the words that should be split -> move up the indeces by 1
        for subj in word_not_fixated:
            lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line2)
            id_fix_wordid = lines_to_fix_wordid.index
            for i in id_fix_wordid:
                self.data.at[i, "word_in_screen_id"] = (self.data.loc[i, "word_in_screen_id"] + 2)

        # check if it worked (use id 6 for word fixated; 7 for word not fixated)
        # df_filtered = self.data[
        #     (self.data['word_in_screen_id'] >= 76) & (self.data['word_in_screen_id'] <= 86)]
        # df_filtered = df_filtered[
        #     (df_filtered['subject_id'] == 6) & (df_filtered['text_id'] == 4) & (df_filtered['screen_id'] == 4)]
        # print(df_filtered.head(100))

    def clean_drop_line(self, text, screen, line1, line2, seperated_word):
        """remove one line, move others one idx up
        for proplem in text 2, screen 3:text 2, screen 3, wordid 28: lexical features word early vs. reading measures word Burdon-Sanderson’s
        (Burdon-Sanderson’s appeared twice at idx 27, 28 -> this takes care of 28)

        """
        # get problematic lines: Which subjects fixated on the line and which didn't?
        problem_subjs = self.get_problematic_fixations(text, screen, line1, line2)
        word_fixated = problem_subjs[0] + problem_subjs[2]  # subj 6, 12, 25, 30, 33, 37, 42, 18, 18, 55, 57
        word_not_fixated = problem_subjs[1] + problem_subjs[3]  # subj 7, 19, 21, 27, 36, 49, 54, 61

        # subject fixated on the words that should be split -> move subsequent lines 1 idx up & rename line
        for subj in word_fixated:
            # drop line
            line_to_drop = self.get_line_by_wordid(subj, text, screen, line1)
            ids_drop = line_to_drop.index.tolist()
            for id_drop in ids_drop:
                self.data.drop(id_drop, axis=0, inplace=True)

            lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line2)
            id_fix_wordid = lines_to_fix_wordid.index
            # move index
            for i in id_fix_wordid:
                self.data.at[i, "word_in_screen_id"] = (self.data.loc[i, "word_in_screen_id"] - 1)



        # subject didn't fixate on the words that should be split -> move up the indeces by 1
        for subj in word_not_fixated:
            lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line2)
            id_fix_wordid = lines_to_fix_wordid.index
            for i in id_fix_wordid:
                self.data.at[i, "word_in_screen_id"] = (self.data.loc[i, "word_in_screen_id"] + 1)

    def clean_adapt_id(self, text, screen, line1, line2):
        """
        # todo
        in case of hyphens
        For 2/6 -> text 2, screen 6, wordid 41: lexical features word — vs. reading measures word NA
        delete some lines because - not in fix"""
        problem_subjs = self.get_problematic_fixations(text, screen, line1, line2)
        word_fixated = problem_subjs[0] + problem_subjs[2]
        word_not_fixated = problem_subjs[1] + problem_subjs[3]

        # if word fixated, drop line, move up ids by 2
        for subj in word_fixated:
            line_to_drop = self.get_line_by_wordid(subj, text, screen, line1)
            ids_drop = line_to_drop.index.tolist()
            for id_drop in ids_drop:
                self.data.drop(id_drop, axis=0, inplace=True)
            # change ids +2
            lines_to_change = self.get_lines_geq_by_wordid(subj, text, screen, line2)
            ids_change = lines_to_change.index.tolist()
            for i in ids_change:
                self.data.at[i, "word_in_screen_id"] = (self.data.loc[i, "word_in_screen_id"] + 2)


        for subj in word_not_fixated:
            lines_to_fix_wordid = self.get_lines_geq_by_wordid(subj, text, screen, line2)
            id_fix_wordid = lines_to_fix_wordid.index
            for i in id_fix_wordid:
                self.data.at[i, "word_in_screen_id"] = (self.data.loc[i, "word_in_screen_id"] + 2)


        #if word not fixated, only move up ids by 2

    def get_problematic_fixations(self, text: int, screen: int, l1: int, l2: int) -> Tuple[List[int]]:
        """Helper fct. for clean_indico_optimized. Returns 4 lists with subject IDs of subjects that fixated on one, both
        or neither part of the problematic word (e.g. 'Eis' and 'Zeitein').
        Function replaces readerTextDict from the original clean function.
        text, screen = text & screen where the problem occured (where the word was fixated).
        l1, l2 = the two word id's from the lines that need to be merged."""
        # subdf with text and screen present
        subdf = self.data.loc[(self.data["text_id"] == text) & (self.data["screen_id"] == screen)]

        # get sub-dataframes for the subjects who fixated on both, neither or one of the
        # problematic words (=line present in df)
        fix_l1_not_l2 = sorted(list(set(subdf.loc[subdf["word_in_screen_id"].isin([l1, l2])]["subject_id"].unique())
                                    - set(subdf.loc[subdf["word_in_screen_id"] == l2]["subject_id"].unique())))
        fix_l2_not_l1 = sorted(list(set(subdf.loc[subdf["word_in_screen_id"].isin([l1, l2])]["subject_id"].unique())
                                    - set(subdf.loc[subdf["word_in_screen_id"] == l1]["subject_id"].unique())))
        fix_both = sorted(list(set(subdf.loc[subdf["word_in_screen_id"].isin([l1, l2])]["subject_id"].unique())
                               - set(fix_l2_not_l1) - set(fix_l1_not_l2)))
        fix_neither = sorted(list(set(subdf["subject_id"].unique())
                                  - set(subdf.loc[subdf["word_in_screen_id"].isin([l1, l2])]["subject_id"].unique())))

        return fix_l1_not_l2, fix_l2_not_l1, fix_both, fix_neither

    def get_line_by_wordid(self, subjId, textId, screenId, wordId):
        """Helper function for clean() to get lines with mistake where clean function should be performed."""
        return self.data[
            (self.data["subject_id"] == subjId)
            & (self.data["text_id"] == textId)
            & (self.data["screen_id"] == screenId)
            & (self.data["word_in_screen_id"] == wordId)
        ]

    def get_lines_geq_by_wordid(self, subjId, textId, screenId, wordId):
        """Helper function for clean () to get lines after mistake where word Id should be updated."""
        return self.data[
            (self.data["subject_id"] == subjId)
            & (self.data["text_id"] == textId)
            & (self.data["screen_id"] == screenId)
            & (self.data["word_in_screen_id"] >= wordId)
        ]

    def merge_with_lexfeats_optimized(self):
        print("start merging...")

        # rename columns in the features df
        self.lexFeat.rename(
            columns={
                "text_number": "text_id",
                "screen_number": "screen_id",
                "w_in_screen_id": "word_in_screen_id",
                "surprisal_bert": "surprisal",
            },
            inplace=True,
        )

        # merge the dataframes for fixation and lexical features
        self.data = pd.merge(self.data, self.lexFeat, how="left",
                             on=['text_id', 'screen_id', 'word_in_screen_id'])

        # convert dtypes if necessary
        self.data = self.data.astype({"CURRENT_FIX_X": "float64", "CURRENT_FIX_Y": "float64"})

        # re-order rows
        self.data.index = [self.data.text_id, self.data.screen_id, self.data.word_in_screen_id]
        self.data.drop(
            [
                "text_id",
                "screen_id",
                "word_in_screen_id"
            ],
            axis=1,
            inplace=True,
        )

        # delete extra row 103 which appeared in text 3, screen 3 of sbsat due to corrections. this line doesn't exist
        # in the lexical features and is filled with NaN values
        self.data.dropna(subset=["word_y"], inplace=True)

        # in fixation data, word-final punctuation needs to be deleted.
        # create a boolean mask indicating which rows need to be updated
        mask = (self.data['word_x'].str[-1].isin([char for char in string.punctuation])) & (
            ~self.data['word_y'].str[-1].isin([char for char in string.punctuation]))
        # update the words column using the boolean mask
        self.data.loc[mask, 'word_y'] = self.data.loc[mask, 'word_y'] + self.data.loc[mask, 'word_x'].str[-1]
        self.data['word_x'].update(self.data['word_y'])

        # the columns are now equal, we can drop one of them and rename the other to 'word'
        if self.data['word_x'].equals(self.data['word_y']):
            print("TRUUUE!")
            self.data = self.data.rename(columns={'word_x': 'word'})
            self.data = self.data.drop(columns='word_y')

        self.data.reset_index(inplace=True)

        # print("Columns after indexing: ")
        # print(
        #     self.data[["fixation_id", "word_in_screen_id", "CURRENT_FIX_INTEREST_AREA_RUN_ID", "word_x", "word_y",
        #     "CURRENT_FIX_X", "CURRENT_FIX_Y"]].iloc[50:70])

        column_order = ['subject_id', 'session_id', 'text_id', 'screen_id', 'fixation_id', 'word_in_screen_id',
                        'word', 'CURRENT_FIX_X', 'CURRENT_FIX_Y', 'CURRENT_FIX_PUPIL', 'CURRENT_FIX_DURATION',
                        'ffd', 'tfd', 'n_fix', 'fpr', 'incoming_sac_regressive_norm', 'outgoing_sac_regressive_norm',
                        'outgoing_sac_progressive_norm', 'CURRENT_FIX_INTEREST_AREA_RUN_ID', 'sentence_number', 'lemma',
                        'word_n_char', 'pos', 'simplified_pos', 'wordfreq_lemma', 'NE_IOB', 'NE_label', 'content_word',
                        'technical_term', 'synonym_homonym', 'synt_surprisal', 'surprisal', 'deps', 'rights', 'lefts',
                        'n_rights', 'n_lefts', 'dep_distance', 'sent_n_words', 'sent_n_char', 'sent_mean_word_length',
                        'sent_lexical_density_tokens', 'sent_cut', 't_n_char', 't_n_words', 't_n_phrases',
                        't_mean_word_length', 't_lexical_density', 't_lemma_TTR', 't_content_w_TTR', 't_function_w_TTR',
                        't_nouns_TTR', 't_verbs_TTR', 't_adj_TTR', 't_adv_TTR', 't_genre', 'tf', 'idf', 'tf_idf',
                        'lex_overlap', 'semantic_similarity_adjacent_sent', 'pron_det_to_noun_ratio', 'voice']

        self.data = self.data.reindex(columns=column_order)

        return self.data

    def _get_words(self):
        """Helper function for check_alignment function. Get words of the screens, index by text, screen, word"""
        words = self.data.copy()
        words.drop(["subject_id"], axis=1, inplace=True)
        words = (
            words.sort_values(by=["text_id", "screen_id", "word_in_screen_id"])[
                ["text_id", "screen_id", "word_in_screen_id", "word"]
            ]
            .drop_duplicates(subset=["text_id", "screen_id", "word_in_screen_id"])
            .reset_index(drop=True)
        )
        words.index = [words.text_id, words.screen_id, words.word_in_screen_id]
        words.drop(["text_id", "screen_id", "word_in_screen_id"], axis=1, inplace=True)
        return words

    def check_alignment_indico(self, n_texts, n_screens):
        """Locate alignment problems between lexical features file & fixation report & print them with text, screen and
        word ID.
        input: n_texts is the number of texts in the dataset (indico 16, sbsat 4)
        n_screens is the number of screens in the texts (indico 5, sbsat 5-6)"""
        non_fixated = 0
        reading_data = self.data.copy()  # table with reading measures

        reading_data.index = [
            reading_data.text_id,
            reading_data.screen_id,
            reading_data.word_in_screen_id,
        ]
        reading_data.drop(
            [
                "text_id",
                "screen_id",
            ],
            axis=1,
            inplace=True,
        )

        words_reading_data = self._get_words()

        text_screen = []
        problems = []
        chars = set("ÄÖÜäöüß„”“–")
        for text in range(1, n_texts+1):
            for screen in range(1, n_screens+1):
                text_screen.append((text, screen))
                for wn in range(1, reading_data.loc[text, screen, :].word_in_screen_id.max()):
                    try:
                        w_lex_feat = self.lexFeat.loc[text, screen, wn].word.strip(
                            "().,:;?!"
                        )
                        w_rm = words_reading_data.loc[text, screen, wn].word.strip(
                            "().,:;?!”„'“"
                        )

                        if w_lex_feat != w_rm:
                            # problems.append((text, screen, wn, w_lex_feat, w_rm))
                            if any((c in chars) for c in w_lex_feat):
                                continue
                            else:
                                print(
                                    f"text {text}, screen {screen}, wordid {wn}: "
                                    f"lexical features word {w_lex_feat} vs. reading measures word {w_rm}"
                                )
                                if (text, screen) not in problems:
                                    problems.append((text, screen))
                                break
                    except KeyError:

                        # print(self.lexFeat.loc[text, screen, wn-1:])
                        # print(words_reading_data.loc[text, screen, wn-1:])
                        # print(text, screen, wn)

                        if self.lexFeat.loc[text, screen, wn].word in ["-", "–"]:
                            continue
                        else:
                            # keep count of non-fixated words
                            non_fixated += 1
                            # print(f'missing word: {self.lexFeat.loc[text, screen, wn].word} in {text, screen, wn}')
                            # print(words_reading_data.loc[text, screen, wn - 1:wn + 2])
                            continue

        # print(f'non-fixated words = {non_fixated}')
        return problems

    def check_alignment(self, n_texts):
        """Locate alignment problems between lexical features file & fixation report & print them with text, screen and
        word ID.
        input: n_texts is the number of texts in the dataset (indico 16, sbsat 4)
        n_screens is the number of screens in the texts (indico 5, sbsat 5-6)"""
        non_fixated = 0
        reading_data = self.data.copy()  # table with reading measures

        reading_data.index = [
            reading_data.text_id,
            reading_data.screen_id,
            reading_data.word_in_screen_id,
        ]
        reading_data.drop(
            [
                "text_id",
                "screen_id",
            ],
            axis=1,
            inplace=True,
        )

        words_reading_data = self._get_words()

        text_screen = []
        problems = []
        chars = set("ÄÖÜäöüß„”“–")
        for text in range(1, n_texts + 1):
            n_actual_screens = max(reading_data.loc[text].index.get_level_values('screen_id'))
            for screen in range(1, n_actual_screens + 1):
                text_screen.append((text, screen))
                for wn in range(1, reading_data.loc[text, screen, :].word_in_screen_id.max()):
                    try:
                        w_lex_feat = self.lexFeat.loc[text, screen, wn].word.strip(
                            "().,:;?!"
                        )
                        w_rm = words_reading_data.loc[text, screen, wn].word.strip(
                            "().,:;?!”„'“"
                        )

                        if w_lex_feat != w_rm:
                            # problems.append((text, screen, wn, w_lex_feat, w_rm))
                            if any((c in chars) for c in w_lex_feat):
                                continue
                            else:
                                print(
                                    f"text {text}, screen {screen}, wordid {wn}: "
                                    f"lexical features word {w_lex_feat} vs. reading measures word {w_rm}"
                                )
                                if (text, screen) not in problems:
                                    problems.append((text, screen))
                                break
                    except KeyError:

                        # print(self.lexFeat.loc[text, screen, wn-1:])
                        # print(words_reading_data.loc[text, screen, wn-1:])
                        # print(text, screen, wn)
                        try:
                            if self.lexFeat.loc[text, screen, wn].word in ["-", "–"]:
                                continue
                        except KeyError:
                            pass
                        else:
                            # keep count of non-fixated words
                            non_fixated += 1
                            # print(f'missing word: {self.lexFeat.loc[text, screen, wn].word} in {text, screen, wn}')
                            # print(words_reading_data.loc[text, screen, wn - 1:wn + 2])
                            continue

        # print(f'non-fixated words = {non_fixated}')
        return problems

    def check_dependency_alignment(self):
        for text in range(1, 17):
            for screen in range(1, 6):
                text_dep = self.deps[str(text)][str(screen)]["text"].split()
                assert len(text_dep) == len(
                    self.lexFeat.loc[text, screen, :]
                ), f"length does not match {text}{screen}"
                for wn in range(
                    1, self.lexFeat.loc[text, screen, :].w_in_sent_id.max()
                ):
                    try:
                        # wn-1 since list index
                        w_text_dep = text_dep[wn - 1]
                        w_lex_feat = str(self.lexFeat.loc[text, screen, wn].word)
                        if w_lex_feat != w_text_dep:
                            print(
                                f"word in lex features {w_lex_feat} != word in deps {w_text_dep}"
                            )
                    except KeyError:
                        print(f"no deps for {text, screen, wn}")
        print("done checking alignment between dependencies and lexical features")

    def __len__(self):
        return len(self.data.index)


def read_and_clean_indico():
    """Create Dataset, find alignment problems between data.csv & lexical_features_fixed.csv, merge dataframes."""
    print("Start arranging Fixation data")
    fix_df = FixationDataset(type="indico")
    # fix_df.check_dependency_alignment()
    print("Things to fix: ")
    p_before = fix_df.check_alignment(16)
    print(f"problems: {p_before}")

    # print("fixations before fixing: ")
    # df_filtered = fix_df.data[
    #     (fix_df.data['word_in_screen_id'] >= 1) & (fix_df.data['word_in_screen_id'] <= 5)]
    # df_filtered = df_filtered[
    #     (df_filtered['subject_id'] == 4) & (df_filtered['text_id'] == 9) & (df_filtered['screen_id'] == 4)]
    # print(df_filtered.head(50))

    fix_df.clean_optimized(type="indico")
    # print("Things to still fix: ")
    p_after = fix_df.check_alignment(16)

    # check if it worked

    # df_filtered = fix_df.data[
    #     (fix_df.data['word_in_screen_id'] >= 1) & (fix_df.data['word_in_screen_id'] <= 5)]
    # df_filtered = df_filtered[
    #     (df_filtered['subject_id'] == 4) & (df_filtered['text_id'] == 9) & (df_filtered['screen_id'] == 4)]
    # print(df_filtered.head(50))

    if p_after:
        sys.exit("Not all problems fixed, abort...")
    else:
        print("All problems fixed!")

    print("fixations after fixing: ")

    fix_df = fix_df.merge_with_lexfeats_optimized()  # optimized
    print("Dataframes merged!")

    return fix_df

def read_and_clean_sbsat():
    """Create Dataset, find alignment problems between data.csv & lexical_features_fixed.csv, merge dataframes."""
    print("Start arranging Fixation data")
    # fix_df = FixationDataset(type="et_indiff")
    fix_df = FixationDataset(type="sbsat")

    # fix_df.check_dependency_alignment()
    p_before = fix_df.check_alignment(4)
    print(f"problems: {p_before}")
    print("Cleaning df...")

    # --------------------------------
    # print("Data before cleaning:")
    # df_filtered = fix_df.data[
    #     (fix_df.data['word_in_screen_id'] >= 60) & (fix_df.data['word_in_screen_id'] <= 70)]
    # df_filtered = df_filtered[
    #     (df_filtered['subject_id'] == 10) & (df_filtered['text_id'] == 3) & (df_filtered['screen_id'] == 3)]
    # print(df_filtered.head(20))
    # --------------------------------

    fix_df.clean_rename_line(1, 1, 95, 96, "didn’t") # encoding
    fix_df.clean_remove_line_v2(1, 1, 107, 108, "them")  # split mulitple, hyphen ¨
    fix_df.clean_remove_line(1, 1, 116, 117, "decipher", 2)  # split multiplpe, hyphen
    fix_df.clean_rename_line(1, 2, 4, 5, "I’d")  # encoding
    fix_df.clean_rename_line(1, 2, 15, 16, "didn’t")  # encoding
    fix_df.clean_rename_line(1, 2, 74, 75, "If")  # encoding
    fix_df.clean_rename_line(1, 2, 85, 86, "you’ll")  # encoding
    fix_df.clean_rename_line(1, 2, 87, 88, "sorry")  # encoding
    fix_df.clean_rename_line(1, 3, 18, 19, "I’d")  # encoding
    fix_df.clean_rename_line(1, 3, 24, 25, "I’d")  # encoding
    fix_df.clean_rename_line(1, 3, 24, 25, "I’d")  # encoding
    fix_df.clean_remove_line(1, 3, 43, 44, "&", 1) # Sempre & Sons, it only merged & Sons together
    fix_df.clean_rename_line(1, 3, 84, 85, "heart’s")  # encoding
    fix_df.clean_rename_line(1, 3, 104, 105, "wasn’t")  # encoding
    fix_df.clean_rename_line(1, 3, 106, 107, "I’d")  # encoding
    fix_df.clean_rename_line(1, 3, 110, 111, "I’d")  # encoding
    fix_df.clean_remove_line_v2(1, 3, 124, 125, "change")  # split change?if
    fix_df.clean_rename_line(1, 3, 127, 128, "I’d")  # encoding
    fix_df.clean_rename_line(1, 4, 60, 61, "Great")  # encoding
    fix_df.clean_rename_line(1, 4, 64, 65, "Dickens")  # encoding
    fix_df.clean_rename_line(1, 4, 104, 105, "A")  # encoding
    fix_df.clean_rename_line(1, 5, 1, 2, "A")  # encoding
    fix_df.clean_rename_line(1, 5, 8, 9, "he’s")  # encoding
    fix_df.clean_rename_line(1, 5, 11, 12, "too")  # encoding
    fix_df.clean_rename_line(1, 5, 28, 29, "wouldn’t")  # encoding
    fix_df.clean_rename_line(1, 5, 96, 97, "didn’t")  # encoding
    fix_df.clean_rename_line(1, 5, 109, 110, "Mr")  # encoding Mr. Dickens -> Mr
    fix_df.clean_rename_line(1, 5, 110, 111, "Dickens")  # encoding Mr. Dickens -> Dickens
    fix_df.clean_rename_line(2, 1, 43, 44, "it’s")  # encoding
    fix_df.clean_rename_line(2, 2, 24, 25, "doesn’t")  # encoding
    fix_df.clean_rename_line(2, 2, 104, 105, "doesn’t")  # encoding
    fix_df.clean_rename_line(2, 2, 124, 125, "bug’s")  # encoding
    fix_df.clean_rename_line(2, 3, 27, 28, "Burdon-Sanderson’s")  # encoding
    fix_df.clean_drop_line(2, 3, 28, 29, "NA")  # encoding
    fix_df.clean_rename_line(2, 5, 23, 24, "don’t")  # encoding
    fix_df.clean_rename_line(2, 5, 33, 34, "won’t")  # encoding
    fix_df.clean_rename_line(2, 6, 1, 2, "didn’t")  # encoding
    fix_df.clean_remove_line_v2(2, 6, 40, 41, "microcoulombs")  # encoding
    fix_df.clean_remove_line_v2(2, 6, 55, 56, "together")  # encoding
    fix_df.clean_rename_line(3, 1, 66, 67, "pharming")  # encoding
    fix_df.clean_rename_line(3, 1, 93, 94, "bodies’")  # encoding
    fix_df.clean_rename_line(3, 2, 16, 17, "it’s")  # encoding
    fix_df.clean_remove_line_v2(3, 2, 63, 64, "animals")  # encoding
    fix_df.clean_remove_line(3, 2, 69, 70, "species", 1)  # encoding
    fix_df.clean_rename_line(3, 2, 71, 72, "in")  # encoding
    fix_df.clean_rename_line(3, 3, 9, 10, "’90s")  # encoding
    # two interest areas in one word; gee-whiz on both ia 42&43 instead of just 42
    fix_df.clean_merge_lines(3, 3, 42, 43, "gee-whiz")
    fix_df.clean_rename_line(3, 3, 43, 45, "scientific")  # encoding
    fix_df.clean_rename_line(3, 4, 96, 97, "company’s")  # encoding
    fix_df.clean_rename_line(3, 5, 59, 56, "goat’s")  # encoding
    fix_df.clean_rename_line(3, 5, 78, 79, "goats’")  # encoding
    fix_df.clean_remove_line_v2(3, 5, 103, 104, "voilà")
    fix_df.clean_rename_line(3, 5, 120, 121, "world’s")  # encoding
    fix_df.clean_rename_line(3, 6, 9, 56, "milking")  # encoding
    fix_df.clean_rename_line(3, 6, 10, 11, "parlors")  # encoding
    fix_df.clean_rename_line(3, 6, 12, 13, "GTC’s")  # encoding
    fix_df.clean_rename_line(4, 1, 33, 34, "don’t")  # encoding
    fix_df.clean_rename_line(4, 2, 113, 114, "don’t")  # encoding
    fix_df.clean_rename_line(4, 3, 44, 45, "soul’s")  # encoding
    fix_df.clean_rename_line(4, 3, 55, 56, "isn’t")  # encoding
    # filter out word id that doesn't exist in lexical features and occured due to my corrections (i.e. text 3, screen 3,


    #----------------------------------------
    # print("After cleaning: ")
    # df_filtered = fix_df.data[
    #     (fix_df.data['word_in_screen_id'] >= 60) & (fix_df.data['word_in_screen_id'] <= 70)]
    # df_filtered = df_filtered[
    #     (df_filtered['subject_id'] == 3) & (df_filtered['text_id'] == 3) & (df_filtered['screen_id'] == 3)]
    # print(df_filtered.head(20))
    #---------------------------------------

    p_after = fix_df.check_alignment(4)
    print(f"problems after: {p_after}")
    if p_after:
        sys.exit("Not all problems fixed, abort...")
    else:
        print("All problems fixed!")

    fix_df = fix_df.merge_with_lexfeats_optimized()  # optimized
    print("Dataframes merged!")

    return fix_df


def process_data(type: str = "both", write: bool = True) -> object:
    """Read & clean specified data (ET/SPR/both) & write the output to a csv file."""

    if type == "indico":
        et_df = read_and_clean_indico()
        if write:
            et_df.to_csv("data/InDiCo/processed/indico_fix_lexfeats_final.csv")
    elif type == "sbsat":
        et_df = read_and_clean_sbsat()
        # for x in et_df.columns:
        #     print(x)
        if write:
            et_df.to_csv("data/SB-SAT/processed/sbsat_fix_lexfeats_final.csv")

        pass


def main() -> int:
    if args.InDiCo is True:
        process_data(type="indico", write=True)

        # read_and_clean_indico()

    elif args.SBSAT is True:
        # print(read_and_clean_sbsat())
        process_data(type="sbsat", write=True)

        # read_and_clean_sbsat()




    return 0


if __name__ == "__main__":
    raise SystemExit(main())