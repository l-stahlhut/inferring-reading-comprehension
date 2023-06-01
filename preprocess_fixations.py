"""
Preprocess raw fixations
INDICO: more extensive preprocessing. Raw fixation report -> clean -> extract & rename columns
SBSAT: drop some lines and columns, add subj id, reformat and reset some datatypes. This data was likely already processedby ahn et al.
To process a file containing the fixations of all sessions, run:
python3 src/preprocess_indico_fixations.py --InDiCo --all_sessions

To process the files containing the fixations of an individual session each, run:
python3 src/preprocess_indico_fixations.py --InDiCo --individual_sessions
SBSAT:
python3 src/preprocess_indico_fixations.py --SBSAT

"""

import argparse
import pandas as pd
import re
from map_fix_to_roi import get_aois_from_event_data
import os
from typing import List
import numpy as np

parser = argparse.ArgumentParser(
    prog='preprocess_indico_fixations',
    description='Preprocess raw fixations from indico (Dataviewer output): Clean & map interest areas')
parser.add_argument('--all_sessions', action='store_true', help='Process the txt file containing all sessions')
parser.add_argument('--individual_sessions', action='store_true', help='Process the txt files containing individual '
                                                                       'sessions')
parser.add_argument('--InDiCo', action='store_true', help='Process the txt file containing all sessions')
parser.add_argument('--SBSAT', action='store_true', help='Process the txt files containing individual '
                                                                       'sessions')
args = parser.parse_args()


class PreprocessedFixations():

    def __init__(self):
        if args.InDiCo is True:
            if args.all_sessions is True:
                self.infile = "data/InDiCo/raw/fixation/all_participants/Output/indico_all_fix_raw.txt"
                # self.infile = "data/InDiCo/interim/fixation/fix_pipeline_ET_004_2/indico_individual_fix_raw_ET_004_2.csv" # for individual file
                # self.outfile = "data/InDiCo/interim/fixation/all_participants_with_ia_mapping/indico_all_fixations_with_ia.csv"
                self.data_cleaned = self.clean_indico_fixations(self.infile)  # log messages sorted, NA columns removed
                self.data_mapped = self.map_indico_fixations(self.infile)  # map ia char level -> ia word level
                self.data_preprocessed = self.fixfinal_indico(self.infile) # remove and rename some columns
                self.outfile = "data/InDiCo/interim/fixation/indico_fix_preprocessed.csv"
                # self.outfile = "data/InDiCo/interim/fixation/fix_pipeline_ET_004_2/ET_004_2_merged.csv" # for individual file
            elif args.individual_sessions is True:
                self.inpath = r'data/InDiCo/raw/fixation/all_participants/Output/indico_individual_fix'
                self.outpath = r'data/InDiCo/interim/fixation/sessionwise_with_ia_mapping'
                self.files = self.read_files()
        elif args.SBSAT:
            self.inpath_fixation = 'data/SB-SAT/raw/18sat_fixfinal.csv'
            self.data_preprocessed = self.fixfinal_sbsat()
            self.outfile = "data/SB-SAT/interim/fixation/sbsat_fix_preprocessed.csv"

    def read_files(self) -> List[str]:
        """Read individual files for indico"""
        filenames = [fn for fn in os.listdir(self.inpath) if
                     os.path.isfile(os.path.join(self.inpath, fn)) and fn.endswith('.txt')]
        filenames.sort()

        return filenames

    def clean_indico_fixations(self, file) -> pd.DataFrame:
        """Read in df, remove rows that are filled with UNDEFINEDnull values (in the beginning of each session) and
        remove log messages such as 'aMSG 1068273 !V TRIAL_VAR PARTICIPANT_ASTIGMATISM j'.
        We always need to keep the first letter of the string, since it contains the actual column value."""
        columns = ['RECORDING_SESSION_LABEL', 'SUBJECT_ID', 'SESSION_ID', 'TRIAL_INDEX', 'READING_TRIAL_ID',
                    'TRIAL_LABEL', 'textid', 'SCREEN_ID', 'CURRENT_FIX_X', 'CURRENT_FIX_Y', 'CURRENT_FIX_PUPIL',
                    'CURRENT_FIX_DURATION', 'CURRENT_FIX_INTEREST_AREA_ID', 'CURRENT_FIX_INTEREST_AREA_LABEL',
                    'CURRENT_FIX_INTEREST_AREA_PIXEL_AREA', 'CURRENT_FIX_INTEREST_AREA_RUN_ID',
                    'CURRENT_FIX_INTEREST_AREA_DWELL_TIME', 'PREVIOUS_SAC_DIRECTION', 'PREVIOUS_SAC_AMPLITUDE',
                    'PREVIOUS_SAC_ANGLE', 'PREVIOUS_SAC_AVG_VELOCITY', 'PREVIOUS_SAC_CONTAINS_BLINK',
                    'PREVIOUS_SAC_BLINK_DURATION', 'EYE_USED', 'TRIAL_FIXATION_TOTAL', 'NATIVE_GERMAN',
                    'NATIVE_SWISSGER', 'OTHER_MOTHER_TONGUE', 'PARTICIPANT_AGE', 'PARTICIPANT_ALCOHOL_TODAY',
                    'PARTICIPANT_ALCOHOL_YESTERDAY', 'PARTICIPANT_ASTIGMATISM', 'PARTICIPANT_EYE_SIGHT',
                    'PARTICIPANT_GENDER', 'PARTICIPANT_GLASSES', 'PARTICIPANT_HANDEDNESS',
                    'PARTICIPANT_HOURS_SLEEP', 'PARTICIPANT_KAROLINSKA_SCORE', 'source', 'questiontype',
                    'ACC_Q1', 'ACC_Q2', 'ACC_Q3', 'ACC_Q4', 'ACC_Q5', 'ACC_Q6', 'ACC_Q7', 'ACC_Q8', 'ACC_Q9',
                    'ACC_Q10', 'CURRENT_FIX_START']

        # read file into df
        df = pd.read_csv(file, delimiter='\t', index_col=False, low_memory=False, usecols=columns)

        # filter out NA lines in the beginning of each session and screen 0 (title?)
        mask = (df['textid'] != 'UNDEFINEDnull') | (df['SCREEN_ID'] != 'UNDEFINEDnull')
        df = df[mask]
        # this also helps get rid of one kind of MSG that the loop below doesn't remove (because 29 = 2 digits)
        df = df[df['SCREEN_ID'] != 0]
        df = df[df['SCREEN_ID'] != '0']

        # cleaning: take care of log messages & UNDEFINEDnull values ---------------------------------------------
        # e.g. if there is the log message 'aMSG 1068273 !V TRIAL_VAR PARTICIPANT_ASTIGMATISM j', the correct value
        # would have been 'a' + there is now the value 'UNDEFINEDnull' instead of 'j' in the column 'PARTICIPANT
        # ASTIGMATISM'. this loop writes the correct value in both columns in such cases.
        pattern = r'([^\t]+)\s?MSG\s\d+\s!V\sTRIAL_VAR\s([^\t\n]+)\s?([^\t\n]+)?'
        # blacklist unwanted column names that occur in log messages
        columns_blacklist = ['q3ans1', 'q1', 'textscreen2', 'q5ans3', 'q10ans1', 'q8ans2', 'q1ans1', 'q3ans2', 'q5ans1',
                             'q8ans3', 'q6corrans', 'SUBJECT_ID']

        # replace strings that match pattern 2 with first letter in the string
        # correct subject id if undefinednull
        mask = df['SUBJECT_ID'] == 'UNDEFINEDnull'
        df.loc[mask, 'SUBJECT_ID'] = df.loc[mask, 'RECORDING_SESSION_LABEL'].str[-4:-2]
        # convert all columns to string type

        for col in df.columns:
            for i, cell in enumerate(df[col]):

                if re.match(pattern, str(cell)):
                    # value that actually belongs in cell which contained message
                    column_value = re.sub(pattern, lambda match: match.group(1)[0], str(cell))

                    if column_value == "'":
                        column_value = 'start'  #capturing group didn't catch that start was in quotation marks
                        df.at[i, col] = column_value
                    # trial variable and column which is written in the MSG -> put in correct cell
                    val2 = re.sub(pattern, lambda match: match.group(2), str(cell)).split()[0]
                    val3 = ' '.join(re.sub(pattern, lambda match: match.group(2), str(cell)).split()[1:])

                    if val2 not in columns_blacklist:
                        df[val2] = val3

                else:
                    pass

        # change some messages/undefined values that the loop didn't catch because the regex didn't include it
        mask = df['SESSION_ID'] == '4 MSG 1264737 !V TRIAL_VAR q3ans1 None'
        df.loc[mask, 'SESSION_ID'] = '4'
        mask = df['READING_TRIAL_ID'] == '3 MSG 1102569 !V TRIAL_VAR q1 Durch die Zerschneidung der Landschaft werden Populationen getrennt.'
        df.loc[mask, 'READING_TRIAL_ID'] = '3'
        mask = df['READING_TRIAL_ID'] == '3 MSG 4103170 !V TRIAL_VAR PARTICIPANT_ALCOHOL_TODAY n'
        df.loc[mask, 'READING_TRIAL_ID'] = '3'
        mask = df['READING_TRIAL_ID'] == '1 MSG 4453472 !V TRIAL_VAR q5ans1 None'
        df.loc[mask, 'READING_TRIAL_ID'] = '1'
        mask = df['READING_TRIAL_ID'] == '4 MSG 1685175 !V TRIAL_VAR STIMULUS_COMBINATION_ID 35_3'
        df.loc[mask, 'READING_TRIAL_ID'] = '4'
        mask = df['SCREEN_ID'] == '5 MSG 427086 !V TRIAL_VAR q8ans3 moderne Technik zur Verfügung stehen.'
        df.loc[mask, 'SCREEN_ID'] = '5'

        # stuff the loop didn't cath because the column including the trial message is not exported from DataViewer
        mask = df['RECORDING_SESSION_LABEL'] == 'ET_008_2'
        df.loc[mask, 'PARTICIPANT_ASTIGMATISM'] = 'j'  # from other session
        mask = df['RECORDING_SESSION_LABEL'] == 'ET_002_4'
        df.loc[mask, 'PARTICIPANT_ALCOHOL_YESTERDAY'] = 'j'  # from other session
        # mask = df['RECORDING_SESSION_LABEL'] == 'ET_008_2'
        # df.loc[mask, 'PARTICIPANT_ASTIGMATISM '] = 'j'  # from other trials
        mask = df['RECORDING_SESSION_LABEL'] == 'ET_010_3'
        df.loc[mask, 'PARTICIPANT_AGE'] = 24  # from other session
        mask = df['RECORDING_SESSION_LABEL'] == 'ET_046_3'
        df.loc[mask, 'PARTICIPANT_AGE'] = 22  # from other session
        mask = df['RECORDING_SESSION_LABEL'] == 'ET_042_1'
        df.loc[mask, 'PARTICIPANT_GLASSES'] = 'n'  # from other trials
        mask = df['RECORDING_SESSION_LABEL'] == 'ET_043_1'
        df.loc[mask, 'PARTICIPANT_ASTIGMATISM'] = 'j'  # from other trials
        mask = df['RECORDING_SESSION_LABEL'] == 'ET_035_3'
        df.loc[mask, 'NATIVE_SWISSGER'] = 'n'  # from other session
        mask = df['RECORDING_SESSION_LABEL'] == 'ET_035_3'
        df.loc[mask, 'questiontype'] = 'multiple-choice'  # compared with other participant
        mask = df['RECORDING_SESSION_LABEL'] == 'ET_022_2'
        df.loc[mask, 'PARTICIPANT_EYE_SIGHT'] = 'k'  # from other trials
        mask = df['RECORDING_SESSION_LABEL'] == 'ET_055_1'
        df.loc[mask, 'SCREEN_ID'] = '4' # überprüft bei participapnt 61: text 14, mit mapped words verglichen

        # question scores: NA because they're not relevant (i'm taking scores from results files anyway, not from here)
        mask = df['RECORDING_SESSION_LABEL'] == 'ET_043_1'
        df.loc[mask, 'ACC_Q10'] = 'NA'
        mask = df['RECORDING_SESSION_LABEL'] == 'ET_049_1'
        df.loc[mask, 'ACC_Q5'] = 'NA'
        mask = df['RECORDING_SESSION_LABEL'] == 'ET_054_2'
        df.loc[mask, 'ACC_Q4'] = 'NA'
        mask = df['RECORDING_SESSION_LABEL'] == 'ET_058_3'
        df.loc[mask, 'ACC_Q2'] = 'NA'
        mask = df['RECORDING_SESSION_LABEL'] == 'ET_049_1'
        df.loc[mask, 'ACC_Q5'] = 'NA'
        mask = df['RECORDING_SESSION_LABEL'] == 'ET_049_1'
        df.loc[mask, 'ACC_Q5'] = 'NA'
        mask = df['RECORDING_SESSION_LABEL'] == 'ET_049_1'
        df.loc[mask, 'ACC_Q5'] = 'NA'

        # -------------------------------------------------------------------------
        # replace '.' with NaN
        df = df.replace('.', np.nan)

        # change data types if necessary
        df['textid'] = df['textid'].astype(int)
        df['SUBJECT_ID'] = df['SUBJECT_ID'].astype(int)
        df['SESSION_ID'] = df['SESSION_ID'].astype(int)
        df['READING_TRIAL_ID'] = df['READING_TRIAL_ID'].astype(int)
        df['SCREEN_ID'] = df['SCREEN_ID'].astype(int)
        df['PARTICIPANT_AGE'] = df['PARTICIPANT_AGE'].astype(int)
        df['SESSION_ID'] = df['SESSION_ID'].astype(int)
        df['SESSION_ID'] = df['SESSION_ID'].astype(int)
        df['CURRENT_FIX_INTEREST_AREA_ID'] = df['CURRENT_FIX_INTEREST_AREA_ID'].astype(float)
        df['CURRENT_FIX_INTEREST_AREA_PIXEL_AREA'] = df['CURRENT_FIX_INTEREST_AREA_PIXEL_AREA'].astype(float)
        df['CURRENT_FIX_INTEREST_AREA_DWELL_TIME'] = df['CURRENT_FIX_INTEREST_AREA_DWELL_TIME'].astype(float)
        df['PREVIOUS_SAC_AMPLITUDE'] = df['PREVIOUS_SAC_AMPLITUDE'].astype(float)
        df['PREVIOUS_SAC_ANGLE'] = df['PREVIOUS_SAC_ANGLE'].astype(float)
        df['PREVIOUS_SAC_AVG_VELOCITY'] = df['PREVIOUS_SAC_AVG_VELOCITY'].astype(float)
        df['PREVIOUS_SAC_BLINK_DURATION'] = df['PREVIOUS_SAC_BLINK_DURATION'].astype(float)
        df['CURRENT_FIX_INTEREST_AREA_RUN_ID'] = df['CURRENT_FIX_INTEREST_AREA_RUN_ID'].astype(float)

        # # change column names
        df = df.rename(columns={'SUBJECT_ID': 'subject_id', 'SESSION_ID': 'session_id',
                                'TRIAL_INDEX': 'trial_id', 'SCREEN_ID': 'screen_id', 'textid': 'text_id'})

        # drop all fixations outside of IAS
        df = df.dropna(subset=['CURRENT_FIX_INTEREST_AREA_LABEL'])

        return df

    def map_indico_fixations(self, file) -> pd.DataFrame:
        """Fixations of the raw report are mapped with characters as interest areas.
        Map interest areas on char level to interest areas on word level and add interest area run ID on word level."""
        df = self.clean_indico_fixations(file)

        # group the DataFrame by subject_id, session_id, text_id, and screen_id
        grouped = df.groupby(['subject_id', 'session_id', 'text_id', 'screen_id'])

        # initialize an empty list to store the processed sub-dataframes
        dfs = []

        # iterate through each sub-dataframe and apply the function to it
        for (subject_id, session_id, text_id, screen_id), sub_df in grouped:
            processed_sub_df = get_aois_from_event_data(sub_df, text_id, screen_id)
            dfs.append(processed_sub_df)

        # concatenate the processed sub-dataframes back together into a single DataFrame
        result_df = pd.concat(dfs)

        # replace '.' with NaN
        result_df = result_df.replace('.', np.NaN)

        # drop columns we don't need anymore (character level) -> char, indes of char in screen
        result_df = result_df.drop(columns=['CURRENT_FIX_INTEREST_AREA_LABEL', 'CURRENT_FIX_INTEREST_AREA_ID'])

        # rename new columns
        result_df = result_df.rename(columns={'word_roi_str': 'word',
                                              'word_roi_id': 'word_in_screen_id'})

        # drop all fixations outside of IAS
        result_df = result_df.dropna(subset=['word'])
        # Apply the encoding conversion to all strings in the 'col1' column
        result_df['word'] = result_df['word'].apply(lambda s: s.encode('latin-1').decode('utf-8'))

        result_df['CURRENT_FIX_INTEREST_AREA_RUN_ID'] = result_df.groupby(['subject_id', 'session_id', 'text_id', 'word_in_screen_id']).cumcount() + 1

        return result_df

    def fixfinal_indico(self, file) -> pd.DataFrame:
        """Make some changes to the initial fixation report before feature engineering.
        Same columns as indico data"""
        # Exclude subjects
        # bad quality: 2, 3, 31, 38, 43, 46, 59, 60
        # other blacklisted ID's: no or only one ET session (only include participants who did 2 ET sessions)
        BLACKLIST = [2, 3, 31, 38, 43, 46, 59, 60, 1, 9, 13, 15, 17, 24, 39, 40, 44, 45, 48, 51, 52, 62, 63, 64]


        df = self.map_indico_fixations(file)

        # exclude subjects that don't have 2 sessions/where data quality is extremely bad
        df = df[~df['subject_id'].isin(BLACKLIST)]


        columns_in_order = ['RECORDING_SESSION_LABEL', 'subject_id', 'session_id', 'screen_id', 'text_id','trial_id',
                            "CURRENT_FIX_X", "CURRENT_FIX_Y", "CURRENT_FIX_PUPIL", "CURRENT_FIX_DURATION",
                            "CURRENT_FIX_INTEREST_AREA_DWELL_TIME", "CURRENT_FIX_INTEREST_AREA_PIXEL_AREA",
                            "PREVIOUS_SAC_DIRECTION", "PREVIOUS_SAC_ANGLE", "PREVIOUS_SAC_AMPLITUDE",
                            "PREVIOUS_SAC_AVG_VELOCITY", "PREVIOUS_SAC_CONTAINS_BLINK", "PREVIOUS_SAC_BLINK_DURATION",
                            "word_in_screen_id", "word", "CURRENT_FIX_INTEREST_AREA_RUN_ID"]

        df = df.reindex(columns=columns_in_order)
        # don't sort by w_in_screen_id since we're interested in the sequence of fixations
        df = df.sort_values(['subject_id', 'session_id', 'text_id', 'screen_id'])
        return df

    def fixfinal_sbsat(self) -> pd.DataFrame:
        """Make some changes to the initial fixation report before feature engineering.
        Same columns as indico data"""
        df = pd.read_csv(self.inpath_fixation, delimiter=',')
        # delete rows with type = question
        df = df[~(df['type'] == 'question')]

        # extract subject id from RECORDING_SESSION_LABEL, set session id to 1
        df['subject_id'] = df['RECORDING_SESSION_LABEL'].str.slice(start=-3)
        df['session_id'] = 1  # only one session for sbsat
        # rename book and page variables, as well as CURRENT_FIX_INTEREST_AREA_LABEL
        df = df.rename(columns={'book': 'text_id', 'page': 'screen_id', 'CURRENT_FIX_INTEREST_AREA_LABEL': 'word',
                                'TRIAL_INDEX': 'trial_id', 'CURRENT_FIX_INTEREST_AREA_ID': 'word_in_screen_id'})

        # # drop unnecessary columns
        df = df.drop(columns=['Session_Name_', 'type', 'book_name', 'RT', 'answer', 'correct_answer', 'page_name'])
        # sort columns
        # word_in_screen id needed for merging with lexfeats; corrent_fix_ia_run_id is number of fixations on a word.
        columns_in_order = ['RECORDING_SESSION_LABEL', 'subject_id', 'session_id', 'screen_id', 'text_id', 'trial_id',
                            "CURRENT_FIX_X", "CURRENT_FIX_Y", "CURRENT_FIX_PUPIL", "CURRENT_FIX_DURATION",
                            "CURRENT_FIX_INTEREST_AREA_DWELL_TIME", "CURRENT_FIX_INTEREST_AREA_PIXEL_AREA",
                            "PREVIOUS_SAC_DIRECTION", "PREVIOUS_SAC_ANGLE", "PREVIOUS_SAC_AMPLITUDE",
                            "PREVIOUS_SAC_AVG_VELOCITY", "PREVIOUS_SAC_CONTAINS_BLINK", "PREVIOUS_SAC_BLINK_DURATION",
                            "word_in_screen_id", "word", "CURRENT_FIX_INTEREST_AREA_RUN_ID"]

        df = df.reindex(columns=columns_in_order)

        # correct dtype if necessary
        df['subject_id'] = df['subject_id'].astype(int)
        # drop all fixations outside of IAS
        df = df[df['word'].notna()]

        # drop fixations on 'functional' material (not part of text)
        df = df[df.word != 'GO_TO_QUESTION']
        df = df[df.word != 'next_PAGE']
        df = df[df.word != 'previous_PAGE']  # index -2
        df = df[df.word != '(FIN)']
        # word id -3 because of the functional stuff which has indeces up to 3
        df["word_in_screen_id"] = (df['word_in_screen_id'].apply(lambda x: x - 3)).astype(int)

        # don't sort in s_in_screen_id since we're interested in the sequence of fixations
        # df = df.sort_values(['subject_id', 'session_id', 'text_id', 'screen_id'])

        return df

    # def write_to_csv_individual(self, file, outpath):
    #     """Write cleaned fixations to csv."""
    #     # write the resulting dataframe to CSV
    #     self.map_fixations_individual(file).to_csv(outpath, index=False)
    #
    #     return None


def main():
    # SBSAT: write processed fixation report to csv
    if args.SBSAT is True:
        F = PreprocessedFixations()
        F.data_preprocessed.to_csv(F.outfile)

    elif args.InDiCo is True:
        if args.all_sessions is True:
            F = PreprocessedFixations()
            F.data_preprocessed.to_csv(F.outfile)
            # print(F.map_indico_fixations(F.infile))


        elif args.individual_sessions is True:
            " careful, this only writes cleaned and merged fixations to individual files. doesn't create final dataset"
            F = PreprocessedFixations()
            filenames = F.files
            for file in filenames:

                # get file/path names
                infile_path = os.path.join(F.inpath, file)
                outfile_name = file.rstrip('.txt') + '_with_ia.csv'
                outpath_of_file = os.path.join(F.outpath, outfile_name)
                # preprocess file & write csv
                F.write_to_csv_individual(infile_path, outpath_of_file)


if __name__ == '__main__':
    main()
