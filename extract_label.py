"""
Extract the labels from the data:
Label to predict text comprehension is one score per subject per file
Write to csv.
"""
import pandas as pd
import argparse
import csv
import os

# how to use the code:
# python3 src/extract_label.py --SBSAT
# python3 src/extract_label.py --InDiCo

# indexes: change to subj_ID, book_ID, page_ID

parser = argparse.ArgumentParser(
    prog='annotate_texts',
    description='Annotate Eye Tracking stimulus texts (SB-SAT, InDiCo) with several linguistic '
                'categories (Dependency, lexical frequency and many more).')
parser.add_argument('--SBSAT', action='store_true', help='English version: If argument is given, SB-SAT '
                                                         'labels will be extracted')
parser.add_argument('--InDiCo', action='store_true', help='German version: If argument is given, InDiCo '
                                                          'labels will be extracted')
args = parser.parse_args()


class Labels():
    def __init__(self):
        if args.SBSAT:
            self.file_path = "data/SB-SAT/raw/18sat_fixfinal.csv"  # inpath
            self.labels_path = "data/SB-SAT/interim/labels/sbsat_labels.csv"  # outpath
            self.answers = self.get_answers()
        if args.InDiCo:
            self.file_path = "data/InDiCo/interim/labels/rc_results_indico.csv"
            self.labels_path = "data/InDiCo/interim/labels/indico_labels.csv"
        self.scores = self.get_scores()


    def get_answers(self):
        """
        Get answers per subject, per book, per page (=trial) and correct answer respectively.
        Variables:
        - RECORDING_SESSION_LABEL ==
        """
        # trialfinal = pd.read_csv(self.file_path, delimiter=',', index_col=False)

        if args.SBSAT is True:
            trialfinal = pd.read_csv(self.file_path, delimiter=',', index_col=False)
            # chose relevant columns, aggregate data by page_name column -> result is one line per trial (=page/screen)
            trialfinal = trialfinal[['RECORDING_SESSION_LABEL', 'book', 'page', 'page_name', 'answer', 'correct_answer']].drop_duplicates()
            # filter out rows that contain the word 'reading' in the page_name column
            trialfinal = trialfinal[~trialfinal['page_name'].str.startswith('reading')]
            # filter out rows that contain '-99' in the correct_answer column (= there is no correct answer)
            trialfinal = trialfinal[~(trialfinal['correct_answer'] == -99)]
            # rename columns
            trialfinal = trialfinal.rename(
                columns={"RECORDING_SESSION_LABEL": "subject_id", "book": "text_id", "page": "screen_id"})

            return trialfinal

        else:
            pass

    def get_scores(self):
        """
        Get a df containing the subject_id, book_id, number of questions/book and number of correctly answered
        questions/book
        """
        scores = self.get_answers()
        if args.SBSAT is True:
            # new column checking if score is correct
            scores['answer_is_correct'] = (scores['answer'] == scores['correct_answer']).astype(int)
            # number of possible questions per book. note: in sbsat always 5 questions
            scores['num_questions'] = scores.groupby(['subject_id', 'text_id'])[
                'answer_is_correct'].transform(
                'count')
            # drop unneccessary columns again
            scores = scores[['subject_id', 'text_id', 'answer_is_correct', 'num_questions']]
            # aggregate data by session label and book by adding up scores in answer_is_correct
            scores = scores.groupby(['subject_id', 'text_id', 'num_questions'])['answer_is_correct'].sum().reset_index()
            scores = scores.rename(columns={'answer_is_correct':'n_correct_answers'})
            scores["percentage_correct"] = scores["n_correct_answers"] / scores["num_questions"]
            # add median
            scores["median_score_text"] = scores.groupby('text_id')['n_correct_answers'].transform('median').astype(int)
            # create binary score (higher = 1/lower than medium = 0)
            scores['binary_score'] = scores.apply(lambda x: 1 if x['n_correct_answers'] > x['median_score_text'] else 0, axis=1)

        elif args.InDiCo is True:
            # Load in data
            columns = ['SUBJECT_ID', 'textid', 'SESSION_ID', 'READING_TRIAL_ID', 'ACC_Q1', 'ACC_Q2', 'ACC_Q3', 'ACC_Q4', 'ACC_Q5', 'ACC_Q6',
                       'ACC_Q7', 'ACC_Q8', 'ACC_Q9', 'ACC_Q10']
            scores = pd.read_csv(self.file_path, delimiter=',', index_col=False, usecols=columns)

            # rename columns
            scores = scores.rename(columns={'SUBJECT_ID': 'subject_id', "textid": "text_id",
                                            "READING_TRIAL_ID" : "trial_id", "SESSION_ID" : "session_id"})
            # calculate score
            scores["num_questions"] = 10
            scores["n_correct_answers"] = scores.apply(lambda row: row[4:14].sum(), axis=1)
            scores["percentage_correct"] = scores["n_correct_answers"] / scores["num_questions"]

            # select columns and sort
            scores = scores.loc[:, ['subject_id', 'session_id', 'trial_id', 'text_id', 'num_questions',
                                    'n_correct_answers', 'percentage_correct']].sort_values(['subject_id', 'session_id',
                                                                                             'trial_id'])
            # add median
            scores["median_score_text"] = scores.groupby('text_id')['n_correct_answers'].transform('median').astype(int)
            # create binary score (higher = 1/lower than medium = 0)
            scores['binary_score'] = scores.apply(lambda x: 1 if x['n_correct_answers'] > x['median_score_text'] else 0,
                                                  axis=1)


        return scores

    def write_scores_to_csv(self):
        """Write labels to csv"""
        self.scores.to_csv(self.labels_path, sep='\t', encoding='utf-8')

        return None

def write_indico_results_to_csv():
    """Indico results have to be extracted from results folder of the project, since some values in fix.report have
    UNDEFINEDnull values."""
    directory_path = "/Users/laurastahlhut/Documents/Jobs/HA_Lena/Iva_MA_ET_deploy/results"
    csv_file_path = "data/InDiCo/interim/labels/rc_results_indico.csv"

    # Initialize an empty list to store the dataframes read from each file
    dfs = []

    # Loop through each dir in the folder and if the file exists, read it into a dataframe
    for session in os.listdir(directory_path):
        file_path = os.path.join(directory_path, session, 'RESULTS_READING.txt')

        if os.path.isfile(file_path):

            df = pd.read_csv(file_path, sep='\t')

            # Select the desired columns
            df = df[['Session_Name_', 'SUBJECT_ID', 'SESSION_ID', 'READING_TRIAL_ID', 'textid', 'source', 'ACC_Q1',
                     'ACC_Q2', 'ACC_Q3', 'ACC_Q4', 'ACC_Q5', 'ACC_Q6', 'ACC_Q7', 'ACC_Q8', 'ACC_Q9', 'ACC_Q10']]
            # return df
            dfs.append(df)

    # Concatenate all the dataframes into a single dataframe
    concatenated_df = pd.concat(dfs, ignore_index=True)
    concatenated_df = concatenated_df.sort_values(by=["Session_Name_"])

    # Write the concatenated dataframe to a CSV file
    concatenated_df.to_csv(csv_file_path, index=False)

def main():
    if args.SBSAT is True:
        L = Labels()
        # print(L.answers)
        # print(L.scores)
        L.write_scores_to_csv()
    elif args.InDiCo is True:
        # write csv with results from the individual results file
        # write_indico_results_to_csv() #only needs to be done once.
        L = Labels()
        # print(L.get_scores())
        L.write_scores_to_csv()

if __name__ == '__main__':
    main()
