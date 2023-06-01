"""Reformat lexical surprisal file from David
More specifically, just add indeces for screens such that I can incorporate the surprisal scores in annotate texts
go by indeces (text, screen, word)."""

import pandas as pd
import os

file_surprisal = 'data/SB-SAT/raw/stimuli/surprisal_screens.csv'

df = pd.read_csv(file_surprisal)

# create new columns based on the 'title' column
# text id
df['text_id'] = df['title'].apply(
    lambda x: '01' if x.split("-")[1] == 'dickens' else '02' if x.split("-")[1] == 'flytrap'
    else '03' if x.split("-")[1] == 'genome' else '04' if x.split("-")[1] == 'northpole' else "NA")
# screen id
df['screen_id'] = df['title'].apply(
    lambda
        x: '1' if x.split("-")[2] == '1' else '2' if x.split("-")[2] == '2'
    else '3' if x.split("-")[2] == '3' else '4' if x.split("-")[2] == '4'
    else '5' if x.split("-")[2] == '5' else '6' if x.split("-")[2] == '6' else "NA")

# create new column with text_screen counter
df["text_screen_id"] = df['text_id'] + '_' + df['screen_id']

# create new column that creates word index in screen
df['w_in_screen_idx'] = df.groupby('text_screen_id').cumcount() + 1

# add sentence index

# Create a new column to hold the sentence index
df['sentence_id'] = 0

# Initialize the sentence index counter and screen ID tracker
sentence_index = 0
prev_screen_id = df['screen_id'][0]

# Iterate over each row in the DataFrame
for i, row in df.iterrows():
    # Check if the screen ID has changed
    if row['screen_id'] != prev_screen_id:
        # Reset the sentence index counter
        sentence_index = 0

    # Check if this is the first word in a new sentence
    if row['token_in_sent'] == 1:
        # Increment the sentence index counter
        sentence_index += 1

    # Update the Sentence Index column for this row
    df.at[i, 'sentence_id'] = sentence_index

    # Update the previous screen ID tracker
    prev_screen_id = row['screen_id']


# write relevant columns to csv
path = 'data/SB-SAT/interim/stimuli/lexical_surprisal'
df = df.drop(columns=['title', 'story', 'sentence_nr', 'text_screen_id'])
# create new column with text_screen counter as tuple to join later
df['text_id'] = df['text_id'].astype(int)
df['screen_id'] = df['screen_id'].astype(int)
df['text_screen_id'] = df.apply(lambda row: (row['text_id'], row['screen_id']), axis=1)

os.makedirs(path, exist_ok=True)

# write to csv
df.to_csv(os.path.join(path, 'lexical_surprisal.csv'), index=False, encoding="utf-8")


