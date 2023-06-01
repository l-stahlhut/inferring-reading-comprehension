"""Convert the fixation report that i obtained from the dataviewer.
The original fixation report's interest areas are character level
, this function maps to word level.
Source: https://github.com/hallerp/indiff-preprocessing
"""

import numpy as np
import pandas as pd
from typing import Dict, Collection, List, Tuple
from os.path import exists
import pickle
import os
import logging

def get_word_aoi_dict() -> Dict[int, Dict[int, Tuple[np.array, List[str]]]]:
    word_XY_dict: Dict[int, Dict[int, Tuple[np.array, List[str]]]] = dict()
    for text in range(1, 17):
        word_XY_dict[text] = dict()
        for screenid in range(1, 6):
            roifile = f"data/InDiCo/raw/aoi/text_{text}_{screenid}.ias"
            adf = pd.read_csv(
                roifile,
                delimiter="\t",
                engine="python",
                encoding="utf-8",
                header=None,
            )
            word_XY_l = []
            words_l = []
            characters_in_word = 0
            word = ""
            for row in range(len(adf)):
                if adf.iloc[row][6] != "_":
                    characters_in_word += 1
                    word += adf.iloc[row][6]
                    cur_x_right = adf.iloc[row][4]
                    cur_y_top = adf.iloc[row][3]
                    cur_y_bot = adf.iloc[row][5]
                else:
                    cur_x_left = adf.iloc[row - characters_in_word][2]
                    word_XY_l.append(
                        [[cur_x_left, cur_x_right], [cur_y_top, cur_y_bot]],
                    )
                    words_l.append(word)
                    word = ""
                    characters_in_word = 0
            # after last iteration, store last word (no "-" in text)
            cur_x_left = adf.iloc[len(adf) - characters_in_word][2]
            word_XY_l.append(
                [[cur_x_left, cur_x_right], [cur_y_top, cur_y_bot]],
            )
            words_l.append(word)
            word_XY = np.array(word_XY_l)
            word_XY_dict[text][screenid] = (word_XY, words_l)
    return word_XY_dict  # dimensions: [textid][screenid][[x_left x_right][y_left y_right]]


def get_word_rois(df_subset, word_rois, word_rois_str) -> Tuple[List[str], List[str]]:
    # TODO change 2 lines back
    xs = df_subset.CURRENT_FIX_X.to_numpy(dtype=float)
    ys = df_subset.CURRENT_FIX_Y.to_numpy(dtype=float)
    # xs = df_subset.fix_mean_x.to_numpy(dtype=float)
    # ys = df_subset.fix_mean_y.to_numpy(dtype=float)
    fix_rois: List = ["."] * len(xs)
    fix_rois_str: List = ["."] * len(xs)
    assert len(xs) == len(ys)
    # for each fixation
    for i in range(0, len(xs)):
        found_roi = False
        current_fix_x = xs[i]
        current_fix_y = ys[i]
        # loop through roi dict to find associated roi
        for j in range(len(word_rois)):
            if (
                ((current_fix_x - word_rois[j][0][0]) >= 0)
                & ((current_fix_x - word_rois[j][0][1]) <= 0)
                & ((current_fix_y - word_rois[j][1][0]) >= 0)
                & ((current_fix_y - word_rois[j][1][1]) <= 0)
            ):
                fix_rois[i] = str(j+1)
                fix_rois_str[i] = word_rois_str[j]
                # print(f'found roi {j} for fixation {i}')
                found_roi=True
                break
        if found_roi:
            continue
    return fix_rois, fix_rois_str


def load_aois():
    # data/InDiCo/raw/aoi
    # if exists("aoi/xy_dict.pickle"):
    #     with open("aoi/xy_dict.pickle", "rb") as handle:
    if exists("data/InDiCo/raw/aoi/xy_dict.pickle"):
        with open("data/InDiCo/raw/aoi/xy_dict.pickle", "rb") as handle:
            word_XY_dict = pickle.load(handle)
            return word_XY_dict

    else:
        word_XY_dict = get_word_aoi_dict()
        with open("data/InDiCo/raw/aoi/xy_dict.pickle", "wb") as handle:
            pickle.dump(word_XY_dict, handle)
            return word_XY_dict


def get_aois_from_event_data(event_dat: pd.DataFrame, text_id: int, screen_id: int) -> pd.DataFrame:
    word_roi_dict = load_aois()
    word_rois = word_roi_dict[text_id][screen_id][0]
    word_rois_str = word_roi_dict[text_id][screen_id][1]
    word_roi_ids, word_rois_str = get_word_rois(event_dat, word_rois, word_rois_str)
    event_dat['word_roi_id'] = word_roi_ids
    event_dat['word_roi_str'] = word_rois_str
    return event_dat


def process_file(filename, outpath):
    """Process a file containing one session"""
    print("Working on file ", filename)

    df = pd.read_csv(filename, delimiter='\t')
    # filter out NA lines in the beginning
    mask = (df['textid'] == 'UNDEFINEDnull') & (df['SCREEN_ID'] == 'UNDEFINEDnull')
    df = df[~mask]

    # All texts of 1 session
    splits = dict(tuple(df.groupby('textid')))
    # convert keys to str if they are strings
    splits = {str(k): v for k, v in splits.items()}  # Convert all keys to strings

    processed_subdfs = []  # of one file
    # iterate over text id
    for text_id in range(1, 17):
        if str(text_id) in splits:
            for screen_id in range(1, 6):
                processed_subdf = get_aois_from_event_data(splits[str(text_id)], text_id, screen_id)
                processed_subdfs.append(processed_subdf)

    # concatenate processed sub-dfs
    result_df = pd.concat(processed_subdfs)

    # write the resulting dataframe to CSV
    result_df.to_csv(outpath, index=False)

    return None

def process_seperate_files():
    """Process all the seperate files containing one session each in a certain folder"""
    inpath = r'/Users/laurastahlhut/Documents/UZH/CL/Thesis_ET_ReadingComprehension/data/InDiCo/raw/fixation/all_participants/Output/individual_files_csv'
    outpath = r'/Users/laurastahlhut/Documents/UZH/CL/Thesis_ET_ReadingComprehension/data/InDiCo/interim/fixation/sessionwise_with_ia_mapping'
    filenames = [fn for fn in os.listdir(inpath) if os.path.isfile(os.path.join(inpath, fn)) and fn.endswith('.csv')]
    filenames.sort()

    for file in filenames:
        infile_path = os.path.join(inpath, file)
        outfile_name = file.rstrip('.csv') + '_with_ia.csv'
        outpath_of_file = os.path.join(outpath, outfile_name)

        process_file(infile_path, outpath_of_file)

def process_joined_file():
    """Process a file containing all sessions"""
    infile = "data/InDiCo/raw/fixation/all_participants/Output/indico_fixfinal_all.csv"
    outfile = "data/InDiCo/interim/fixation/all_participants_with_ia_mapping/fixfinal_all_with_ia.csv"
    print("Working on file ", infile)

    df = pd.read_csv(infile, delimiter='\t')
    # filter out NA lines in the beginning
    mask = (df['textid'] == 'UNDEFINEDnull') & (df['SCREEN_ID'] == 'UNDEFINEDnull')
    df = df[~mask]

    # All texts of 1 session
    splits = dict(tuple(df.groupby('textid')))
    # convert keys to str if they are strings
    splits = {str(k): v for k, v in splits.items()}  # Convert all keys to strings

    processed_subdfs = []  # of one file
    # iterate over text id
    for text_id in range(1, 17):
        if str(text_id) in splits:
            for screen_id in range(1, 6):
                processed_subdf = get_aois_from_event_data(splits[str(text_id)], text_id, screen_id)
                processed_subdfs.append(processed_subdf)

    # concatenate processed sub-dfs
    result_df = pd.concat(processed_subdfs)

    # write the resulting dataframe to CSV
    result_df.to_csv(outfile, index=False)

    return None


def main():
    # process the files containing fixations of one session each
    # process_seperate_files()

    # process the file containing all fixations of all sessions
    # process_joined_file()

    print(get_word_aoi_dict()) # encoding is utf8, not latin1

if __name__ == '__main__':
    main()
