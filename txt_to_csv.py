import argparse
import os
import csv

import pandas as pd

parser = argparse.ArgumentParser(
    prog='txt_to_csv',
    description='Convert txt-files to csv files')
parser.add_argument('--all_fix', action='store_true', help='Convert txt file that contains all sessions')
parser.add_argument('--individual_fix', action='store_true', help='Convert seperate txt files that each contain one session')
parser.add_argument('--example_fix', action='store_true', help='Convert example txt file that contains two trials')
args = parser.parse_args()

columns = ""

if args.all_fix is True:

    # indico fixation report, all participants
    inpath = '/Users/laurastahlhut/Documents/UZH/CL/Thesis_ET_ReadingComprehension/data/InDiCo/raw/fixation/all_participants/Output/indico_fixfinal_all.txt'
    outpath = '/Users/laurastahlhut/Documents/UZH/CL/Thesis_ET_ReadingComprehension/data/InDiCo/raw/fixation/all_participants/Output/indico_fixfinal_all.csv'

    # with open(inpath, 'r') as f:
    #     data = f.read().replace('\t', ',')
    #     print(data, file=open(outpath, 'w'))

    with open(inpath, 'r') as f1:
        data = [line.strip().split('\t') for line in f1]
        with open(outpath, 'w', newline='') as f2:
            writer = csv.writer(f2, delimiter='\t')
            writer.writerows(data)

elif args.example_fix is True:
# intico fixation report, one participant (?)
    inpath = "/Users/laurastahlhut/Documents/UZH/CL/Thesis_ET_ReadingComprehension/data/InDiCo/raw/fixation/fixationreport1.txt"
    outpath = '/Users/laurastahlhut/Documents/UZH/CL/Thesis_ET_ReadingComprehension/data/InDiCo/raw/fixation/fixationreport1.csv'

    with open(inpath, 'r') as f1:
        data = [line.strip().split('\t') for line in f1]
        with open(outpath, 'w', newline='') as f2:
            writer = csv.writer(f2, delimiter='\t')
            writer.writerows(data)

    # with open(inpath, 'r') as f:
    #     data = f.read().replace('\t', ',')
    #     print(data, file=open(outpath, 'w'))

elif args.individual_fix is True:
    inpath = r'/Users/laurastahlhut/Documents/UZH/CL/Thesis_ET_ReadingComprehension/data/InDiCo/raw/fixation/all_participants/Output/individual_files_txt'
    outpath = r'/Users/laurastahlhut/Documents/UZH/CL/Thesis_ET_ReadingComprehension/data/InDiCo/raw/fixation/all_participants/Output/individual_files_csv'

    filenames = [fn for fn in os.listdir(inpath) if os.path.isfile(os.path.join(inpath, fn))]
    for file in filenames:
        with open(os.path.join(inpath, file), 'r') as f1:
            data = [line.strip().split('\t') for line in f1]
            csv_name = file.rstrip('.txt') + '.csv'
            with open(os.path.join(outpath, csv_name), 'w', newline='') as f2:
                writer = csv.writer(f2, delimiter='\t', quoting=csv.QUOTE_ALL)
                writer.writerows(data)


