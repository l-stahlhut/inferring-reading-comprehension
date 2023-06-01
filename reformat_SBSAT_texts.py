"""
Reformats raw stimulus texts from SB-SAT such that they are in the same format as raw stimulus texts from InDiCo.
Reason: This way, texts from both corpora can be sent through the same annotation pipeline (annotate_texts.py)
Raw SB-SAT texts: One txt-file that contains all sentences (tsv, formatted like this: title\tsentence_nr\tsentence)
Raw InDiCo texts:
- daf_sentences: One txt-file per text; one sentence per line
- daf_sentences_screens: One txt-file per screen for each text; one sentence per line

Also, I added a file to tokenize the text for further processing
"""

import argparse
import os
import typing
from typing import List, Tuple
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import string

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--reformat", action="store_true", help="Reformating SB-SAT stimulus texts: "
                                                                  "If parameter is given, two new folders are "
                                                                  "created in data/SB-SAT/interim/stimuli that contain "
                                                                  "texts split in sentences for the entire texts and "
                                                                  "the individual screens respectively.")
parser.add_argument("-t", "--tokenize", action="store_true", help="tokenize")
parser.add_argument("-j", "--join", action="store_true", help="join")
args = parser.parse_args()


class ReformatedTexts:
    def __init__(self):
        self.raw_texts_directory: str = "data/SB-SAT/raw/stimuli/texts_sb_sat.txt"
        self.out_directory: str = "data/SB-SAT/interim/stimuli"
        self.sentences_directory = (os.path.join(self.out_directory, 'sbsat_sentences'))
        self.sentences_screens_directory = (os.path.join(self.out_directory, 'sbsat_sentences_screens'))
        self.sentences_screens_corrected_directory = (os.path.join(self.out_directory, 'sbsat_sentences_screens_corr'))
        self.text_files = self.read_file()  # [(text_id, screen_id, sentence_id, sentence), ...]
        #self.frequencies = self.load_frequencies()
        # self.annotations = None

    def create_new_directories(self):
        if not os.path.exists(os.path.join(self.out_directory, 'sbsat_sentences')):
            os.makedirs(os.path.join(self.out_directory, 'sbsat_sentences'))
        if not os.path.exists(os.path.join(self.out_directory, 'sbsat_sentences_screens')):
            os.makedirs(os.path.join(self.out_directory, 'sbsat_sentences_screens'))
        if not os.path.exists(os.path.join(self.out_directory, 'sbsat_sentences_screens_corr')):
            os.makedirs(os.path.join(self.out_directory, 'sbsat_sentences_screens_corr'))

    def read_file(self):
        """Read in raw stimulus file (from David's repo: one txt file for all sentences.
         Return info to write csv files with which are formatted like raw InDiCo stimulus files.
         :param: None
         :returns: List[Tuple[int, int, int, str]], e.g.  [(4, 5, 4, '(FIN)'), ...]
         """
        with open(self.raw_texts_directory, 'r') as f:
            lines = f.readlines()[1:]  # don't read header line
            lines = [l.rstrip() for l in lines]
            lines_data = []
            for l in lines:
                title, sentence_nr, sentence = l.split("\t")
                reading, text_id, screen_id = title.split("-")

                # convert screen_id and text_id to int
                screen_id = int(screen_id)
                if text_id == "dickens":  # 01 = dickens
                    text_id = 1
                elif text_id == "flytrap":  # 02 = dickens
                    text_id = 2
                elif text_id == "genome":  # 03 = genome
                    text_id = 3
                elif text_id == "northpole":  # 04 = northpole
                    text_id = 4
                else:
                    pass

                sentence_id = int(sentence_nr) + 1  # count starts at 0, I want it to start at 1 (same as InDiCo)
                data = (text_id, screen_id, sentence_id, sentence)
                lines_data.append(data)

            return lines_data

    def get_data_for_text(self, text):
        """Return tuples from function above for only one specific text.
        ouput is a list of tuples which each contain (text_id, screen_id, sentence_id, sentence)
        :param text: int (text index)
        :returns List[Tuple[int, int, int, str]], e.g.  [(4, 5, 4, '(FIN)'), ...] but only form one spceific text"""
        sentences = self.read_file()
        texts = [s for s in sentences if s[0] == text][:-1]  # exclude the last item which is (FIN)
        return texts

    def write_sentences_to_file(self):
        """Takes a list of tuples that contains text_id, screen_id, sentence_id and sentence and writes csv files:
        One file per text; one sentence per line. Text id written in file name."""
        with open(os.path.join(self.out_directory, 'sbsat_sentences', "01_sentences.txt"), "w") as outfile1:
            for i in self.get_data_for_text(1):  # 1 = dickens
                outfile1.write(i[3] + "\n")
        with open(os.path.join(self.out_directory, 'sbsat_sentences', "02_sentences.txt"), "w") as outfile2:
            for i in self.get_data_for_text(2):  # 2 = flytrap
                outfile2.write(i[3] + "\n")
        with open(os.path.join(self.out_directory, 'sbsat_sentences', "03_sentences.txt"), "w") as outfile3:
            for i in self.get_data_for_text(3):  # 3 = genome
                outfile3.write(i[3] + "\n")
        with open(os.path.join(self.out_directory, 'sbsat_sentences', "04_sentences.txt"), "w") as outfile4:
            for i in self.get_data_for_text(4):  # 4 = northpole
                outfile4.write(i[3] + "\n")

    def write_sentences_of_one_screen_to_file(self, text_id, screen_id):
        """ Helper fct. to write sentences of all screens to file.
        Takes a list of tuples that contains text_id, screen_id, sentence_id and sentence and writes csv files:
                One file per screen; one sentence per line. Text id & screen id written in file name."""
        outpath = os.path.join(self.out_directory, 'sbsat_sentences_screens')
        # change text_id for correct naming of files
        if text_id == 1:
            text_id_str = "01"
        elif text_id == 2:
            text_id_str = "02"
        elif text_id == 3:
            text_id_str = "03"
        elif text_id == 4:
            text_id_str = "04"

        filename = text_id_str + "_" + str(screen_id) + "_sentences.txt"

        with open(os.path.join(outpath, filename), 'w') as outfile:
            for sent in self.get_data_for_text(text_id):
                if int(sent[1]) == screen_id:
                    outfile.write(sent[3] + "\n")

    def write_sentences_of_all_screens_to_file(self, text_id, screen_id):
        # dickens -> 5 screens
        for i in range(1, 6):
            self.write_sentences_of_one_screen_to_file(1, i)  # 01 = dickens
        # flytrap -> 6 screens
        for i in range(1, 7):
            self.write_sentences_of_one_screen_to_file(2, i)  # 02 = flytrap
        # genome -> 6 screens
        for i in range(1, 7):
            self.write_sentences_of_one_screen_to_file(3, i)  # 03 = genome
        # northpole -> 5 screens
        for i in range(1, 6):
            self.write_sentences_of_one_screen_to_file(4, i)  # 04 = northpole



    def tokenize(self): #todo fix if needed, else delete
        """Tokenize texts, remove punctuation, add word_id, write to csv.
        2 cases: entire texts/screens as input"""
        # tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
        # get file names (1 sentence/line)
        path_screens = os.path.join(self.out_directory, 'sbsat_sentences_screens')  # one file per screen (just created)
        # create new folders
        newpath_screens = os.path.join(self.out_directory, "sbsat_words_screens")
        newpath_screens_idx = os.path.join(self.out_directory, "sbsat_words_screens_idx")
        newpath_texts = os.path.join(self.out_directory, "sbsat_words")
        if not os.path.exists(newpath_screens):
            os.makedirs(newpath_screens)
        if not os.path.exists(newpath_screens_idx):
            os.makedirs(newpath_screens_idx)
        if not os.path.exists(newpath_texts):
            os.makedirs(newpath_texts)

        # tokenize and write new files for screens
        punctuations = list(string.punctuation)
        punctuations.extend(["”", "—"])
        filenames = [f for f in os.listdir(path_screens) if f.endswith('.txt')]
        for file in filenames:
            fn = file.rstrip("sentences.txt")
            with open(os.path.join(path_screens, file), 'r') as infile, \
                    open(os.path.join(newpath_screens_idx, fn + "words_idx.txt"), 'w') as outfile, \
                    open(os.path.join(newpath_screens, fn + "words.txt"), 'w') as outfile2:
                outfile.write("word_id,word\n")
                outfile2.write("word\n")
                sent = infile.read()
                words = [word for word in word_tokenize(sent) if word not in punctuations]  # remove punctuation
                i = 1
                for word in words:
                    outfile.write(str(i) + "," + word + "\n")
                    outfile2.write(word + "\n")
                    i += 1

    def join_tokens_screens():
        path = "data/SB-SAT/interim/stimuli/sbsat_words_screens_idx"
        newpath_words_dickens = 'data/SB-SAT/interim/stimuli/sbsat_words_idx'
        filenames = os.listdir(path)
        if not os.path.exists(newpath_words_dickens):
            os.makedirs(newpath_words_dickens)
        dickens_fn = [f for f in filenames if "01" in f]  # dickens
        flytrap_fn = [f for f in filenames if "02" in f]  # flytrap
        genome_fn = [f for f in filenames if "03" in f]  # genome
        northpole_fn = [f for f in filenames if "04" in f]  # northpole

        with open(os.path.join(newpath_words_dickens, "01_words_idx.txt"), "w") as outfile:  # 01 = dickens
            for fname in dickens_fn:
                with open(os.path.join(path, fname)) as infile:
                    lines = infile.readlines()[1:]  # omit header
                    lines = [l.strip() for l in lines]
                    for i in lines:
                        outfile.write(i + "\n")

        with open(os.path.join(newpath_words_dickens, "02_words_idx.txt"), "w") as outfile:  # 02 = flytrap
            for fname in flytrap_fn:
                with open(os.path.join(path, fname)) as infile:
                    lines = infile.readlines()[1:]  # omit header
                    lines = [l.strip() for l in lines]
                    for i in lines:
                        outfile.write(i + "\n")

        with open(os.path.join(newpath_words_dickens, "03_words_idx.txt"), "w") as outfile:  # 03 = genome
            for fname in genome_fn:
                with open(os.path.join(path, fname)) as infile:
                    lines = infile.readlines()[1:]  # omit header
                    lines = [l.strip() for l in lines]
                    for i in lines:
                        outfile.write(i + "\n")

        with open(os.path.join(newpath_words_dickens, "04_words_idx.txt"), "w") as outfile:  # 04 = northpole
            for fname in northpole_fn:
                with open(os.path.join(path, fname)) as infile:
                    lines = infile.readlines()[1:]  # omit header
                    lines = [l.strip() for l in lines]
                    for i in lines:
                        outfile.write(i + "\n")


def main():
    r = ReformatedTexts()

    print(r.read_file())

    # if args.reformat is True:
    #     print("Starting annotation of English stimulus texts (SB-SAT)...")
    #     r.create_new_directories()
    #     print("Reformating files...")
    #     r.write_sentences_to_file()  # write data with one file per text, one sentence per line
    #     r.write_sentences_of_all_screens_to_file()  # write data with one file per screen, one sentence per line
    #     print("Reformated data can be found in ", r.out_directory)
    # #elif args.tokenize is True:
    #     #r.tokenize()



if __name__ == "__main__":
    raise SystemExit(main())
