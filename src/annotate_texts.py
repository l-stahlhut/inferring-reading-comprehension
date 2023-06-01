"""
Function: Raw stimulus data -> linguistic features
Source: Original code written by Patrick Haller to process InDiCo data (ET&SPR) -> class AnnotatedTexts, class DependencyParser,
class SurprisalScorer.
Adapted by Laura Stahlhut to work for SB-SAT & added features --> Argparse

# SB-SAT file names
# text 01 = dickens
# text 02 = flytrap
# text 03 = genome
# text 04 = northpole

"""

import torch
from transformers import BertTokenizerFast, BertForMaskedLM
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from collections import defaultdict
import glob
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, List, Any
import spacy.attrs
import re
import csv
import pandas as pd
import string
from scipy.special import softmax
from math import log
import os
import argparse
import logging
import math
import wordfreq
from nltk.corpus import wordnet
from germanetpy.germanet import Germanet
import json
import itertools

parser = argparse.ArgumentParser(
    prog='annotate_texts',
    description='Annotate Eye Tracking stimulus texts (SB-SAT, InDiCo) with several linguistic '
                'categories (Dependency, lexical frequency and many more).')
parser.add_argument('--SBSAT', action='store_true', help='English version: If argument is given, SB-SAT '
                                                         'stimulus texts will be annotated.')
parser.add_argument('--InDiCo', action='store_true', help='German version: If argument is given, InDiCo '
                                                          'stimulus texts will be annotated.')
args = parser.parse_args()


class AnnotatedTexts:
    def __init__(self):
        if args.InDiCo is True:
            self.texts_directory: str = "data/InDiCo/raw/stimuli/daf_sentences_screens/text*DAF*"
            self.annotated_texts_path: str = "data/InDiCo/interim/stimuli/annotated_texts/InDiCo_annotated.csv"
            self.nlp = spacy.load("de_dep_news_trf")  # load English model
            self.nlp_lg = spacy.load("de_core_news_lg")  # lg model for semantic similarity (vectors) and entities
            self.genres = {
                1: "scientific",
                2: "scientific",
                3: "informative",
                4: "informative",
                5: "informative",
                6: "informative",
                7: "informative",
                8: "scientific",
                9: "informative",
                10: "scientific",
                11: "informative",
                12: "informative",
                13: "scientific",
                14: "informative",
                15: "informative",
                16: "scientific"
            }
            self.sentences_cut_by_page = []
            self.STOP_CHARS = [
                "!",
                "?",
                ".",
                ";",
                "”",
                "”",
                ":",
                ",",
                "(",
                ")",
                "“",
                ",",
                "„",
                " ",
                "'",
            ]

        elif args.SBSAT is True:
            self.texts_directory: str = "data/SB-SAT/interim/stimuli/sbsat_sentences_screens/*txt"
            self.surprisal_file: str = "data/SB-SAT/interim/stimuli/lexical_surprisal/lexical_surprisal.csv"
            self.annotated_texts_path: str = "data/SB-SAT/interim/stimuli/annotated_texts/SBSAT_annotated.csv"
            self.nlp = spacy.load("en_core_web_trf")  # load English model
            self.nlp_lg = spacy.load("en_core_web_lg")  # lg model for semantic similarity (trf doesn't support vectors)
            self.genres = {1: "fiction",  # dickens
                           2: "scientific",  # flytrap
                           3: "scientific",  # genome
                           4: "fiction"}  # northpole
            self.sentences_cut_by_page = [
                (1, 1), (1, 2), (1, 3),
                (2, 2), (2, 3), (2, 4), (2, 5),
                (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
                (4, 1), (4, 2), (4, 4)
            ]
        self.text_files = self.load_text_screens()
        self.STOP_CHARS = [
            "!",
            "?",
            ".",
            ";",
            "”",
            "”",
            ":",
            ",",
            "(",
            ")",
            "“",
            ",",
            "„",
            " ",
            "'",
        ]
        self.IGNORE = "””“„'"
        self.MOVE = ["(", ")", ",", ".", "?", "!", ":", ";"]
        self.STOP_CHARS_SURP = []  # [";", "”", "”", ":", ",", "(", ")", ",", "„", "'"]
        self.annotations = self.get_annotations()

    def load_text_screens(self):
        """Returns dictionary with (text, screen) as key and text file path as value.
        d = {(1, 1): 'data/InDiCo/raw/stimuli/daf_sentences_screens/text01_1_DAF.txt', ...} """
        text_file_dict = dict()
        text_files = sorted(glob.glob(self.texts_directory))
        for text_file in text_files:
            if args.InDiCo is True:
                text_file_dict[(int(text_file[-12:-10]), int(text_file[-9]))] = text_file
            elif args.SBSAT is True:
                text_file_dict[(int(text_file[-18:-16]), int(text_file[-15]))] = text_file
            else:
                pass

        return text_file_dict

    def get_annotations(self):

        annotations = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        if args.InDiCo is True:
            S = SurprisalScorer(model_name="gpt")  # initializes a SurprisalScorer object with the gpt model
            D = DependencyParser()

        if args.SBSAT is True:
            D = DependencyParser()

            # create dictionary with surprisal scores from the csv in davids repo
            surprisal_scores = {}
            with open(self.surprisal_file, 'r') as csvfile:

                reader = csv.reader(csvfile)
                next(reader)  # skip the header row
                for row in reader:
                    text_id = int(row[3])
                    screen_id = int(row[4])
                    sentence_id = int(row[6])
                    word_id = int(row[0])
                    score = float(row[2])
                    key = (text_id, screen_id, sentence_id)
                    if key not in surprisal_scores:
                        surprisal_scores[key] = {}
                    surprisal_scores[key][word_id] = score

        else:
            pass

        for text_screen_id, sent_file in self.text_files.items():  # (1,1), path

            print(f"working on text {sent_file}")
            text_id, screen_id = text_screen_id

            with open(sent_file) as f:
                sents = f.readlines()
                sents = [sent.strip() for sent in sents]  # list of sentences -> do sentence-level annotation on this

                # get sentence-level annotations from list of sentences:
                sent_cut_by_screen = 1 if text_screen_id in self.sentences_cut_by_page else 0

                # get annotations that were implemented on the text level ------------------------------------------
                text_list = self.get_text_list(text_id)

                # text length: same for each word in text
                (t_n_words, t_n_chars, t_mean_word_length) = self.get_text_len_features(text_list)
                # lexical density & TTR (Token-type-ratio)
                (t_lexical_density, t_lemma_TTR, t_content_w_TTR, t_function_w_TTR, t_nouns_TTR, t_verbs_TTR, t_adj_TTR,
                 t_adv_TTR) = self.get_TTR_densitiy(text_list)
                # genres are hardcoded above
                text_genre = self.genres[text_id]
                # semantic similarity between adjacent sentences
                semantic_similarity_adjacent_sent = self.get_semantic_similarity_sentences(text_list)  # one score/sent

                # get word-level annotations ---------------------------------------------------------------------

                continuous_word_id = -1
                len_prev_sent = 0
                sent_count = -1

                for sent_id, sent in enumerate(sents, 1):
                    sent_count += 1

                    original_range = len(sent.split())  # length of sentence

                    # get word-level annotations from DependencyParser
                    (token_ids, tokens, lemmas, n_char, pos, simplified_pos, lemma_freqs, NE_iob, NE_label,
                     content_word, technical_term, has_more_freq_synonym_or_homonym, morph_tags_Case,
                     morph_tags_Gender, morph_tags_Number, morph_tags_Person, morph_tags_PronType, morph_tags_Mood,
                     morph_tags_Tense, morph_tags_VerbForm, morph_tags_Definite, morph_tags_PunctType,
                     morph_tags_Degree, morph_tags_Abbr, morph_tags_Poss, morph_tags_Prefix, morph_tags_Reflex,
                     synt_surprisal, deps, rights, lefts, n_rights, n_lefts, dep_distance, sentence_length_n_char,
                     sentence_length_n_words, sentence_mean_word_length, sentence_lexical_density_tokens, tf, idf,
                     tf_idf, lex_overlap, pron_det_noun_ratio, voice, no_of_phrases
                    ) = D.parse_dependency(sent, text_id, screen_id, sent_id)


                    if args.InDiCo is True:

                        surprisal = []
                        # tokenize the input sent into a list of words and obtain the probs and offset values
                        # probs = surprisal value for each token returned by the tokenizer;
                        # offset = the start and end character positions of each token in the original sent string.
                        probs, offset = S.score(sent)
                        words = sent.split()

                        if len(words) + 1 == len(probs):
                            probs = np.append(probs[:-2], sum(probs[-2:]))
                            offset = offset[:-2] + [tuple((offset[-2][0], offset[-1][1]))]

                        j = 0  # j index for subword suprisal list
                        # loop through all words
                        for i in range(0, len(words)):  # i index for reference word list
                            # print(f'i={i}, j={j}')
                            # print(f'word={words[i]}')
                            # print(f'sent[offset[j][0]:offset[j][1]]={sent[offset[j][0]:offset[j][1]]}')
                            # print(f'probs[j]={probs[j]}')

                            try:
                                # case 1: tokenized word = reference word in text
                                # print(f'{words[i]} ~ {sent[offset[j][0]:offset[j][1]]}')
                                if words[i] == sent[offset[j][0]: offset[j][1]].strip():

                                    surprisal += [probs[i]]
                                    j += 1

                                # case 2: tokenizer split subword tokens
                                # merge subwords and add up surprisal values until the same
                                elif len(words[i]) >= len(sent[offset[j][0]: offset[j][1]].strip()):
                                    concat_token = sent[offset[j][0]: offset[j][1]].strip()
                                    # print(f'concat_token={concat_token}')
                                    concat_surprisal = probs[j]
                                    while concat_token != words[i]:
                                        j += 1
                                        concat_token += sent[
                                                        offset[j][0]: offset[j][1]
                                                        ].strip()
                                        if (
                                                sent[offset[j][0]: offset[j][1]].strip()
                                                not in self.STOP_CHARS_SURP
                                        ):
                                            # add surprisal value if not punctuation
                                            concat_surprisal += probs[j]
                                        if concat_token == words[i]:
                                            surprisal += [concat_surprisal]
                                            j += 1
                                            break
                                    # print(f'concat_token_processed={concat_token}')

                                else:
                                    break
                                    print(f'concat_token_processed={concat_token}')
                            except IndexError:
                                print(
                                    f"Index error in sentence: {sent}, length: {len(sent)}"
                                )
                                break

                        surprisal = [round(s, 6) for s in surprisal]

                    elif args.SBSAT is True:
                        """Get scores from surprisal.csv provided by David (& reformatted by me)"""
                        surprisal_of_sent = surprisal_scores[(text_id, screen_id, sent_id)]
                        surprisal = list(surprisal_of_sent.values())

                    # assigning dictionary with features to each sentence
                    # (a list per feature -> each word gets a feature)
                    lex_feats = {
                        'text_str': sent,
                        "w_in_sent_id": [token_id + 1 for token_id in token_ids], # start at 1, not 0
                        "word": tokens,
                        "lemma": lemmas,
                        "word_n_char": n_char,
                        "pos": pos,
                        "simplified_pos": simplified_pos,
                        "wordfreq_lemma": lemma_freqs,
                        "NE_IOB": NE_iob,
                        "NE_label": NE_label,
                        "content_word": content_word,
                        "technical_term": technical_term,
                        "synonym_homonym": has_more_freq_synonym_or_homonym,
                        "Case": morph_tags_Case,
                        "Gender": morph_tags_Gender,
                        "Number": morph_tags_Number,
                        "Person": morph_tags_Person,
                        "PronType": morph_tags_PronType,
                        "Mood": morph_tags_Mood,
                        "Tense": morph_tags_Tense,
                        "VerbForm": morph_tags_VerbForm,
                        "Definite": morph_tags_Definite,
                        "PunctType": morph_tags_PunctType,
                        "Degree": morph_tags_Degree,
                        "Abbr": morph_tags_Abbr,
                        "Poss": morph_tags_Poss,
                        "Prefix": morph_tags_Prefix,
                        "Reflex": morph_tags_Reflex,
                        "synt_surprisal": synt_surprisal,
                        "surprisal_bert": surprisal,
                        "deps": deps,
                        "rights": rights,
                        "lefts": lefts,
                        "n_rights": n_rights,
                        "n_lefts": n_lefts,
                        "dep_distance":dep_distance,
                        "sent_n_words": sentence_length_n_words,
                        "sent_n_char": sentence_length_n_char,
                        "sent_mean_word_length": sentence_mean_word_length, # in n_char
                        "sent_lexical_density_tokens": sentence_lexical_density_tokens,
                        "sent_cut": [sent_cut_by_screen for _ in range(original_range)],
                        "t_n_char": [t_n_chars for _ in range(original_range)], # same for all words of sentence
                        "t_n_words": [t_n_words for _ in range(original_range)],
                        "t_n_phrases": no_of_phrases,
                        "t_mean_word_length": [t_mean_word_length for _ in range(original_range)],
                        "t_lexical_density": [t_lexical_density for _ in range(original_range)],
                        "t_lemma_TTR": [t_lemma_TTR for _ in range(original_range)],
                        "t_content_w_TTR": [t_content_w_TTR for _ in range(original_range)],
                        "t_function_w_TTR": [t_function_w_TTR for _ in range(original_range)],
                        "t_nouns_TTR": [t_nouns_TTR for _ in range(original_range)],
                        "t_verbs_TTR": [t_verbs_TTR for _ in range(original_range)],
                        "t_adj_TTR": [t_adj_TTR for _ in range(original_range)],
                        "t_adv_TTR": [t_adv_TTR for _ in range(original_range)],
                        "t_genre": [text_genre for _ in range(original_range)],
                        "tf": tf,
                        "idf": idf,
                        "tf_idf": tf_idf,
                        "lex_overlap": lex_overlap,
                        "semantic_similarity_adjacent_sent": len([semantic_similarity_adjacent_sent[sent_count]
                                                               for _ in range(original_range)]),
                        "pron_det_to_noun_ratio": pron_det_noun_ratio,
                        "voice": voice

                    }

                    annotations[text_id][screen_id][sent_id] = lex_feats

                    #len_prev_sent += len(words)
        return annotations


    def get_text_list(self, text_id: int) -> List[str]:
        """
        Get a list of sentences belonging to a text, obtained by the individual screen files belonging to a text.
        this is needed to build annotations on the level of the entire text (cohesion, coherence: lexical density, ttr,
        tf-idf, lexical overlap, semantic similarity between adjacent sentences).
        input: text id
        output: list of sentences belonging to the text id.
        """
        # get list of files names i.e. screens that belong to a text
        screens_from_text = [f[1] for f in self.text_files.items() if text_id == f[0][0]]

        # get list of sentences from all screens belonging to the text
        sentences_of_text = []
        for screen_file in screens_from_text:
            with open(screen_file, 'r') as infile:
                sentences = infile.readlines()
                sentences = [sent.strip() for sent in sentences]
                for sent in sentences:
                    sentences_of_text.append(sent)

        # create doc object for further operations
        text = ''.join(sentences_of_text)

        return sentences_of_text

    def get_text_len_features(self, text_list):
        """Get the text length in n words and n char and token-type-ratio given a list of sentences as input.
        These features are the same for all words of the text.
        input: list of sentences of a text (List[str]])
        output: list of sentences single score for each metric -> extend to larger list in other document
        """
        # Quantitative indeces ------------------------------------------------
        # text length
        t_n_words = 0
        t_n_chars = 0
        for s in text_list:
            t_n_words += len(s.split())  # incl. spaces
            t_n_chars += len(s.replace(" ", ""))  # incl. punctuation, excl. spaces

        # mean word length in text
        t_mean_word_length = round((t_n_chars / t_n_words), 3)

        return t_n_words, t_n_chars, t_mean_word_length

    def get_TTR_densitiy(self, text_list):
        """Get Token-type ratio for lemmas (of different types) on a text level.
        These features are the same for all words of one text.
        input: List of sentences of a text List[str]
        output: single score for each metric -> convert to list in the larger document"""
        text_str = ''.join(text_list)
        doc = self.nlp(text_str)  # get words with [t.text for t in doc]
        # lexical density: percentage of content words in text -------------------------
        content_word_tags = ["ADJ", "ADV", "NOUN", "PROPN", "NUM", "VERB"]
        text_words = [token.text for idx, token in enumerate(doc) if token.text not in self.STOP_CHARS]
        content_words = [token.lemma_ for idx, token in enumerate(doc) if token.pos_ in content_word_tags]
        t_lexical_density = round(((len(content_words)) / len(text_words)), 4)
        # TTR: token-type ratio -------------------------------------------------------------------------
        # lemma TTR
        lemmas = [token.text for idx, token in enumerate(doc) if token.text not in self.STOP_CHARS]
        t_lemma_TTR = round(len(set(lemmas)) / len(lemmas), 4)
        # content TTR
        t_content_w_TTR = round(len(set(content_words)) / len(content_words), 4)
        # function TTR
        function_words = [token.lemma_ for idx, token in enumerate(doc) if token.pos_ not in content_word_tags]
        t_function_w_TTR = round(len(set(function_words)) / len(function_words), 4)
        # noun TTR
        nouns = [token.lemma_ for idx, token in enumerate(doc) if token.pos_ == "NOUN"]
        t_nouns_TTR = round(len(set(nouns)) / len(nouns), 4)
        # verb TTR
        verbs = [token.lemma_ for idx, token in enumerate(doc) if token.pos_ == "VERB"]
        t_verbs_TTR = round(len(set(verbs)) / len(verbs), 4)
        # adjective TTR
        adj = [token.lemma_ for idx, token in enumerate(doc) if token.pos_ == "ADJ"]
        t_adj_TTR = round(len(set(adj)) / len(adj), 4)
        # adv TTR
        adv = [token.lemma_ for idx, token in enumerate(doc) if token.pos_ == "ADV"]
        t_adv_TTR = round(len(set(adv)) / len(adv), 4)

        return t_lexical_density, t_lemma_TTR, t_content_w_TTR, t_function_w_TTR, t_nouns_TTR, t_verbs_TTR, t_adj_TTR, \
               t_adv_TTR

    def get_simple_word_list(self, text_id):
        text_list_idx = self.get_text_lists_with_index(text_id)
        d_screens = {}
        for screen_id, sentences in text_list_idx.items():
            d_screens[screen_id] = []
            sent_count = -1
            for sent_id, sent in enumerate(sentences, 1):
                sent_count += 1

                # create space before commas
                original_range = len(sent.split())  # length of sentence
                text = re.sub(r"www.vhb.org", r"VHB", sent)

                # for parser, punctuation marks need to be white-space separated
                text = re.sub(r"(?<=[A-Za-z])(?=[(),.?!:;])", r" ", text)
                text = re.sub(r"(?<=[()])(?=[A-Za-z:.,])", r" ", text)
                # remove unwanted characters
                for character in self.IGNORE:
                    text = text.replace(character, "")

                # construct spacy doc object from sentence
                doc = self.nlp(text)  # get words with [t.text for t in doc]
                # tokens = [token.text for idx, token in enumerate(doc)]
                tokens = []

                for idx, token in enumerate(doc):
                    if token.text not in self.MOVE:
                        tokens.append(token.text)
                # return tokens

                d_screens[screen_id].append(tokens)
        return d_screens

        return text_list_idx

    def get_sentence_pairs(self, text_list):
        """Helper function for lexical overlap and semantic similarity.
        Input: list of sentences of a text.
        Output: List of tuples with preprocessed sentences.
        """
        # get list of tuples List(Tuple(str, str)) of the sentence and subsequent sentence for each sentence in the text
        sentence_pairs = []

        # iterate over sentences
        for i, text in enumerate(text_list):  # for sentence in text ...

            # preprocessing: same as in dependencyparser
            original_range = len(text.split())  # length of sentence
            text = re.sub(r"www.vhb.org", r"VHB", text)
            text = re.sub(r"(?<=[A-Za-z])(?=[(),.?!:;])", r" ", text)
            text = re.sub(r"(?<=[()])(?=[A-Za-z:.,])", r" ", text)
            for character in self.IGNORE:
                text = text.replace(character, "")

            # last sentence -> NA as second tuple element
            if i == len(text_list) - 1:
                sentence_pairs.append((text, "NA"))
            # not last sentence: adjacent sentence als second tuple element
            else:
                sentence_pairs.append((text, text_list[i + 1]))

        return sentence_pairs

    def get_lexical_overlap_old(self, text_list):
        """Does the lemma appear in the next sentence? (0, 1)
        Same preprocessing as in dependency parser to get the same list length"""
        # get list of tuples List(Tuple(str, str)) of the sentence and subsequent sentence for each sentence in the text
        sentence_pairs = self.get_sentence_pairs(text_list)

        # calculate lexical overlap in adjacent sentences
        t_lexical_overlap_adjacent_sent = []
        for i, sent_pair in enumerate(sentence_pairs):

            if i == len(text_list) - 1:  # last item
                # get list of sentence lemmas
                s1_lemmas = [token.lemma_ for idx, token in enumerate(self.nlp(sent_pair[0])) if
                             token.text not in self.STOP_CHARS]
                t_lexical_overlap_adjacent_sent.append([0 for i in s1_lemmas])  # last sentence: no subsequent sentence

            else:
                # get list of sentence lemmas
                s1_lemmas = [token.lemma_ for idx, token in enumerate(self.nlp(sent_pair[0])) if
                             token.text not in self.STOP_CHARS]
                s2_lemmas = [token.lemma_ for idx, token in enumerate(self.nlp(sent_pair[1])) if
                             token.text not in self.STOP_CHARS]

                # iterate over tokens of sent, see if there is any overlap with next sentence
                token_overlap = []
                for lemma in s1_lemmas:
                    if lemma in s2_lemmas:
                        token_overlap.append(1)
                    else:
                        token_overlap.append(0)
                t_lexical_overlap_adjacent_sent.append(token_overlap)

        return t_lexical_overlap_adjacent_sent

    def get_semantic_similarity_sentences(self, text_list):
        """Semantic similarity between each sentence and the following sentence.
        input: list of sentences
        output: list of scores (one score per sentence)
        This is calculated with the lg model since the trf doesn't have vector representations.

        """
        # self.nlp_lg
        # get list of tuples List(Tuple(str, str)) of the sentence and subsequent sentence for each sentence in the text
        sentence_pairs = self.get_sentence_pairs(text_list)

        # calculate semantic similarity for each sentence pair
        semantic_similarity_adjacent_sent = []
        for i, sent_pair in enumerate(sentence_pairs):

            if i == len(text_list) - 1:
                semantic_similarity_adjacent_sent.append(0)  # last sentence: no subsequent sentence
            else:
                # use lg model since trf doesn't have vector representations
                similarity = round(self.nlp_lg(sent_pair[0]).similarity(self.nlp_lg(sent_pair[1])), 4)
                semantic_similarity_adjacent_sent.append(similarity)

        return semantic_similarity_adjacent_sent

    def write_annotations_to_csv(self):
        if not self.annotations:
            print("No annotations available")
            return -1
        all_df = []
        for text in self.annotations:
            for screen in self.annotations[text]:
                text_df = []
                for sentence in self.annotations[text][screen]:
                    temp = pd.DataFrame.from_dict(
                        self.annotations[text][screen][sentence]
                    )
                    temp.insert(0, "sentence_number", sentence)
                    temp.insert(0, "screen_number", screen)
                    temp.insert(0, "text_number", text)
                    text_df.append(temp)
                all_df.append(pd.concat(text_df))
        all_df = pd.concat(all_df)
        all_df.reset_index(drop=True, inplace=True)

        # add column to index word in screen
        # group by screen_id and add a new column with word count within each screen
        # all_df['word_in_screen_id'] = all_df.groupby('screen_number').cumcount() + 1
        all_df['word_in_screen_id'] = all_df.groupby(['text_number', 'screen_number']).cumcount() + 1
        # all_df.to_csv(outfile)
        all_df.to_csv(self.annotated_texts_path)


class DependencyParser:
    """Tokenizes a sentence and returns a Tuple with a multitude of tags on token and sentence level."""

    def __init__(self):
        # self.nlp = spacy.load("de_core_news_lg")
        if args.SBSAT is True:
            self.language = "en"  # for wordfreq score
            self.nlp = spacy.load("en_core_web_trf")  # load English model
            self.texts_directory: str = "data/SB-SAT/interim/stimuli/sbsat_sentences_screens/*txt"
            self.full_texts_directory: str = "data/SB-SAT/interim/stimuli/sbsat_sentences/*txt"
            self.special_terms = ["reales", "reale", "Dionea muscipula", "Dionea", "muscipula",
                                  "action potential", "action", "potential", "calcium ions", "Flytrap", "Venus",
                                  "lobe", "electric charge", "electric", "charge", "cell membrane", "cell",
                                  "membrane", "coupling", "dissipation", "electrode", "electrical current",
                                  "electrical", "electric", "current", "mocrocoulombs", "genome", "genetic",
                                  "genetical", "engineer", "engineered", "pharming", "transgenic", "enzyme",
                                  "hormone", "clotting", "antibody", "industrial", "scale", "antithrombin",
                                  "anticoagulant", "molecular", "proof", "microinjection", "promoter", "mammary",
                                  "gland", "et", "voilà", "velocity", "expedition", "mathematical", "mathematics",
                                  "machinery"]
            self.mapping_file = "data/SB-SAT/interim/stimuli/sbsat_sentence_mapping.json"
        elif args.InDiCo is True:
            self.language = "de"
            self.nlp = spacy.load("de_dep_news_trf")  # load German model
            self.nlp_lg = spacy.load("de_core_news_lg")
            self.texts_directory: str = "data/InDiCo/raw/stimuli/daf_sentences_screens/text*DAF*"
            self.full_texts_directory: str = "data/InDiCo/raw/stimuli/daf_sentences/text*DAF*"
            # self.nlp_entities = spacy.load("de_core_news_lg") # for NE only, since trf doesn't support NE recognition!
            self.gn_path = Germanet(
                'data/InDiCo/external/GermaNet/v16.0/GN_V160_XML')
            # technical terms, foreign words, loan words
            self.special_terms = ['treibhauseffekt', 'symposium', 'gebirgsforscher', 'quadratkilometer', 'überwärmung',
                                  'klimafaktoren', 'gletscherschwund', 'pleistozän', 'temperatur', 'januartemperatur',
                                  'vegetationsverhältnisse', 'industrialisierung', 'neolithikum', 'botaniker',
                                  'temperaturunterschiede', 'durchschnittstemperatur', 'kohlendioxid', 'prognosen',
                                  'tagungsleiter', 'längsschnittstudie', 'entwicklungsstand', 'beobachtungsinstrument',
                                  'bindungssicherheit', 'ausgewogenheit', 'interrater-reliabilität', 'interrater',
                                  'reliabilität', 'ausprägungsskala', 'merkmal', 'variable', 'persönlichkeitsvariable',
                                  'stichprobe', 'präsenz', 'spielsituation', 'familiensituation', 'korrelation',
                                  'ernährungsphysiologisch', 'fettsäure', 'nährwert', 'fettstoffwechsel',
                                  'schalenfrucht', 'schalenfrüchte', 'aflatoxin', 'aflotoxine', 'grenzwert',
                                  'evolutionsbiologe', 'charakteristikum', 'nützlichkeitserwägung', 'gesteinsprobe',
                                  'datenstrom', 'systematik', 'speicherkapazität', 'botenstoff', 'these', 'lernprozess',
                                  'sozialpsychologie', 'geophysiker', 'eurasisch', 'platte', 'ebbe', 'flut',
                                  'seismisch', 'welle', 'richterskala', 'metropole', 'feuergürtel', 'gummipuffer',
                                  'evakuierungsplan', 'bautechnisch', 'forschungsgemeinschaft', 'riechvermögen',
                                  'geruchsinn', 'geschmacksinn', 'tonfrequenz', 'testmethode', 'kognitiv',
                                  'geruchsgedächtnis', 'testmethode', 'tiermodell', 'diskriminierungsaufgabe',
                                  'totenkopfaffe', 'virtuell', 'konzept', 'fernstudium', 'telekooperation',
                                  'multimedia-techniken', 'multimedia', 'immatrikuliert', 'testsicherheit',
                                  'kommerzialisierung', 'evaluierung', 'chronobiologie', 'organismus', 'astronomisch',
                                  'chronobiologisch', 'versuchsbedingung', 'endogen', 'aktivitätsrhythmus', 'circadian',
                                  'chronotypen', 'chronotyp', 'eule', 'lerche', 'säuregehalt', 'physikalisch', 'befund',
                                  'stoffwechselvorgang', 'pflanzenphysiologe', 'kohlendioxidverbrauch', 'parasit',
                                  'rezeptor', 'schall', 'empirisch', 'universalgrammatik', 'protosprache',
                                  'syntaktisch',
                                  'informationsverarbeitung', 'hirnmasse', 'intelligenzsprung', 'bündnispolitik',
                                  'katogisierung', 'wissenserwerb', 'fazit', 'methan', 'kohlendioxid', 'co2',
                                  'emission',
                                  'abgasreinigung', 'kernspaltung', 'kernfusion', 'geothermisch', 'regenerativ',
                                  'energiequelle', 'gedächtnisforscher', 'gedächtnisleistung', 'langzeitgedächtnis',
                                  'kurzzeitgedächtnis', 'speicherkapazität', 'versuchsleiter', 'testteilnehmer',
                                  'gehirntraining', 'experiment', 'artenreichtum']
        self.IGNORE = "””“„'"
        self.MOVE = ["(", ")", ",", ".", "?", "!", ":", ";"]
        self.STOP_CHARS = [
            "!",
            "?",
            ".",
            ";",
            "”",
            "”",
            ":",
            ",",
            "(",
            ")",
            "“",
            ",",
            "„",
            " ",
            "'",
        ]
        self.nlp.tokenizer = spacy.tokenizer.Tokenizer(
            self.nlp.vocab,
            token_match=re.compile(r"\S+").match,
        )
        self.tagger = self.nlp.get_pipe("tagger")
        self.text_files = self.load_text_screens()

    def parse_dependency(self, text, text_id, screen_id, sent_id) -> Tuple[List[Any], ...]:
        """Takes a sentence as input and returns annotations for the sentence (list of tags per word for each category).
         Some sentences from the SB-SAT dataset are cut off at the end of the screen. For the annotation, I replace
         the cut off sentences by the full sentence stored in a mapping file and annotate the full sentence. Then, I
         just return the tags for the cut off sentence."""

        # get mapping dictionary for SB-SAT texts that were cut off by the end of the screen ----------------------
        if args.SBSAT is True:
            cut_sentences_mapping = []
            with open(self.mapping_file, 'r') as file:
                data = json.load(file)
                for i in data:
                    cut_sentences_mapping.append((i["sentence_1"], i["joined_sentence"], "s1"))
                    cut_sentences_mapping.append((i["sentence_2"], i["joined_sentence"], "s2"))

            all_cut_off_sentences = [i[0] for i in cut_sentences_mapping]  # only cut off sentences

            # if sentence is in list of cut off sentences, change text variable to full sentence (ready for annotation)
            # and save the split sentence to another variable (we need the range of the parsed original sentence to
            # output correct tags at the end)
            if text in all_cut_off_sentences:
                # text_id = text_id-1
                for i in cut_sentences_mapping:
                    if i[0] == text:
                        # case 1: cut off at the end of the page = first part of the full sentence
                        if i[2] == "s1":
                            text = i[1]  # save full sentence to text variable for tagging
                            sentence1 = i[0]  # cut off sentence, saved to variable sentence 1
                            doc_sent1 = self.nlp(sentence1)
                            split_sent_range1 = len(doc_sent1)

                        elif i[2] == "s2":
                            text = i[1]  # full sentence
                            # sentence1 = ""
                            sentence2 = i[0]  # cut off sentence, saved to variable sentence 1
                            doc_sent2 = self.nlp(sentence2)
                            split_sent_range2 = len(doc_sent2)

            # if sentence is not in list of cut off sentences, leave text variable as it is.
            else:
                pass
        # ------------------------------------ parse and tag sentence ------------------------------------------------

        # preprocessing text that will be parsed
        # create space before commas
        original_range = len(text.split())  # length of sentence

        # preprocess sentence
        text = self.sentence_preprocessing(text)
        # construct spacy doc object from sentence
        doc = self.nlp(text)  # get words with [t.text for t in doc]

        pos_pred = self.tagger.model.predict([doc])

        # Get stuff for idf score with full texts the sentence is from -----------------------------------------------
        # stuff needed for idf score: dictionary that contains sentence count for each word
        # get list of sentences
        with open(self.load_full_texts()[text_id], "r") as infile_full:

            sents_full = infile_full.readlines()
            sents_full = [sent.strip() for sent in sents_full]  # list of all sentences that occur in a text

            # sents_full_with_index =

        # create a list containing the set of lemmas for each text
        lemmas_sentences = []
        for sentence_full in sents_full:
            # same preprocessing as above
            sentence_full = self.sentence_preprocessing(sentence_full)

            doc_full = self.nlp(sentence_full)
            # get list of set of lemmas per sentence
            sent_lemmas = list(set([token.lemma_ for token in doc_full]))
            lemmas_sentences.append(sent_lemmas)

        # create dictionary that contains sentence count for each word
        sent_counts = {}
        for sent in lemmas_sentences:
            for lemma in sent:
                if lemma in sent_counts:
                    sent_counts[lemma] += 1
                else:
                    sent_counts[lemma] = 1

        #lexical overlap for screen -> get next sentence
        next_sentence = next((pair[1] for pair in self.get_sentence_pairs(text_id) if pair[0] == text), None)
        if next_sentence is not None:
            next_sentence_doc = self.nlp(next_sentence)
            next_sentence = [token.lemma_ for token in next_sentence_doc]

        # ohter: active or passive voice? Num phrases etc.
        sentence_voice = self.voice_detector(doc)
        num_phrases = len(list(doc.noun_chunks))

        # initiate lists for tags for each word -----------------------------------------------------------------

        # General annotations on the word level:
        token_ids = [[] for _ in range(original_range)]
        tokens = [[] for _ in range(original_range)]
        lemmas = [[] for _ in range(original_range)]
        n_char = [[] for _ in range(original_range)]
        pos = [[] for _ in range(original_range)]
        simplified_pos = [[] for _ in range(original_range)]
        lemma_freqs = [[] for _ in range(original_range)]  # word frequency, calculated with wordfreq library
        if args.SBSAT is True:
            NE_iob = [[] for _ in range(original_range)]  # Named Entity, IOB scheme (inside, outside, beginning)
            NE_label = [[] for _ in range(original_range)]  # Named Entity,label
        content_word = [[] for _ in range(original_range)]  # Boolean
        technical_term = [[] for _ in range(original_range)]  # Technical term/ foreign word / old, outdated word
        has_more_freq_synonym_or_homonym = [[] for _ in range(original_range)]  # Boolean

        # morphological annotations
        morph_tags = [[] for _ in range(original_range)]
        morph_tags_Case = [[] for _ in range(original_range)]
        morph_tags_Gender = [[] for _ in range(original_range)]
        morph_tags_Number = [[] for _ in range(original_range)]
        morph_tags_Person = [[] for _ in range(original_range)]
        morph_tags_PronType = [[] for _ in range(original_range)]
        morph_tags_Mood = [[] for _ in range(original_range)]
        morph_tags_Tense = [[] for _ in range(original_range)]
        morph_tags_VerbForm = [[] for _ in range(original_range)]
        morph_tags_Definite = [[] for _ in range(original_range)]
        morph_tags_Degree = [[] for _ in range(original_range)]
        morph_tags_Abbr = [[] for _ in range(original_range)]
        morph_tags_Poss = [[] for _ in range(original_range)]
        morph_tags_Prefix = [[] for _ in range(original_range)]
        morph_tags_Reflex = [[] for _ in range(original_range)]
        morph_tags_PunctType = [[] for _ in range(original_range)]

        # syntax
        synt_surprisal = [[] for _ in range(original_range)]
        n_rights = [[] for _ in range(original_range)]
        n_lefts = [[] for _ in range(original_range)]
        rights = [[] for _ in range(original_range)]
        lefts = [[] for _ in range(original_range)]
        deps = [[] for _ in range(original_range)]
        dep_distance = [[] for _ in range(original_range)]

        # tf-idf, lexical overlap
        tf = [[] for _ in range(original_range)]
        idf = [[] for _ in range(original_range)]
        tf_idf = [[] for _ in range(original_range)]
        lex_overlap = [[] for _ in range(original_range)]

        #other
        voice = [sentence_voice for _ in range(original_range)]
        no_of_phrases = [num_phrases for _ in range(original_range)]


        # iterate over words in text except for "forbidden" words (then don't include and move index)
        adjusted_id = -1
        # get annotations for each token in the sentence
        for idx, token in enumerate(doc):
            if token.text not in self.MOVE:
                adjusted_id += 1
                # token-level tags (length, freq, terms,...)
                token_ids[adjusted_id] = adjusted_id  # word_id
                tokens[adjusted_id] = token.text  # original word
                lemmas[adjusted_id] = token.lemma_
                n_char[adjusted_id] = len(token)  # token length in n char
                pos[adjusted_id] = token.pos_  # UPOS tags (coarse grained pos tags in spaCy)
                simplified_pos[adjusted_id] = self.get_simplified_pos(token)  # A, N, V, FUNC
                content_word[adjusted_id] = token.pos_ in ["ADJ", "ADV", "NOUN", "PROPN", "VERB"]
                lemma_freqs[adjusted_id] = round(wordfreq.zipf_frequency(token.lemma_, self.language, "large"), 6)

                # NE only for EN with this model; for German further down below with different model.
                if args.SBSAT is True:
                    NE_iob[adjusted_id] = token.ent_iob_  # ent_iob_, doc[1].ent_type_]
                    NE_label[adjusted_id] = token.ent_type_ if token.ent_type_ != "" else "NA"

                technical_term[adjusted_id] = token.lemma_.lower() in self.special_terms  # True lemma is in list
                has_more_freq_synonym_or_homonym[adjusted_id] = self.get_synonyms_homonyms(token.lemma_)

                # morphology
                morph_tags[adjusted_id] = self.get_morph_tags(token.morph.to_dict())  # dictionary with possible tags
                morph_tags_Case[adjusted_id] = morph_tags[adjusted_id]['Case']
                morph_tags_Gender[adjusted_id] = morph_tags[adjusted_id]['Gender']
                morph_tags_Number[adjusted_id] = morph_tags[adjusted_id]['Number']
                morph_tags_Person[adjusted_id] = morph_tags[adjusted_id]['Person']
                morph_tags_PronType[adjusted_id] = morph_tags[adjusted_id]['PronType']
                morph_tags_Mood[adjusted_id] = morph_tags[adjusted_id]['Mood']
                morph_tags_Tense[adjusted_id] = morph_tags[adjusted_id]['Tense']
                morph_tags_VerbForm[adjusted_id] = morph_tags[adjusted_id]['VerbForm']
                morph_tags_Definite[adjusted_id] = morph_tags[adjusted_id]['Definite']
                morph_tags_PunctType[adjusted_id] = morph_tags[adjusted_id]['PunctType']
                morph_tags_Degree[adjusted_id] = morph_tags[adjusted_id]['Degree']
                morph_tags_Abbr[adjusted_id] = morph_tags[adjusted_id]['Abbr']
                morph_tags_Poss[adjusted_id] = morph_tags[adjusted_id]['Poss']
                morph_tags_Prefix[adjusted_id] = morph_tags[adjusted_id]['Prefix']
                morph_tags_Reflex[adjusted_id] = morph_tags[adjusted_id]['Reflex']

                # dependencies
                synt_surprisal[adjusted_id] = round(-log(max(softmax(pos_pred[0][idx]))), 5)
                deps[adjusted_id] = token.dep_
                rights[adjusted_id] = [str(item) for item in token.rights]
                lefts[adjusted_id] = [str(item) for item in token.lefts]
                n_rights[adjusted_id] = token.n_rights
                n_lefts[adjusted_id] = token.n_lefts
                dep_distance[adjusted_id] = token.i - token.head.i

                # tf-idf
                # TF (lemma): Number of times a word appears in the sentence/total number of words in the sentence
                tf[adjusted_id] = round((self.get_tf_score(doc, token.lemma_) / original_range), 4)
                # IDF (lemma): log of (number of sentences in the text/ number of sentences containing the term)
                idf[adjusted_id] = round(math.log(len(sents_full) / sent_counts[token.lemma_]), 4)
                # TF-IDF = TF*IDF
                tf_idf[adjusted_id] = round(tf[adjusted_id]*idf[adjusted_id], 4)
                #lexical overlap
                lex_overlap[adjusted_id] = 0 if next_sentence is None else 1 if token.lemma_ in next_sentence else 0
                # other
                # pron_det_noun_ratio = pron_det_to_noun_ratio[adjusted_id]
                # pron_det_ambiguity[adjusted_id] = pronoun_det_to_noun_ratio[adjusted_id]
                # pron_det_ambiguity[adjusted_id] = self.check_pronoun_det_ambiguity(text_id, text)[idx]

        # pronoun noun ratio
        pron_det_to_noun_ratio = self.pronoun_noun_ratio(text_id, text, pos)

        # seperately add lists for NE in German since I had to do that with another model
        if args.InDiCo is True:
            NE_iob = self.get_NE_tags_DE(text, text_id, screen_id)[0]
            NE_label = self.get_NE_tags_DE(text, text_id, screen_id)[1]

        # --------------------------------------- return tags --------------------------------------------------
        # ---------------------3 different scenarios depending on whether the sentence was split----------------

        # case 1: sentence was split in the end of the page (= return first part of sentence)
        if "sentence1" in locals():
            # quantitative indeces on sentence level
            sentence_length_n_char = [len(text) for _ in range(len(doc_sent1))]  # incl. spaces
            sentence_length_n_words = [len(text.split()) for _ in range(len(doc_sent1))]  # incl. punctuation
            sentence_mean_word_length = [sum(n_char) / len(text.split()) for _ in range(len(doc_sent1))]  # in n char
            # percentage of pos tags that are content word tags:
            sentence_lexical_density_tokens = [round((len([p for p in pos
                                                           if p in ['ADJ', 'ADV', 'VERB', 'PROPN', 'NOUN']]) / len(
                pos)), 4)
                                               for _ in range(len(doc_sent1))]

            return token_ids[:len(doc_sent1)], tokens[:len(doc_sent1)], lemmas[:len(doc_sent1)], \
                   n_char[:len(doc_sent1)], pos[:len(doc_sent1)], simplified_pos[:len(doc_sent1)], \
                   lemma_freqs[:len(doc_sent1)], NE_iob[:len(doc_sent1)], NE_label[:len(doc_sent1)], \
                   content_word[:len(doc_sent1)], technical_term[:len(doc_sent1)], \
                   has_more_freq_synonym_or_homonym[:len(doc_sent1)], morph_tags_Case[:len(doc_sent1)], \
                   morph_tags_Gender[:len(doc_sent1)], morph_tags_Number[:len(doc_sent1)], \
                   morph_tags_Person[:len(doc_sent1)], morph_tags_PronType[:len(doc_sent1)], \
                   morph_tags_Mood[:len(doc_sent1)], morph_tags_Tense[:len(doc_sent1)], \
                   morph_tags_VerbForm[:len(doc_sent1)], morph_tags_Definite[:len(doc_sent1)], \
                   morph_tags_PunctType[:len(doc_sent1)], morph_tags_Degree[:len(doc_sent1)], \
                   morph_tags_Abbr[:len(doc_sent1)], morph_tags_Poss[:len(doc_sent1)], \
                   morph_tags_Prefix[:len(doc_sent1)], morph_tags_Reflex[:len(doc_sent1)], \
                   synt_surprisal[:len(doc_sent1)], deps[:len(doc_sent1)], rights[:len(doc_sent1)], \
                   lefts[:len(doc_sent1)], n_rights[:len(doc_sent1)], n_lefts[:len(doc_sent1)], \
                   dep_distance[:len(doc_sent1)], sentence_length_n_char, sentence_length_n_words, \
                   sentence_mean_word_length, sentence_lexical_density_tokens, tf[:len(doc_sent1)], \
                   idf[:len(doc_sent1)], tf_idf[:len(doc_sent1)], lex_overlap[:len(doc_sent1)], \
                   pron_det_to_noun_ratio[:len(doc_sent1)], voice[:len(doc_sent1)], no_of_phrases[:len(doc_sent1)]

        # case 2: sentence was split in the beginning of the page (= return second part of sentence)
        elif "sentence2" in locals():
            # quantitative indeces on sentence level
            sentence_length_n_char = [len(text) for _ in range(len(doc_sent2))]  # incl. spaces
            sentence_length_n_words = [len(text.split()) for _ in range(len(doc_sent2))]  # incl. punctuation
            sentence_mean_word_length = [sum(n_char) / len(text.split()) for _ in range(len(doc_sent2))]  # in n char
            # percentage of pos tags that are content word tags:
            sentence_lexical_density_tokens = [len([p for p in pos
                                                    if p in ['ADJ', 'ADV', 'VERB', 'PROPN', 'NOUN']]) / len(pos)
                                               for _ in range(len(doc_sent2))]

            return token_ids[-len(doc_sent2):], tokens[-len(doc_sent2):], lemmas[-len(doc_sent2):], \
                   n_char[-len(doc_sent2):], pos[-len(doc_sent2):], simplified_pos[-len(doc_sent2):], \
                   lemma_freqs[-len(doc_sent2):], NE_iob[-len(doc_sent2):], NE_label[-len(doc_sent2):], \
                   content_word[-len(doc_sent2):], technical_term[-len(doc_sent2):], \
                   has_more_freq_synonym_or_homonym[-len(doc_sent2):], morph_tags_Case[-len(doc_sent2):], \
                   morph_tags_Gender[-len(doc_sent2):], morph_tags_Number[-len(doc_sent2):], \
                   morph_tags_Person[-len(doc_sent2):], morph_tags_PronType[-len(doc_sent2):], \
                   morph_tags_Mood[-len(doc_sent2):], morph_tags_Tense[-len(doc_sent2):], \
                   morph_tags_VerbForm[-len(doc_sent2):], morph_tags_Definite[-len(doc_sent2):], \
                   morph_tags_PunctType[-len(doc_sent2):], morph_tags_Degree[-len(doc_sent2):], \
                   morph_tags_Abbr[-len(doc_sent2):], morph_tags_Poss[-len(doc_sent2):], \
                   morph_tags_Prefix[-len(doc_sent2):], morph_tags_Reflex[-len(doc_sent2):], \
                   synt_surprisal[-len(doc_sent2):], deps[-len(doc_sent2):], rights[-len(doc_sent2):], \
                   lefts[-len(doc_sent2):], n_rights[-len(doc_sent2):], n_lefts[-len(doc_sent2):], \
                   dep_distance[-len(doc_sent2):], sentence_length_n_char, sentence_length_n_words, \
                   sentence_mean_word_length, sentence_lexical_density_tokens, tf[-len(doc_sent2):], \
                   idf[-len(doc_sent2):], tf_idf[-len(doc_sent2):], lex_overlap[-len(doc_sent2):], \
                   pron_det_to_noun_ratio[-len(doc_sent2):], voice[-len(doc_sent2):], no_of_phrases[-len(doc_sent2):]

        # case 3: sentence was not split -> return entire length
        else:
            # quantitative indeces on sentence level
            sentence_length_n_char = [len(text) for _ in range(original_range)]  # incl. spaces
            sentence_length_n_words = [len(text.split()) for _ in range(original_range)]  # incl. punctuation
            sentence_mean_word_length = [sum(n_char) / len(text.split()) for _ in range(original_range)]  # in n char
            # percentage of pos tags that are content word tags:
            sentence_lexical_density_tokens = [len([p for p in pos
                                                    if p in ['ADJ', 'ADV', 'VERB', 'PROPN', 'NOUN']]) / len(pos)
                                               for _ in range(original_range)]

            return token_ids, tokens, lemmas, n_char, pos, simplified_pos, lemma_freqs, NE_iob, NE_label, content_word, \
                   technical_term, has_more_freq_synonym_or_homonym, morph_tags_Case, morph_tags_Gender, \
                   morph_tags_Number, morph_tags_Person, morph_tags_PronType, morph_tags_Mood, morph_tags_Tense, \
                   morph_tags_VerbForm, morph_tags_Definite, morph_tags_PunctType, morph_tags_Degree, morph_tags_Abbr, \
                   morph_tags_Poss, morph_tags_Prefix, morph_tags_Reflex, synt_surprisal, deps, rights, lefts, n_rights, n_lefts, \
                   dep_distance, sentence_length_n_char, sentence_length_n_words, sentence_mean_word_length, \
                   sentence_lexical_density_tokens, tf, idf, tf_idf, lex_overlap, pron_det_to_noun_ratio, voice, no_of_phrases


    def get_tf_score(self, sentence, term):
        """TF: score: how often does the word appear in the sentence?
        input: term must be lemma!"""
        count = 0
        wordlist = [token.lemma_ for token in sentence]
        for word in wordlist:
            if word == term:
                count += 1
        return count

    def get_lexical_overlap(self, text_id, sentence):
        """sentence = the first sentence in the tuple is the sentence we're looking at here"""
        sentence_pairs = self.get_sentence_pairs(text_id)

        sent_token_overlap = []
        for sent_id, pair in enumerate(sentence_pairs):  # pair = tuple of two subsequent sentences

            # lexical_overlap_in_sent[sent_id] = []
            if pair[0] == sentence:
                if pair[1] == 'NA':  # last entry
                    # get list of sentence lemmas
                    s1_lemmas = [token.lemma_ for idx, token in enumerate(self.nlp(pair[0])) if
                                 token.text not in self.STOP_CHARS]
                    s2_lemmas = ["NA" for idx, token in enumerate(self.nlp(pair[0])) if
                                 token.text not in self.STOP_CHARS]

                else:
                    # get list of sentence lemmas
                    s1_lemmas = [token.lemma_ for idx, token in enumerate(self.nlp(pair[0])) if
                                 token.text not in self.STOP_CHARS]
                    s2_lemmas = [token.lemma_ for idx, token in enumerate(self.nlp(pair[1])) if
                                 token.text not in self.STOP_CHARS]

                # iterate over tokens of sent, see if there is any overlap with next sentence
                token_overlap = []

                for lemma in s1_lemmas:
                    if lemma in s2_lemmas:
                        token_overlap.append(1)
                    else:
                        token_overlap.append(0)

                return token_overlap

    def get_lexical_overlap_old(self, text_id, screen_id):
        """
        Does the lemma appear in the next sentence? (0, 1)
        Same preprocessing as in dependency parser to get the same list length
        """
        # get list of tuples List(Tuple(str, str)) of the sentence and subsequent sentence for each sentence in the text
        sentence_pairs = self.get_sentence_pairs_with_index(text_id)

        lex_overlap_in_screens = {}
        for screen_idx, screen in sentence_pairs.items():  # screen = list of tuples from the screens (adjacent sents)
            lex_overlap_in_screens[screen_idx] = []

            for sent_id, pair in enumerate(screen):  # pair = tuple of two subsequent sentences

                if pair[1] == 'NA':

                    # get list of sentence lemmas
                    s1_lemmas = [token.lemma_ for idx, token in enumerate(self.nlp(pair[0])) if
                                 token.text not in self.STOP_CHARS]
                    s2_lemmas = ["NA" for idx, token in enumerate(self.nlp(pair[0])) if
                                 token.text not in self.STOP_CHARS]

                else:
                    # get list of sentence lemmas
                    s1_lemmas = [token.lemma_ for idx, token in enumerate(self.nlp(pair[0])) if
                                 token.text not in self.STOP_CHARS]
                    s2_lemmas = [token.lemma_ for idx, token in enumerate(self.nlp(pair[1])) if
                                 token.text not in self.STOP_CHARS]

                # iterate over tokens of sent, see if there is any overlap with next sentence
                token_overlap = []

                for lemma in s1_lemmas:
                    if lemma in s2_lemmas:
                        token_overlap.append(1)
                    else:
                        token_overlap.append(0)
                lex_overlap_in_screens[screen_idx].append(token_overlap)

        return lex_overlap_in_screens[screen_id]

    def get_NE_tags_DE(self, text, text_id, screen_id):
        """For InDiCo only:
        Named entity tags aren't supported in trf model -> for German texts, they need to be generated with
        large model instead of trf. Then do the same process as above.
        """

        # construct spacy doc object from sentence
        doc_ent = self.nlp_lg(text)  # get words with [t.text for t in doc], text variable is the sentence
        # NE_tokens = [[] for _ in len(original_range)]
        NE_tokens = []
        NE_iob = []
        NE_label = []

        adjusted_id = -1
        for idx, token in enumerate(doc_ent):
            if token.text not in self.MOVE:
                NE_tokens.append(token.text)
                NE_iob.append(token.ent_iob_)
                if token.ent_type == "":
                    NE_label.append("NA")
                else:
                    NE_label.append(token.ent_type_)

        # take care of segmentaiton error in text 15, screen 2: '60', '-', '75' (lg) vs. '60-75' (trf)
        if text_id == 15 and screen_id == 2:
            for i in range(len(NE_tokens)):
                if NE_tokens[i:i + 3] == ['60', '-', '75']:
                    NE_tokens[i:i + 3] = ['60-75']
                    NE_iob[i:i + 3] = ['O']
                    NE_label[i:i + 3] = ['O']

        return NE_iob, NE_label

    def get_simplified_pos(self, word):
        """From Reich et al (2022):"
            we additionally compute Simplified PoS-tags by collapsing the PoS of all function words to a single category
            FUNC, merging proper nouns and common nouns as well as adjectives and adverbs to a category N and A,
            respectively. """
        if word.pos_ in ["NOUN", "PROPN"]:
            simplified_pos = "N"
        elif word.pos_ in ["ADJ", "ADV"]:
            simplified_pos = "A"
        elif word.pos_ == "VERB":
            simplified_pos = "VERB"
        else:
            simplified_pos = "FUNC"

        return simplified_pos

    def get_morph_tags(self, token_morph_features):
        """take a dictionary of morphological tags, and extend with NA values for morphological classes that don't correspond
            with this POS.
            I manually checked spacy DE for interesting classes, then looked at a few EN sentences
            to compare and hopefully got all the important ones. In E, PunctType exists, while it doesnt for german."""

        # list of morphological classes that could possibly occur
        morph_classes = [
            "Case",  # Nom, Dat, Acc, ...
            "Gender",  # Masc, Fem, Neut
            "Number",  # Sing, Plur
            "Person",  # 1, 2, 3
            "PronType",  # Prs, Rel, Dem,
            "Mood",  # Ind, Sub
            "Tense",  # Past, Present
            "VerbForm",  # Fin, Part, Inf
            "Definite",  # Def, Ind
            "Abbr",
            "Degree",
            "Poss",
            "Prefix",
            "Reflex",
            "PunctType",  # only in en model

        ]
        # create extended dictionary with with all morphological classes as values and  original values for keys
        # or "NA" as value if morphological class wasn't present
        extended_tags_dict = {c: token_morph_features[c] if c in token_morph_features else "NA" for c in morph_classes}

        return extended_tags_dict

    def get_synonyms_homonyms(self, word):
        """
            For a word/lemma look up synsets the word appear in and make a list of all words that occur in those synsets.
            They are the possible homonyms/synonyms (we can't tell which).
            If any of the words have a higher wordfreq than the original word, return True else False.
            """
        if args.SBSAT is True:
            language = 'en'
            synsets = wordnet.synsets(word)

            # get synonyms and homonyms
            synonyms_homonyms = []
            for synset in synsets:
                for lemma in synset.lemmas():
                    if lemma.name() != word:
                        synonyms_homonyms.append(lemma.name())

        elif args.InDiCo is True:
            language = 'de'
            # Initialize GermaNet
            gn = self.gn_path

            synonyms_homonyms = []
            # get possible synsets for word
            synsets = gn.get_synsets_by_orthform(word)
            # get list of words from all synsets the target word occured in
            for synset in synsets:
                for i in synset.lexunits:
                    if i.orthform != word:
                        synonyms_homonyms.append(i.orthform)

        # determine frequency
        original_word_freq = wordfreq.word_frequency(word, language, "large")
        syn_hom_freq = {s: wordfreq.word_frequency(s, language, "large") for s in synonyms_homonyms}

        decision = any(value > original_word_freq for value in syn_hom_freq.values())

        return decision

    def get_surprial(self, tokens):
        """Lexical surprisal: Calculate the surprisal of each token in a sequence by computing the negative
            log probability of the token given the preceding context. It measures how unexpected each token is given the
            context in which it occurs."""
        surprisals = []
        for i in range(1, len(tokens)):
            context = " ".join(tokens[:i])
            prob = self.nlp.vocab[context].prob
            surprisal = -1 * math.log2(prob)
            surprisals.append(surprisal)
            return surprisals

    def pronoun_noun_ratio(self, text_id: int, target_sentence: str, pos_tags: List[str]) -> List[float]:
        """Calculate the ratio between nouns/proper names in the target & preceding sentence and pronouns in the target 
        sentene and between nouns/proper nouns and determiners.
        Use this as a proxy for syntactic ambiguity: If there are many more nouns than determiners or pronouns,
        the syntactic relations might be ambiguous. I was planning to do this with KNG-Kongruenz first but Spacys
        Morph. tags aren't detailed/complete enough.
        input: target_sentence is the preprocessed sentence (but not the doc), pos_tags is the list of pos tags
        extracted from the sentence's doc object.
        T"""
        # context of preceding sentence: count nouns and proper names ---------------------------------------
        n_noun_tags = 0

        context_tags = []
        for current_sent, preceding_sent in self.get_sentence_pairs_preceding(text_id):
            if current_sent == target_sentence:
                context = preceding_sent
                context_doc = self.nlp(context)
                context_tags.extend([token.pos_ for token in context_doc])

        for tag in context_tags:
            if tag == "NOUN":
                n_noun_tags += 1
            elif tag == "PROPN":
                n_noun_tags += 1

        # count number of tags in target sentence -----------------
        n_pron_tags = 0
        n_det_tags = 0

        for item in pos_tags:

            if item == 'PRON':
                n_pron_tags += 1
            elif item == 'DET':
                n_det_tags += 1
            elif item == 'NOUN':
                n_noun_tags += 1
            elif item == 'PROPN':
                n_noun_tags += 1

        # calculate ratio
        try:
            pron_noun_ratio = round((n_pron_tags/n_noun_tags), 4)
        except ZeroDivisionError:
            pron_noun_ratio = 0
        try:
            det_noun_ratio = round((n_det_tags / n_noun_tags), 4)
        except ZeroDivisionError:
            pron_noun_ratio = 0

        # assign values
        results = []
        for item in pos_tags:
            if item == "PRON":
                results.append(pron_noun_ratio)
            elif item == "DET":
                results.append(det_noun_ratio)
            else:
                results.append(0)

        return results

    def voice_detector(self, sentence_doc):
        """Detect whether the target sentence is active or passive voice.
        Input: spacy doc object of the preprocessed target sentence (variable name here is 'text').
        Note: For now, it only works if there is a passivised subject, e.g. Die Katze wird gejagt isn't recognized but
        Die Katze wird vom Hund gejagt is recognized."""

        if args.SBSAT is True:
            for token in sentence_doc:
                if token.dep_ == "nsubjpass":  # nsubjpass = 'nominal subject (passive)'
                    return "passive"
                elif token.dep_ == "agent":  # agent of an action, e.g. "The compound made [by our liver cells] ... "
                    return "passive"
            return "active"

        elif args.InDiCo is True:
            for token in sentence_doc:
                if token.dep_ == "sbp":  # sbp = passivized subject (PP)
                    return "passive"
            return "active"

    def sentence_preprocessing(self, text):
        """Preprocessing of raw sentence before creating doc object."""
        text = re.sub(r"www.vhb.org", r"VHB", text)

        # for parser, punctuation marks need to be white-space separated
        text = re.sub(r"(?<=[A-Za-z])(?=[(),.?!:;])", r" ", text)
        text = re.sub(r"(?<=[()])(?=[A-Za-z:.,])", r" ", text)
        # remove unwanted characters
        for character in self.IGNORE:
            text = text.replace(character, "")

        return(text)

    def load_text_screens(self) -> Dict[Tuple[int, int], str]:
        """Returns dictionary with (text, screen) as key and text file path as value.
        d = {(1, 1): 'data/InDiCo/raw/stimuli/daf_sentences_screens/text01_1_DAF.txt', ...} """
        text_file_dict = dict()
        text_files = sorted(glob.glob(self.texts_directory))
        for text_file in text_files:
            if args.InDiCo is True:
                text_file_dict[(int(text_file[-12:-10]), int(text_file[-9]))] = text_file
            elif args.SBSAT is True:
                text_file_dict[(int(text_file[-18:-16]), int(text_file[-15]))] = text_file
            else:
                pass

        return text_file_dict

    def load_full_texts(self) -> Dict[Tuple[int, int], str]:
        """Returns dictionary with text as key and text file path as value.
                    d = {(1, 1): 'data/InDiCo/raw/stimuli/daf_sentences_screens/text01_1_DAF.txt', ...} """
        text_file_dict = dict()
        text_files = sorted(glob.glob(self.full_texts_directory))
        for text_file in text_files:
            if args.InDiCo is True:
                if text_file[-10] == "0":
                    text_file_dict[int(text_file[-9])] = text_file
                else:
                    text_file_dict[int(text_file[-10:-8])] = text_file
            elif args.SBSAT is True:
                text_file_dict[int(text_file[-15:-14])] = text_file
            else:
                pass

        return text_file_dict

    def get_text_lists_with_index(self, text_id: int) -> Dict:
        """
        Get a dictionary with sentences per screen of a text.
        :param text_id: int
        :return: d_text = {1: [sent_1, sent2, ...], 2: [sent_1, sent_2, ...]}}
        """
        # get list of files names i.e. screens that belong to a text
        screens_from_text = [f[1] for f in self.text_files.items() if text_id == f[0][0]]

        d_text = {}
        # get list of sentences of a text
        screen_id = 0
        for screen in screens_from_text:
            screen_id += 1
            with open(screen, 'r') as infile:
                sents = infile.readlines()
                sents = [s.strip() for s in sents]
                d_text[screen_id] = sents

        return d_text

    def get_sentence_pairs_with_index(self, text_id):
        """Helper function for lexical overlap and semantic similarity.
        Input: list of sentences of a text.
        Output: List of tuples with preprocessed sentences. (corrent sentence, sentence following the current sentence)
        """

        screens_of_text = self.get_text_lists_with_index(text_id)
        text_tuples = {}
        for screen_id, screen in screens_of_text.items():

            # get list of tuples List(Tuple(str, str)) of the sentence and subsequent sentence for each sentence in the text
            sentence_pairs = []

            # iterate over sentences
            for i, text in enumerate(screens_of_text[screen_id]):  # for sentence in text ...
                tuple = []
                # preprocessing: same as in dependencyparser
                text = self.sentence_preprocessing(text)

                # last sentence -> NA as second tuple element
                if i == len(screens_of_text[screen_id]) - 1:
                    sentence_pairs.append((text, "NA"))
                # not last sentence: adjacent sentence als second tuple element
                else:
                    sentence_pairs.append((text, self.sentence_preprocessing(screens_of_text[screen_id][i + 1])))

            text_tuples[screen_id] = sentence_pairs

        # replace NA
        d_without_NA = {
            page_id: [(s1, text_tuples[page_id + 1][0][0]) if s2 == "NA" and page_id + 1 in text_tuples else (s1, s2)
                      for i, (s1, s2) in enumerate(page)] for page_id, page in text_tuples.items()}

        return d_without_NA

    def get_sentence_pairs(self, text_id):
        """list of tuples with sentence pairs of text (current sentence, following sentence)
        """
        return list(itertools.chain.from_iterable(self.get_sentence_pairs_with_index(text_id).values()))

    def get_sentence_pairs_preceding(self, text_id):
        """Get list of tuples with sentence pairs of text (current sentence, preceding sentence).
        Note that this is the other way around than in get_sentence_pairs
        here it's: []"""

        sentence_pairs = self.get_sentence_pairs(text_id) # (current sentence, following sentence

        # switch tuples around
        new_sentence_pairs = []
        # first entry: switch ("sentence1", "sentence2") to ("sentence1", "NA")
        new_sentence_pairs.append((sentence_pairs[0][0], "NA"))
        # all other sentences (current sent, follwing sent) -> (following sent, current sent) except last: delete
        new_sentence_pairs.extend([(second, first) if second != "NA" else (first, "NA")
                                   for first, second in sentence_pairs])
        # remove last list item
        new_sentence_pairs = new_sentence_pairs[:-1]

        return new_sentence_pairs


class SurprisalScorer:
    """Score the surprise of a given sentence using pre-trained language models, specifically BERT and GPT-2.
    Implemented by Patrick Haller.
    Syntactic surprisal is a measure of the unexpectedness of a word or phrase in context, based on the probability
    distribution of the language model."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.load_model(model_name)
        self.tokenizer = self.get_tokenizer()
        self.score = self.get_scorer()
        self.STRIDE = 200

    def load_model(self, model_name):
        """Load a fitting model for the language of the dataset."""
        if args.InDiCo is True:
            if model_name == "bert":
                return BertForMaskedLM.from_pretrained("bert-base-german-cased")
            elif model_name == "gpt":
                return GPT2LMHeadModel.from_pretrained("benjamin/gerpt2-large")

        elif args.SBSAT is True:
            if model_name == "bert":
                return BertForMaskedLM.from_pretrained("bert-base-cased")

    def get_scorer(self):
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        if self.model_name == "bert":
            return self.score_bert
        elif self.model_name == "gpt":
            return self.score_gpt
        else:
            raise NotImplementedError

    def get_tokenizer(self):
        if args.InDiCo is True:
            if self.model_name == "bert":
                return BertTokenizerFast.from_pretrained("bert-base-german-cased")
            elif self.model_name == "gpt":
                return GPT2TokenizerFast.from_pretrained("benjamin/gerpt2-large")
            else:
                raise NotImplementedError
        elif args.SBSAT is True:
            if self.model_name == "bert":
                return BertTokenizerFast.from_pretrained("bert-base-cased")
            else:
                raise NotImplementedError

    def score_gpt(self, sentence, BOS=True):
        with torch.no_grad():
            all_log_probs = torch.tensor([], device=self.model.device)
            offset_mapping = []
            start_ind = 0
            while True:
                encodings = self.tokenizer(
                    sentence[start_ind:],
                    max_length=1022,
                    truncation=True,
                    return_offsets_mapping=True,
                )
                if BOS:
                    tensor_input = torch.tensor(
                        [
                            [self.tokenizer.bos_token_id]
                            + encodings["input_ids"]
                            + [self.tokenizer.eos_token_id]
                        ],
                        device=self.model.device,
                    )
                else:
                    tensor_input = torch.tensor(
                        [encodings["input_ids"] + [self.tokenizer.eos_token_id]],
                        device=self.model.device,
                    )
                output = self.model(tensor_input, labels=tensor_input)
                shift_logits = output["logits"][..., :-1, :].contiguous()
                shift_labels = tensor_input[..., 1:].contiguous()
                log_probs = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                )
                assert torch.isclose(
                    torch.exp(sum(log_probs) / len(log_probs)),
                    torch.exp(output["loss"]),
                )
                offset = 0 if start_ind == 0 else self.STRIDE - 1
                all_log_probs = torch.cat([all_log_probs, log_probs[offset:-1]])
                offset_mapping.extend(
                    [
                        (i + start_ind, j + start_ind)
                        for i, j in encodings["offset_mapping"][offset:]
                    ]
                )
                if encodings["offset_mapping"][-1][1] + start_ind == len(sentence):
                    break
                start_ind += encodings["offset_mapping"][-self.STRIDE][1]
            return np.asarray(all_log_probs.cpu()), offset_mapping

    def score_bert(self, sentence):
        """Scores the surprisal of a given sentence using a BERT model. It does this by tokenizing the sentence,
        replacing each token with a [MASK] token in turn, and then computing the log probability of the correct
        token for each [MASK] token.
        The method returns an array of the negative log probabilities for each [MASK] token, as well as a list of
        offset mappings that maps the tokens in the sentence back to their original positions."""
        mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        with torch.no_grad():
            all_log_probs = []
            offset_mapping = []
            start_ind = 0
            while True:
                encodings = self.tokenizer(
                    sentence[start_ind:],
                    max_length=512,
                    truncation=True,
                    return_offsets_mapping=True,
                )
                tensor_input = torch.tensor(
                    [encodings["input_ids"]], device=self.model.device
                )
                mask_input = tensor_input.clone()
                offset = 1 if start_ind == 0 else self.STRIDE
                for i, word in enumerate(encodings["input_ids"][:-1]):
                    if i < offset:
                        continue
                    mask_input[:, i] = mask_id
                    output = self.model(mask_input, labels=tensor_input)
                    log_probs = torch.nn.functional.log_softmax(
                        output["logits"][:, i], dim=-1
                    ).squeeze(0)
                    all_log_probs.append(-log_probs[tensor_input[0, i]].item())
                    mask_input[:, i] = word

                offset_mapping.extend(
                    [
                        (i + start_ind, j + start_ind)
                        for i, j in encodings["offset_mapping"][offset:-1]
                    ]
                )
                if encodings["offset_mapping"][-2][1] + start_ind >= (
                        len(sentence) - 1
                ):
                    break
                start_ind += encodings["offset_mapping"][-self.STRIDE - 1][1]

            return all_log_probs, offset_mapping


def main():
    if args.SBSAT is True:
        a = AnnotatedTexts()
        a.write_annotations_to_csv()

    elif args.InDiCo is True:
        a = AnnotatedTexts()
        a.write_annotations_to_csv()

    else:
        print("Please specify which texts you'd like to annotate!")


if __name__ == "__main__":
    raise SystemExit(main())
