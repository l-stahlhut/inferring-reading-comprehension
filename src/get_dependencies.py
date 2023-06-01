"""
Dependenceis in json file abspeichern (1 file pro screen)
Damit ich sie manuell überprüfen kann -> zu lang für Bindestriche etc.
"""

import os
import json
import spacy
import argparse
import re

parser = argparse.ArgumentParser(
    prog='annotate_texts',
    description='Annotate Eye Tracking stimulus texts (SB-SAT, InDiCo) with several linguistic '
                'categories (Dependency, lexical frequency and many more).')
parser.add_argument('--SBSAT', action='store_true', help='English version.')
parser.add_argument('--InDiCo', action='store_true', help='German version.')
args = parser.parse_args()

def get_paths(path):
    filenames = [i for i in os.listdir(path) if i.endswith('.txt')]
    return filenames

def dep_to_dict(file_path):
    D = DependencyParser()
    # return tuple with original and parsed line for a file
    sentences_info = []
    with open(file_path, 'r') as infile:
        lines = infile.readlines()
        lines = [l.rstrip() for l in lines]  # list of sentences
        parsed = [D.parse_dependency(line) for line in lines]  # list of parsed sentences
        for l, p in zip(lines, parsed):
            sentence_dict = {"sentence": l}
            deps = p[0]
            n_rights = p[1]
            rights = p[2]
            n_lefts = p[3]
            lefts = p[4]
            dep_distance = p[5]
            deps_token = p[6]
            synt_surprisal = p[7]
            # create a list of dictionaries with parsed info per token
            words_dictionaries = []
            count = 0
            for i in range(len(deps_token)):
                # dictionary for one token
                word_d = {
                    deps_token[count]: {
                        "deps": deps[count],
                        "n_rights": n_rights[count],
                        "rights": rights[count],
                        "n_lefts": n_lefts[count],
                        "lefts": lefts[count],
                        "dep_distance": dep_distance[count],
                        "synt_surprisal": synt_surprisal[count],
                    }
                }
                count = count + 1
                words_dictionaries.append(word_d)

            sentence_dict["parsed"] = words_dictionaries

            sentences_info.append(sentence_dict)
    return sentences_info

def write_dict_to_json(filename, dictionary):
    with open(filename, 'w') as fp:
        json.dump(dictionary, fp, indent=4, ensure_ascii=False)


class DependencyParser:
    def __init__(self):
        # self.nlp = spacy.load("de_core_news_lg")
        if args.SBSAT is True:
            self.nlp = spacy.load("en_core_web_trf")   # load english model
        elif args.InDiCo is True:
            self.nlp = spacy.load("de_dep_news_trf")    # load german model
        self.IGNORE = "””“„'"
        self.MOVE = ["(", ")", ",", ".", "?", "!", ":", ";"]
        self.nlp.tokenizer = spacy.tokenizer.Tokenizer(
            self.nlp.vocab,
            token_match=re.compile(r"\S+").match,
        )
        self.tagger = self.nlp.get_pipe("tagger")

    def parse_dependency(self, text) -> Tuple[List[Any], ...]:
        # create space before commas
        original_range = len(text.split())
        text = re.sub(r"www.vhb.org", r"VHB", text)
        # for parser, punctuation marks need to be white-space separated
        text = re.sub(r"(?<=[A-Za-z])(?=[(),.?!:;])", r" ", text)
        text = re.sub(r"(?<=[()])(?=[A-Za-z:.,])", r" ", text)
        for character in self.IGNORE:
            text = text.replace(character, "")
        doc = self.nlp(text)
        pos_pred = self.tagger.model.predict([doc])
        deps_token = [[] for _ in range(original_range)]
        n_rights = [[] for _ in range(original_range)]
        n_lefts = [[] for _ in range(original_range)]
        rights = [[] for _ in range(original_range)]
        lefts = [[] for _ in range(original_range)]
        deps = [[] for _ in range(original_range)]
        dep_distance = [[] for _ in range(original_range)]
        synt_surprisal = [[] for _ in range(original_range)] # todo adjust for sbsat,

        adjusted_id = -1  # todo why?
        for idx, token in enumerate(doc):
            if token.text not in self.MOVE:
                adjusted_id += 1
                deps_token[adjusted_id] = token.text
                deps[adjusted_id] = token.dep_
                rights[adjusted_id] = [str(item) for item in token.rights]
                lefts[adjusted_id] = [str(item) for item in token.lefts]
                n_rights[adjusted_id] = token.n_rights
                n_lefts[adjusted_id] = token.n_lefts
                dep_distance[adjusted_id] = token.i - token.head.i
                synt_surprisal[adjusted_id] = -log(max(softmax(pos_pred[0][idx]))) # todo understand what exactly this does + adjust for sbsat

        return deps, n_rights, rights, n_lefts, lefts, dep_distance, deps_token, synt_surprisal


def main():
    #path_in = '/Users/laurastahlhut/Documents/Jobs/HA_Lena/individual-differences/stimuli/daf_sentences_screens'
    #path_out = '/Users/laurastahlhut/Documents/Jobs/HA_Lena/individual-differences/stimuli/dependencies'
    #path_out = 'stimuli/dependencies'
    path_in = 'data/SB-SAT/interim/stimuli/sbsat_sentences_screens'
    path_out = 'data/SB-SAT/interim/stimuli/dependencies'

    for f in get_paths(path_in):
        outfile_name = os.path.join(path_out, f.rstrip(".txt") + "_dependencies.json")
        write_dict_to_json(outfile_name, dep_to_dict(os.path.join(path_in, f)))



if __name__ == '__main__':
    main()
