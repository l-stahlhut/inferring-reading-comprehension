# Inferring text comprehension from eye movements in reading using deep neural sequence models
Masters Thesis by Laura Stahlhut, University of Zurich, Institute for Computational Linguistics (2023), Grade: 6

This repo contains the code belonging to my masters thesis "Inferring text comprehension from eye movements in reading using deep neural sequence models".

**LinguisticEyeLSTM** is a sequential neural architecture that processes the sequence of fixations during reading and infers 
the level of text comprehension.

![](images/Model_architecture.png)
## Repo Structure
```
project
│   README.md
│   requirements.txt
│      
└───data
│   └───SB-SAT
│   │   ...
│   │
│   └───InDiCo
│       └───external
│       │    
│       └───raw
│       │    └───stimuli
│       │         ...
│       │    └───fixation
│       │         ...
│       └───interim
│       │   ...
│       │   
│       └───processed
│           indico_fix_lexfeats_final.csv
│   
└───src
│   annotate_texts.py
│   encode_data.py
│   extract_label.py
│   map_fix_to_roi.py
│   preprocess_fixations.py
│   reading_measures.py
│   reformat_lexical_surprisal_SBSAT.py
│   reformat__SBSAT_texts.py
│   txt_to_csv.py
│   
└───nn
│   └───sbsat_splits
│   └───indico_splits
│   │   └───data_s1_rm1_lf1
│   │   └───data_s1_rm1_lf0
│   │   └───data_s1_rm0_lf0_pos_cont
│   │   └───data_s1_rm0_lf0
│   │   └───data_s0_rm1_lf0
│   │   └───data_s0_rm0_lf1
│   │        └───book
│   │        └───subject
│   │        └───book-page
│   │  
│   └───model.py
│   └───results
│       └───InDiCo
│       │    └───data_s1_rm1_lf1
│       │    └───...
│       └───SBSAT
│
└───notebooks
```
## How to reproduce the experiments

### Clone this repository
```
git clone https://git@github.com:l-stahlhut/ET-Reading-Comprehension.git
```

### Download the data
**SB-SAT**: You can download the publicly available data here
```
git clone https://github.com/aeye-lab/etra-reading-comprehension
```
**InDiCo**: The Individual Differences Corpus is not publicly available. It can be reproduced by exporting the fixation 
report with the same columns that are in the SB-SAT dataset via SR Research's Dataviewer.

### Install packages
```
pip install -r requirements.txt
```
Follow the instructions below to preprocess the data and perform classification. <br>
The following image gives an overview over the **general workflow** which includes data preprocessing (fixation data, stimulus texts and target variable), data encoding and classification.

![](images/methods_overview_quer.png)

# 1. Data preprocessing
## Stimulus texts 
### Preprocessing
1. reformat SB-SAT texts such that they have the same format as the InDiCo stimulus texts (one file per screen, one line 
per sentence) 
2. reformat file for lexical surprisal
````angular2html
python3 reformat_SBSAT.py --SBSAT
python3 reformat_lexical_surprisal.py
````
### Linguistic annotation
Annotate the texts from both datasets:
````angular2html
python3 annotate_texts.py --InDiCo
python3 annotate_texts.py --SBSAT
````

## Fixation Data
The raw fixation data for SBSAT is publicly available (18sat_fixfinal.csv).
I extracted the raw fixation data from InDiCo directly from SR Research Dataviewer.

### Preprocessing the raw fixation data: <br>

SB-SAT: Preprocessing involves dropping unnecessary rows and columns and deleting fixations outside of interest areas.

````angular2html
python3 preprocess_fixations.py --SBSAT
````
InDiCo: Preprocessing involves data cleaning, choosing relevant columns and mapping interest areas from the level
of characters to the level of words. 
````angular2html
python3 preprocess_fixations.py --InDiCo
````
### Calculating Reading measures and merging fixation reports with lexical features
The following script calculates reading measures and merges the fixation report and the lexical features after fixing
alignment problems. 

````angular2html
python3 reading_measures.py --InDiCo
python3 reading_measures.py --SBSAT
````
## Target variable
The label for the SB-SAT dataset gets extracted from the fixation report while the label for the InDiCo dataset gets 
extracted from the results folder of the experiment. 
````angular2html
python3 src/extract_label.py --SBSAT
python3 src/extract_label.py --InDiCo
````

# 2. Encoding the data
Encode the data by specifying the dataset and the desired features, for example: 
--s1_rm1_lf1
````angular2html
python3 src/encode_data.py --SBSAT --s1_rm1_lf1 
python3 src/encode_data.py --InDiCo --s1_rm1_lf0
````
Please refer to the code to see which flag refers to which version of the features.

# 3. Classification
Train the different model variations, e.g. for the SB-SAT dataset with all features:
````angular2html
python3 nn/model.py --GPU 0 --s1_rm1_lf1 --SBSAT --save_dir nn/results --weights_dir  nn/weights
````
This saves the results and model weights. The flags for the data variations of the dataset can be seen in the script. 
The evaluation of the model (Ahn (2020) baseline implemented by Reich et al.(2022)) is included in this script.


# Datasets
- **InDiCo**: Individual Differences Corpus, compiled at the University of Zurich (German stimuli)
- **SB-SAT**: Stony Brook SAT reading fixation dataset (English stimuli) 

## Sources: 
- SB-SAT data: Ahn et al. (2020): https://github.com/ahnchive/SB-SAT
- Transforming interest areas: https://github.com/l-stahlhut/indiff-preprocessing/blob/main/parsing/map_fix_to_roi.py
- The preprocessing code is based on the InDiCo Preprocessing code (2022): https://github.com/l-stahlhut/individual-differences/tree/main/src (annotate_texts.py, reading_measures.py)
- The code for the data encoding and the model is based on Reich et al. (2022): https://github.com/aeye-lab/etra-reading-comprehension (model.py, 
 generate_text_sequence_splits.py) 
