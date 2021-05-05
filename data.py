import os
import re
import pandas as pd
import numpy as np 
import emoji
import wordsegment

from config import *

wordsegment.load()



def get_primary_data() -> tuple:
    # read 'OLID 2019' training and testing files 
    train, test = _read_primary_files()
    # preprocess the texts
    train[TEXT] = train[TEXT].apply(str).apply(_preprocessing)
    test[TEXT]  = test[TEXT].apply(str).apply(_preprocessing)
    # map the labels to a numeric value
    train[LABEL] = train[LABEL].map(TARGET_MAP)
    test[LABEL]  = test[LABEL].map(TARGET_MAP)
    
    return train, test


def _read_primary_files() -> tuple:
    train = pd.read_csv(os.path.join(DATASET_PATH, PRIMARY_FILES[0]), sep="\t")
    train = train[list(COLUMN_MAP["OLID"].keys())]
    train = train.rename(columns=COLUMN_MAP["OLID"])

    test = pd.read_csv(os.path.join(DATASET_PATH, PRIMARY_FILES[1]), header=0, names=[ID, TEXT], sep="\t")
    test_labels = pd.read_csv(os.path.join(DATASET_PATH, PRIMARY_FILES[2]), header=None, names=[ID, LABEL])
    test[LABEL] = test_labels[LABEL]

    return train, test[[TEXT, LABEL]]


def get_meta_data() -> pd.DataFrame:
    # read HatEval 2019, TRAC-1 datasets
    meta_data = _read_meta_dataframes()
    # preprocess the texts
    meta_data[TEXT]  = meta_data[TEXT].apply(str).apply(_preprocessing)
    # map the labels to a numeric value
    meta_data[LABEL] = meta_data[LABEL].map(TARGET_MAP)
    
    return meta_data
    

def _read_meta_dataframes() -> pd.DataFrame:
    # meta dataframes includes HatEval 2019 & TRAC-1
    # just read files, related to their training set.
    hateval = pd.read_csv(os.path.join(DATASET_PATH, META_FILES[0]), sep="\t")
    hateval[LABEL] = hateval[HATE_SPEECH].apply(lambda hs: "OFF" if hs == 1 else "NOT")
    hateval = hateval[[TEXT, LABEL]]

    trac1 = pd.read_csv(os.path.join(DATASET_PATH, META_FILES[1]), header=None, names=COLUMN_MAP["TRAC-1"])
    trac1[LABEL] = trac1[LABEL].map(TRAGET_TRAC1_MAP)
    trac1 = trac1[[TEXT, LABEL]]
    
    return hateval.append(trac1, ignore_index=True)


def get_training_primary_meta_data() -> pd.DataFrame:
    olid, _ = get_primary_data()
    meta    = get_meta_data()
    
    dataframe_comb = olid.append(meta, ignore_index=True)
    dataframe_comb.drop_duplicates(subset=[TEXT, LABEL], ignore_index=True, inplace=True)
    dataframe_comb = dataframe_comb.sample(frac=1, random_state=RANDOM_SEED)
    return dataframe_comb.reset_index(drop=True)

    
def _preprocessing(text: str) -> str:
    # This function is adapted from 
    # https://github.com/wenliangdai/multi-task-offensive-language-detection/blob/master/data.py
    text = _emoji2word(text)
    text = _replace_rare_words(text)
    text = _remove_duplicate_usernames(text)
    text = _segment_hashtags(text)
    text = _remove_useless_punctuation(text)
    return text


def _emoji2word(text: str) -> str:
    # replace emojis with their text descriptor.
    return emoji.demojize(text)


def _replace_rare_words(text: str) -> str:
    # replace rare tokens with their synonyms.
    for token, synonym in RARE_WORDS.items():
        text = text.replace(token, synonym)
    return text


def _remove_duplicate_usernames(text: str) -> str:
    # remove unnecessary duplicate tokens.
    # 1. replace each token in format of @something into @USER ..
    id_pat = re.compile(r'@[\w_]+')
    text = id_pat.sub(r"@USER", text) if "@USER" not in text else text
    # 2. if there are multiple @USER in text, replace it with @USERS to reduce redundant
    if text.find('@USER') != text.rfind('@USER'):
        text = text.replace('@USER', '')
        text = '@USERS ' + text
    return text


def _segment_hashtags(text: str) -> str:
    # hashtag segmenting -> #AnExample -> an example.
    tokens = text.split(" ")
    for index, token in enumerate(tokens):
        if token.find("#") == 0:
            tokens[index] = " ".join(wordsegment.segment(token))
    return " ".join(tokens)


def _remove_useless_punctuation(text: str, replace_token=" ") -> str:
    # remove usless puntuation.
    for useless_punc in USELESS_PUNCS:
        text = text.replace(useless_punc, replace_token)
    return text
