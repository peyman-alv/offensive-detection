import os
import json
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from utils import *
from data import get_training_primary_meta_data
from config import *



class LikelihoodCalculator(object):
    def __init__(self, tokenizer, formula_name: str) -> None:
        self.tokenizer = tokenizer
        self._masking_method = self.__get_masking_method(formula_name)
        self._likelihood = self.__get_likelihood_dict()

    
    def __get_masking_method(self, formula_name: str):
        assert (formula_name in POSSIBLE_METHODS)
        methods = {POSSIBLE_METHODS[0]: add_likelihood, POSSIBLE_METHODS[1]: add_with_thresholding}
        return methods[formula_name]
        

    def __get_likelihood_dict(self) -> dict:
        filepath = os.path.join(SAVED_PATH, LIKELIHOOD_FILENAME)
        if os.path.isfile(filepath):
            return load_dictionary(filepath=filepath)
        else:
            return self.__calculate_likelihood_dict(filepath)

    
    def __calculate_likelihood_dict(self, filepath: str) -> dict:
        all_dataframes = get_training_primary_meta_data()
        X, Y = all_dataframes[TEXT].tolist(), all_dataframes[LABEL].tolist()

        vectorizer = TfidfVectorizer(tokenizer=self.tokenizer.tokenize, vocabulary=self.tokenizer.get_vocab())
        X = vectorizer.fit_transform(X).toarray()

        mnb = MultinomialNB()
        mnb.fit(X, Y)
        mnb_likelihood = self.__extract_likelihood(mnb, vectorizer)
        save_dictionary(dictionary=mnb_likelihood, filepath=filepath)
        
        return mnb_likelihood

        
    def __extract_likelihood(self, model, vectorizer) -> dict:
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        likelihood_value = sigmoid(model.feature_log_prob_[1, :] - model.feature_log_prob_[0, :])
        unsorted = {token: likelihood_value[index] for token, index in vectorizer.vocabulary_.items()}
        return {k: v for k, v in sorted(unsorted.items(), key=lambda item: item[1], reverse=True)}


    def return_attention_mask(self, input_ids: list) -> list:
        attention_mask = list()
        for idx_token in input_ids:
            token = self.tokenizer.convert_ids_to_tokens(idx_token)
            attention_mask.append(self._masking_method(token, self._likelihood))
        return attention_mask