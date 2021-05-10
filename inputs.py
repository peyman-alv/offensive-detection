import numpy as np

from transformers import AutoTokenizer
from likelihood import LikelihoodCalculator
from tqdm import tqdm
from config import *
from data import get_primary_data



def calcualte_model_io(apply_likelihood: bool, formula_name: str, max_len: int, n_classes: int) -> tuple:
    # one-hot encoder for representing the labels
    one_hot_encoding = lambda labels, num_classes: np.squeeze(np.eye(num_classes)[labels.reshape(-1)])

    train, test = get_primary_data()

    Xtrain = _encode_data(train[TEXT].tolist(), apply_likelihood, formula_name, max_len)
    Xtest  = _encode_data(test[TEXT].tolist(), apply_likelihood, formula_name, max_len)
    Ytrain = one_hot_encoding(train[LABEL].to_numpy(), num_classes=n_classes)
    Ytest  = one_hot_encoding(test[LABEL].to_numpy(), num_classes=n_classes)

    return Xtrain, Ytrain, Xtest, Ytest


def _encode_data(texts: list, apply_likelihood: bool, formula_name: str, max_len: int) -> np.array:
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    likelihood_calculator = LikelihoodCalculator(tokenizer, formula_name)

    data = list()
    for text in tqdm(texts):
        encoded = tokenizer.encode_plus(
            text=text,                  # the sentence to be encoded
            add_special_tokens=True,    # Add [CLS] and [SEP]
            max_length=max_len,         # truncates if len(s) > max_length
            padding='max_length',       # pads to the right by default
            truncation=True,              
            return_token_type_ids=True,
            return_attention_mask=True,
        )

        # input ids = token indices in the tokenizer's internal dict
        # token_type_ids = binary mask identifying different sequences in the model
        input_ids, token_type_ids = encoded["input_ids"], encoded["token_type_ids"]
        
        # attention_mask = binary mask indicating the positions of padded tokens so the model does not attend to them
        if apply_likelihood == False:
            attention_mask = encoded["attention_mask"]
        else:
            attention_mask = likelihood_calculator.return_attention_mask(input_ids)

        data.append([input_ids, attention_mask, token_type_ids])
       
    return np.asarray(data)