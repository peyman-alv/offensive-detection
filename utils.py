import json
from config import THRESHOLD



def save_dictionary(dictionary: dict, filepath: str) -> None:
    with open(filepath, "w") as fp:
        json.dump(dictionary, fp, ensure_ascii=False, indent=4)


def load_dictionary(filepath: str) -> dict:
    with open(filepath) as fp:
        dictionary = json.load(fp)
    return dictionary 


def add_likelihood(token: str, likelihood: dict, unmask=1, mask=0) -> float:
    if token == SPECIAL_TOKENS["PAD"]:
        return mask
    else:
        return likelihood.get(unmask + likelihood[token], unmask)


def add_with_thresholding(token: str, likelihood: dict, unmask=1, mask=0) -> float:
    if token == SPECIAL_TOKENS["PAD"]:
        return mask
    elif token in likelihood and likelihood[token] >= THRESHOLD:
        return unmask + (likelihood[token] - threshold)
    else:
        return unmask